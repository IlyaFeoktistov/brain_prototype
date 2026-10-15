from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import gc
import math

import torch

from brain.wiring import ProceduralProjection, ProceduralRows


# ==================================================================
# GPU-СИМУЛЯТОР СПАЙКОВОЙ СЕТИ
# ==================================================================
#
# Все нейроны всех зон живут в одном векторе:
#
#     [ auditory E | auditory I | association E | association I | ... ]
#
# Каждая проекция — фиксированное число входов на нейрон:
#
#     pre[post, k]     индекс пресинаптического нейрона
#     weight[post, k]
#
# Один шаг времени — gather спайков по индексам и сумма по k.
# Память O(N * K), а не O(N^2), поэтому мозг масштабируется линейно.


@dataclass
class Population:
    name: str
    start: int
    size: int

    @property
    def stop(self) -> int:
        return self.start + self.size

    @property
    def slice(self) -> slice:
        return slice(self.start, self.stop)


@dataclass
class Area:
    """
    Зона мозга: свои E и I нейроны, своё торможение. В зонах новой коры с
    типами тормозных клеток (cfg.interneurons) I — это PV, рядом SOM и VIP.
    """

    name: str
    title: str
    kind: str
    E: Population
    I: Population
    SOM: Population | None = None
    VIP: Population | None = None
    # Колонка по слоям (cfg.layers): слой -> {"E", "PV", "SOM", "VIP"}.
    # E, I, SOM, VIP выше — популяции слоя 2/3.
    layers: dict | None = None

    @property
    def populations(self) -> list[tuple[str, Population]]:
        """(метка, популяция) всех клеток зоны: «E», «I», «L4.E», «L4.SOM» …"""

        if self.layers is None:
            pops = [("E", self.E), ("I", self.I), ("SOM", self.SOM), ("VIP", self.VIP)]
            return [(label, pop) for label, pop in pops if pop is not None]

        out = []

        for layer, group in self.layers.items():
            for kind, pop in group.items():
                label = {"E": "E", "PV": "I"}.get(kind, kind) if layer == "L23" else f"{layer}.{kind}"
                out.append((label, pop))

        return out

    @property
    def excitatory(self) -> list[Population]:
        if self.layers is None:
            return [self.E]

        return [group["E"] for group in self.layers.values()]

    @property
    def inhibitory(self) -> list[Population]:
        if self.layers is None:
            return [pop for pop in (self.I, self.SOM, self.VIP) if pop is not None]

        return [pop for group in self.layers.values() for kind, pop in group.items() if kind != "E"]

    @property
    def cells(self) -> slice:
        """Все клетки зоны подряд."""

        pops = [pop for _, pop in self.populations]
        return slice(pops[0].start, pops[-1].stop)


class Projection:
    """
    pre -> post, fixed in-degree.

    source = "net"      пресинаптические — нейроны мозга
    source = "sensory"  пресинаптические — sensory-каналы

    Пластичная проекция хранит eligibility trace C[post, k]:
    STDP пишет в C, а вес меняется только dopamine-сигналом
    (Izhikevich 2007, «distal reward»).
    """

    def __init__(
        self,
        name: str,
        source: str,
        pre: torch.Tensor,
        post: Population,
        weight: torch.Tensor,
        plastic: bool = False,
        cap: float = 0.0,
        innate_max: float = 0.0,
    ):
        self.name = name
        self.source = source
        self.pre = pre
        self.post = post
        self.weight = weight
        self.plastic = plastic
        self.cap = cap
        self.innate_max = innate_max
        self.innate_mean = float(weight.mean()) if plastic else 0.0
        # Кто превращает след в вес: "dopamine" — ошибка предсказания
        # (reward), "acetylcholine" — бодрствование, запись нового в
        # гиппокамп с одного раза (encode).
        self.modulator = "dopamine"
        # Передачу по пути подавляет ацетилхолин (пресинаптически, Hasselmo
        # 1995): при записи нового гиппокамп не разгоняет сам себя.
        self.ach_suppressed = False
        self.eligibility = (
            torch.zeros_like(weight) if plastic else None
        )


class EventTable:
    """
    Синапсы, сгруппированные по нейрону, без выравнивания (CSR):
    perm — номера синапсов, отсортированные по нейрону; start/count —
    начало и длина строки нейрона в perm. Строку собирают на лету шириной
    width, лишние ячейки — разные фиктивные синапсы.

    Хабы — нейроны с ветвлением больше HUB_FACTOR × среднего (клетки CA1:
    ~600 клеток будят 13 тыс. клеток коры по 200 входов — ~4400 выходов):
    в общей ширине — только первые width синапсов, остаток строки («хвост»)
    собирается отдельно и только для выстреливших хабов, не больше
    hub_capacity за шаг. [замер] без хвоста ширина всех строк стала
    4672 вместо 448 — чтение втрое медленнее.

    [замер] прежняя таблица [нейронов, width] хранила каждую строку
    выровненной под самый разветвлённый нейрон: при 100 тыс. нейронов —
    ширина 384 при среднем 148, 20.5 Б на синапс из 39.7 (две таблицы).
    """

    HUB_FACTOR = 4

    def __init__(self, keys: torch.Tensor, n_slots: int, pad: int, dummies: int):
        device = keys.device
        counts = torch.bincount(keys, minlength=n_slots)
        ptr = torch.zeros(n_slots + 1, dtype=torch.int32, device=device)
        ptr[1:] = torch.cumsum(counts, 0).to(torch.int32)
        perm = torch.argsort(keys, stable=True).to(torch.int32)
        # Хотя бы одна ячейка: на пустую таблицу тоже смотрят индексы.
        perm = perm if len(perm) else torch.full((1,), pad, dtype=torch.int32, device=device)
        self.ptr = ptr
        self._setup(ptr[:-1], counts.to(torch.int32), perm, pad, dummies)

    def _setup(self, start, count, perm, pad: int, dummies: int) -> None:
        device = perm.device
        self.start, self.count, self.perm = start, count, perm
        self.pad, self.dummies = pad, dummies
        used = count[count > 0].float()
        mean = float(used.mean()) if len(used) else 1.0
        hub = count > max(64, int(self.HUB_FACTOR * mean))
        narrow = count[~hub]
        width = max(1, int(narrow.max())) if len(narrow) else 1
        # Ширина — с запасом до кратного 64: рост синапсов редко меняет
        # форму, и собранный шаг (compile_step) не пересобирается.
        self.width = -(-width // 64) * 64
        self.cols = torch.arange(self.width, device=device)
        self.is_hub = hub
        hubs = int(hub.sum())

        if hubs:
            tail = int(count[hub].max()) - self.width
            self.tail_width = -(-tail // 64) * 64
            # Сколько хабов за шаг получают хвост: четверть хабов (зона
            # хабов разрежена — на шаге стреляют проценты её клеток).
            self.hub_capacity = max(32, hubs // 4)
            self.tail_cols = torch.arange(self.tail_width, device=device)
        else:
            self.tail_width = self.hub_capacity = 0

    def events(self, neurons: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Синапсы строк neurons и пресинаптический нейрон каждого (плоско)."""

        neurons = neurons.long()
        start = self.start.index_select(0, neurons).long()
        count = self.count.index_select(0, neurons).long()
        position = (start[:, None] + self.cols).clamp(max=len(self.perm) - 1)
        synapses = self.perm.index_select(0, position.flatten()).view_as(position).long()
        dummy = self.pad + (neurons[:, None] * self.width + self.cols) % self.dummies
        synapses = torch.where(self.cols < count[:, None], synapses, dummy).flatten()
        pre = neurons[:, None].expand(-1, self.width).flatten()

        if not self.hub_capacity:
            return synapses, pre

        # Хвосты выстреливших хабов.
        which = torch.nonzero_static(
            self.is_hub.index_select(0, neurons), size=self.hub_capacity, fill_value=-1,
        ).squeeze(1)
        hubs = torch.where(which >= 0, neurons.index_select(0, which.clamp(min=0)), neurons[-1:].expand_as(which))
        h_start = self.start.index_select(0, hubs).long() + self.width
        h_count = self.count.index_select(0, hubs).long() - self.width
        h_count = torch.where(which >= 0, h_count, torch.zeros_like(h_count))
        position = (h_start[:, None] + self.tail_cols).clamp(max=len(self.perm) - 1)
        tail = self.perm.index_select(0, position.flatten()).view_as(position).long()
        dummy = self.pad + (hubs[:, None] * self.tail_width + self.tail_cols) % self.dummies
        tail = torch.where(self.tail_cols < h_count[:, None], tail, dummy).flatten()
        tail_pre = hubs[:, None].expand(-1, self.tail_width).flatten()

        return torch.cat([synapses, tail]), torch.cat([pre, tail_pre])

    def nbytes(self) -> int:
        return self.ptr.numel() * 4 + self.perm.numel() * 4

    TENSORS = ("ptr", "start", "count", "perm", "cols", "is_hub", "tail_cols")

    def refill(self, new: "EventTable") -> bool:
        """
        Переписать таблицу содержимым new на месте, если форма та же:
        записанный CUDA Graph смотрит на адреса — он остаётся годным.
        Форма другая — False (таблицу надо заменить, граф записать заново).
        """

        if (self.width, self.tail_width, self.hub_capacity) != (new.width, new.tail_width, new.hub_capacity):
            return False

        pairs = [(getattr(self, k, None), getattr(new, k, None)) for k in self.TENSORS]

        if any((a is None) != (b is None) or (a is not None and a.shape != b.shape) for a, b in pairs):
            return False

        for old, fresh in pairs:
            if old is not None:
                old.copy_(fresh)

        return True


class PrefixTable(EventTable):
    """
    Начала строк общей таблицы: пластичные синапсы лежат в массиве
    первыми, и устойчивая сортировка ставит их первыми в строке каждого
    нейрона — своя перестановка не нужна, только их число.
    [замер] отдельная перестановка — 4 Б на каждый пластичный синапс.
    """

    def __init__(self, table: EventTable, keys: torch.Tensor, n_slots: int, pad: int, dummies: int):
        counts = torch.bincount(keys, minlength=n_slots)
        self._setup(table.ptr[:-1], counts.to(torch.int32), table.perm, pad, dummies)

    def nbytes(self) -> int:
        return self.count.numel() * 4


def exact16(values: torch.Tensor) -> torch.Tensor:
    """Ближайшие значения, точно представимые в fp16 (тип — исходный)."""

    return values.to(torch.float16).to(values.dtype)


def pick_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"

    return torch.device(name)


class SpikingBrain:
    """
    Граф зон мозга (см. config.default_areas / default_pathways).

        ухо ──static──► auditory ──► association ◄──STDP──┐
                                       │   ▲ STDP          │
                                       │   └── E -> E      │
                                       ├──► prefrontal ────┘  (рабочая память)
                                       └──► motor ──► рот

    В каждой зоне E -> I -> E держит активность разреженной:
    отвечает малая часть нейронов, а не вся кора.

    Знания хранятся только в весах пластичных проекций.
    """

    DUMMY_NEURONS = 1024
    DUMMY_SYNAPSES = 16384

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = pick_device(cfg.device)

        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(cfg.seed)
        # Шум внутри CUDA Graph идёт из штатного генератора.
        torch.manual_seed(cfg.seed)

        self.time_ms = 0.0
        # Когда свежая память затухала в последний раз (forget).
        self.forgotten_at_ms = 0.0
        self.plasticity = True
        # Норадреналин: множитель пластичности от возбуждения (1 — спокойно).
        # Выставляет язык по эмоциям, во сне — 1.
        self.modulation = 1.0
        # Сон: проигрывание закрепляет свежие изменения (capture).
        self.consolidating = False
        # Наблюдатель (веб-интерфейс): нейроны, выстрелившие за CHUNK.
        self.on_chunk = None

        self._init_areas()
        self._init_neurons()
        self._init_projections()
        self._init_state()

        self.sensory_spikes = torch.zeros(
            self.n_receptors, device=self.device
        )

    # ==============================================================
    # SETUP
    # ==============================================================

    def _init_areas(self) -> None:
        cfg = self.cfg
        total_share = sum(a.share for a in cfg.areas)

        self.areas: dict[str, Area] = {}
        start = 0

        for spec in cfg.areas:
            n = max(10, int(cfg.total_neurons * spec.share / total_share))
            n_i = max(2, int(n * cfg.inhibitory_fraction))
            n_e = n - n_i

            if self._has_layers(spec.name):
                area, start = self._column(spec, n, start)
                self.areas[spec.name] = area
                continue

            E = Population(f"{spec.name}.E", start, n_e)

            if self._has_microcircuit(spec.name):
                # Типы тормозных по долям (ARCHITECTURE 10): I — это PV.
                I = Population(f"{spec.name}.I", E.stop, max(2, round(n_i * cfg.pv_share)))
                SOM = Population(f"{spec.name}.SOM", I.stop, max(2, round(n_i * cfg.som_share)))
                VIP = Population(f"{spec.name}.VIP", SOM.stop, max(1, round(n_i * cfg.vip_share)))
                self.areas[spec.name] = Area(spec.name, spec.title, spec.kind, E, I, SOM, VIP)
                start = VIP.stop
                continue

            I = Population(f"{spec.name}.I", E.stop, n_i)
            start = I.stop

            self.areas[spec.name] = Area(spec.name, spec.title, spec.kind, E, I)

        self.n = start
        # Органы-датчики: участки общего массива рецепторов подряд.
        self.sensors = [sensor for sensor in cfg.sensors() if sensor.area in self.areas]
        self.sensor_offsets = []
        offset = 0

        for sensor in self.sensors:
            self.sensor_offsets.append(offset)
            offset += sensor.size

        self.n_receptors = offset
        # Фиктивные нейроны [n, n_slots): никогда не стреляют, на них
        # указывают пустые ячейки таблиц синапсов. Их много, чтобы
        # пустые ячейки не дрались за один адрес (atomic-конфликты).
        self.n_slots = self.n + self.DUMMY_NEURONS

    def _has_microcircuit(self, name: str) -> bool:
        """Зона новой коры с типами тормозных клеток (cfg.interneurons или cfg.layers)."""

        cfg = self.cfg
        on = getattr(cfg, "interneurons", False) or getattr(cfg, "layers", False)
        return bool(on) and name in getattr(cfg, "microcircuit_areas", ())

    def _has_layers(self, name: str) -> bool:
        """Зона новой коры колонкой по слоям (cfg.layers)."""

        cfg = self.cfg
        return bool(getattr(cfg, "layers", False)) and name in getattr(cfg, "layered_areas", ())

    def _column_layers(self, name: str) -> list[tuple[str, int, int]]:
        """Слои колонки зоны (Potjans): у агранулярной — без L4."""

        cells = list(self.cfg.column_cells)

        if name in self.cfg.agranular_areas:
            cells = [c for c in cells if c[0] != "L4"]

        return cells

    def _column(self, spec, n: int, start: int) -> tuple[Area, int]:
        """
        Зона-колонка: n клеток по долям слоёв колонки 1 мм² (Potjans Table
        5), тормозные слоя — PV / SOM / VIP по долям слоя (остальные типы не
        строим). area.E / I / SOM / VIP — популяции L2/3.
        """

        cfg = self.cfg
        cells = self._column_layers(spec.name)
        total = sum(e + i for _, e, i in cells)
        shares = {layer: (pv, som, vip) for layer, pv, som, vip in cfg.layer_interneurons}
        layers = {}

        for layer, n_e, n_i in cells:
            group = {}
            sizes = [("E", max(4, round(n * n_e / total)))]
            inhibitory = n * n_i / total
            sizes += [(kind, max(1, round(inhibitory * share))) for kind, share in zip(("PV", "SOM", "VIP"), shares[layer])]

            for kind, size in sizes:
                group[kind] = Population(f"{spec.name}.{layer}.{kind}", start, size)
                start += size

            layers[layer] = group

        top = layers["L23"]
        area = Area(spec.name, spec.title, spec.kind, top["E"], top["PV"], top["SOM"], top["VIP"], layers)
        return area, start

    def _fill(self, values: dict[str, float]) -> torch.Tensor:
        """Параметр нейрона по типу популяции: cortex_E, memory_E, I (и SOM, VIP)."""

        out = torch.zeros(self.n_slots, device=self.device)

        for area in self.areas.values():
            for pop in area.excitatory:
                out[pop.slice] = values[f"{area.kind}_E"]

            for pop in area.inhibitory:
                out[pop.slice] = values["I"]

        return out

    def _init_interneurons(self) -> None:
        """
        Свойства PV, SOM и VIP — из измерений (Allen Cell Types; config:
        pv_tau_m …): своя мембрана и адаптация, у SOM — облегчающиеся входы
        от E (см. _step). Расстояние до порога — в _init_state (bias).
        """

        cfg = self.cfg
        self.is_som = torch.zeros(self.n_slots, dtype=torch.bool, device=self.device)
        # Тип клетки зон с типами тормозных: 0 E, 1 PV, 2 SOM, 3 VIP; −1 — прочие.
        self.cell_type = torch.full((self.n_slots,), -1, dtype=torch.long, device=self.device)

        for area in self.areas.values():
            if area.SOM is None:
                continue

            for _, group in self._typed_groups(area):
                for pop, tau, index in (
                    (group["PV"], cfg.pv_tau_m, cfg.pv_adaptation_index),
                    (group["SOM"], cfg.som_tau_m, cfg.som_adaptation_index),
                    (group["VIP"], cfg.vip_tau_m, cfg.vip_adaptation_index),
                ):
                    self.tau_m[pop.slice] = tau
                    self.adapt_b[pop.slice] = cfg.e_adapt_b * index / cfg.e_adaptation_index

                self.is_som[group["SOM"].slice] = True

                for number, kind in enumerate(("E", "PV", "SOM", "VIP")):
                    self.cell_type[group[kind].slice] = number

        self.has_som = bool(self.is_som.any())
        self.typed_pre = self.cell_type >= 0
        # Кратковременная пластичность по парам типов (cfg.column_stp): у
        # клетки-источника — U, восстановление и облегчение к каждому типу
        # получателя (столбцы E, PV, SOM, VIP).
        kinds = ("E", "PV", "SOM", "VIP")
        self.stp_U4 = torch.ones(self.n_slots, 4, device=self.device)
        self.stp_rec4 = torch.zeros(self.n_slots, 4, device=self.device)
        self.stp_fac4 = torch.zeros(self.n_slots, 4, device=self.device)

        for src, dst, U, rec, fac in getattr(cfg, "column_stp", ()):
            rows = self.cell_type == kinds.index(src)
            col = kinds.index(dst)
            self.stp_U4[rows, col] = U
            self.stp_rec4[rows, col] = cfg.dt_ms / rec
            self.stp_fac4[rows, col] = math.exp(-cfg.dt_ms / fac) if fac > 0 else 0.0

    @staticmethod
    def _typed_groups(area: Area) -> list[tuple[str, dict]]:
        """Слои зоны с типами тормозных: (слой, {E, PV, SOM, VIP})."""

        if area.layers is not None:
            return list(area.layers.items())

        return [("L23", {"E": area.E, "PV": area.I, "SOM": area.SOM, "VIP": area.VIP})]

    def _init_neurons(self) -> None:
        cfg = self.cfg

        self.tau_m = self._fill(
            {"cortex_E": cfg.e_tau_m, "memory_E": cfg.memory_tau_m, "I": cfg.i_tau_m}
        )
        self.tau_syn = self._fill(
            {"cortex_E": cfg.e_tau_syn, "memory_E": cfg.e_tau_syn, "I": cfg.i_tau_syn}
        )
        # Медленный NMDA-подобный ток есть только у рабочей памяти:
        # быстрый ток отвечает на текущее слово, медленный держит
        # след нескольких прошлых.
        self.slow_gain = self._fill(
            {"cortex_E": 0.0, "memory_E": cfg.memory_nmda_ratio, "I": 0.0}
        )
        # AdEx только у E коры. У I и рабочей памяти — LIF.
        self.delta_t = self._fill(
            {"cortex_E": cfg.e_delta_t, "memory_E": 0.0, "I": 0.0}
        )
        self.adapt_b = self._fill(
            {"cortex_E": cfg.e_adapt_b, "memory_E": 0.0, "I": 0.0}
        )

        # Потолок медленного тока (насыщение NMDA).
        self.slow_max = torch.full((self.n_slots,), float("inf"), device=self.device)

        for spec in cfg.areas:
            if spec.adapt_b is not None:
                self.adapt_b[self.areas[spec.name].E.slice] = spec.adapt_b
            if spec.slow_max is not None:
                self.slow_max[self.areas[spec.name].E.slice] = spec.slow_max
        self._init_interneurons()
        self.v_peak = self._fill(
            {"cortex_E": 0.0, "memory_E": cfg.v_threshold, "I": cfg.v_threshold}
        )
        self.v_peak[self.n:] = float("inf")
        self.tau_m[self.n:] = 1.0
        self.tau_syn[self.n:] = 1.0
        self.is_adex = self.delta_t > 0
        # Ветка LTD (пре после поста) есть у всех, кроме нейронов
        # рабочей памяти при ковариационном правиле.
        self.ltd_mask = self._fill(
            {
                "cortex_E": 1.0,
                "memory_E": 0.0 if cfg.context_rule == "covariance" else 1.0,
                "I": 1.0,
            }
        )
        self.memory_pops = [a.E for a in self.areas.values() if a.kind == "memory"]
        # Зоны с симметричным STDP внутри себя (CA3): номер зоны у её E,
        # 0 у остальных. Синапс симметричен, если пре и пост из одной такой зоны.
        self.symmetric_area = None

        for number, spec in enumerate(cfg.areas, 1):
            if spec.ee_symmetric:
                if self.symmetric_area is None:
                    self.symmetric_area = torch.zeros(self.n_slots, device=self.device)
                self.symmetric_area[self.areas[spec.name].E.slice] = number
        self.safe_delta_t = self.delta_t.clamp(min=1e-3)

        # Скачок синаптического тока при весе J даёт
        # интеграл PSP ≈ J мВ·τm независимо от τsyn.
        self.syn_scale = self.tau_m / self.tau_syn
        # Быстрый ток рабочей памяти ослаблен: она должна держать
        # прошлые слова, а не дублировать текущее.
        for spec in cfg.areas:
            if spec.kind == "memory":
                ratio = spec.ampa_ratio if spec.ampa_ratio is not None else cfg.memory_ampa_ratio
                self.syn_scale[self.areas[spec.name].E.slice] *= ratio

        dt = cfg.dt_ms
        self.decay_m = dt / self.tau_m
        self.decay_syn = torch.exp(-dt / self.tau_syn)
        self.decay_slow = math.exp(-dt / cfg.memory_tau_syn)
        self.decay_dend = math.exp(-dt / cfg.dendrite_tau_ms)
        # Своё затухание ветки у зоны (AreaSpec.dendrite_tau_ms): у рта —
        # быстрые события тонких веток (буква), а не NMDA-плато смысла.
        custom_dend = [a for a in cfg.areas if getattr(a, "dendrite_tau_ms", None) is not None]

        if custom_dend:
            per_cell = torch.full((self.n_slots,), self.decay_dend, device=self.device)

            for spec in custom_dend:
                per_cell[self.areas[spec.name].E.slice] = math.exp(-dt / spec.dendrite_tau_ms)

            self.decay_dend = per_cell.repeat_interleave(max(1, cfg.dendrite_branches))
        # Сколько плато ветки поднимает сому, мВ — по клеткам (своя у зоны).
        self.dendrite_soma = torch.full((self.n_slots,), float(cfg.dendrite_soma_mv), device=self.device)

        for spec in cfg.areas:
            if getattr(spec, "dendrite_soma_mv", None) is not None:
                self.dendrite_soma[self.areas[spec.name].E.slice] = spec.dendrite_soma_mv
        self.decay_pool = math.exp(-dt / cfg.pool_inhibition_tau_ms)
        self.pool_mask = torch.zeros(self.n_slots, device=self.device)
        if "association" in self.areas:
            self.pool_mask[self.areas["association"].E.slice] = 1.0
        # Дендритный вход: от E рабочей памяти к E смысловой коры.
        self.dendrite_pre = torch.zeros(self.n_slots, dtype=torch.bool, device=self.device)
        self.dendrite_post = torch.zeros(self.n_slots, dtype=torch.bool, device=self.device)
        for area in self.areas.values():
            if area.kind == "memory":
                self.dendrite_pre[area.E.slice] = True
        if "association" in self.areas:
            self.dendrite_post[self.areas["association"].E.slice] = True
        # Рот по буквам: какое слово говорится — дендритами рта.
        if getattr(cfg, "mouth_letters", False) and "motor" in self.areas:
            self.dendrite_post[self.areas["motor"].E.slice] = True
        self.slow_scale = self.slow_gain * self.tau_m / cfg.memory_tau_syn
        # Базовый контур: у кого истощаются синапсы и чьё торможение
        # растёт с частотой — E зон с обучаемыми связями внутри себя (там
        # ансамбли могут замкнуться: смысловая кора). Входные зоны (чувства,
        # слух) — проводники: [замер] истощение у клеток чувства, которые
        # стреляют, пока чувство держится, глушило путь «чувство → слова»
        # (тест «скука → скучно»: «мне человек»).
        self.cortex_e = torch.zeros(self.n_slots, device=self.device)
        # Чьё торможение растёт с частотой: по умолчанию те же клетки; у
        # цели — тоже (стреляет дольше всех — гаснет первой).
        self.inh_adapts = torch.zeros(self.n_slots, device=self.device)
        for spec in cfg.areas:
            if spec.kind == "cortex" and spec.ee_plastic:
                self.cortex_e[self.areas[spec.name].E.slice] = 1.0
            adapts = spec.inhibition_adapts
            if adapts if adapts is not None else (spec.kind == "cortex" and spec.ee_plastic):
                self.inh_adapts[self.areas[spec.name].E.slice] = 1.0
        self.stp_recover = dt / cfg.stp_tau_rec_ms
        self.decay_inh_gain = math.exp(-dt / cfg.inh_gain_tau_ms)
        # Затухание адаптации: у ядер может быть своё (медленное) время.
        self.decay_adapt = torch.full(
            (self.n_slots,), math.exp(-dt / cfg.e_adapt_tau), device=self.device
        )

        for spec in cfg.areas:
            if spec.adapt_tau is not None:
                E = self.areas[spec.name].E
                self.decay_adapt[E.slice] = math.exp(-dt / spec.adapt_tau)

        # Медленная следовая гиперполяризация (sAHP) — своя у зоны.
        self.slow_ahp_b = torch.zeros(self.n_slots, device=self.device)
        self.decay_slow_ahp = torch.ones(self.n_slots, device=self.device)

        for spec in cfg.areas:
            if getattr(spec, "slow_ahp_b", 0.0) > 0:
                E = self.areas[spec.name].E
                self.slow_ahp_b[E.slice] = spec.slow_ahp_b
                self.decay_slow_ahp[E.slice] = math.exp(-dt / spec.slow_ahp_tau)
        self.decay_plus = math.exp(-dt / cfg.stdp_tau_plus)
        # Ядро LTP со сдвигом: exp(-t/τ+) - exp(-t/τrise), нормировано
        # на пик 1. Одновременные спайки (одно слово) почти не
        # потенцируются, спайк ~1 слово назад — сильнее всего.
        rise = cfg.stdp_ltp_rise
        self.decay_rise = math.exp(-dt / rise) if rise > 0 else 0.0

        if rise > 0:
            tau = cfg.stdp_tau_plus
            peak_t = tau * rise / (tau - rise) * math.log(tau / rise)
            self.ltp_gain = 1.0 / (math.exp(-peak_t / tau) - math.exp(-peak_t / rise))
        else:
            self.ltp_gain = 1.0
        self.decay_minus = math.exp(-dt / cfg.stdp_tau_minus)

        # Своё окно STDP у зон (AreaSpec.stdp_*): затухание следов — на клетку.
        custom = [a for a in cfg.areas if getattr(a, "stdp_tau_plus", None) is not None]

        if custom:
            def per_cell(value: float) -> torch.Tensor:
                return torch.full((self.n_slots,), float(value), device=self.device)

            decay_plus, decay_rise = per_cell(self.decay_plus), per_cell(self.decay_rise)
            decay_minus, ltp_gain = per_cell(self.decay_minus), per_cell(self.ltp_gain)

            for spec in custom:
                E = self.areas[spec.name].E
                tau = spec.stdp_tau_plus
                rise = spec.stdp_ltp_rise or 0.0
                decay_plus[E.slice] = math.exp(-dt / tau)
                decay_rise[E.slice] = math.exp(-dt / rise) if rise > 0 else 0.0
                decay_minus[E.slice] = math.exp(-dt / (spec.stdp_tau_minus or cfg.stdp_tau_minus))

                if rise > 0:
                    peak_t = tau * rise / (tau - rise) * math.log(tau / rise)
                    ltp_gain[E.slice] = 1.0 / (math.exp(-peak_t / tau) - math.exp(-peak_t / rise))
                else:
                    ltp_gain[E.slice] = 1.0

            self.decay_plus, self.decay_rise = decay_plus, decay_rise
            self.decay_minus, self.ltp_gain = decay_minus, ltp_gain

        self.decay_eligibility = math.exp(-dt / cfg.eligibility_tau)
        # Число фоновых входных спайков за шаг ~ Poisson(λ).
        self.noise_mean = torch.full((self.n_slots,), cfg.background_rate_hz * dt / 1000.0, device=self.device)
        self.noise_gain = self._fill(
            {
                "cortex_E": cfg.background_weight,
                "memory_E": cfg.memory_background_weight,
                "I": cfg.background_weight,
            }
        )

        # Своя сила фона у зоны (AreaSpec.background) — у E-клеток.
        for spec in cfg.areas:
            if spec.background is not None:
                self.noise_gain[self.areas[spec.name].E.slice] = spec.background

        # Колонки (cfg.layers): внешний вход — остальной мозг, Пуассон K × 8 Гц
        # с PSP 0.15 мВ каждой клетке (Potjans Table 5) вместо слабого шума.
        for area in self.areas.values():
            if area.layers is None:
                continue

            external = {layer: (e, i) for layer, e, i in cfg.external_inputs}
            gaps = {"E": cfg.v_threshold - cfg.v_rest, "PV": cfg.pv_gap_mv, "SOM": cfg.som_gap_mv, "VIP": cfg.vip_gap_mv}
            charge = self._potjans_charge()

            for layer, group in area.layers.items():
                for kind, pop in group.items():
                    inputs = external[layer][0 if kind == "E" else 1]
                    self.noise_mean[pop.slice] = inputs * cfg.external_rate_hz * dt / 1000.0
                    # Заряд PSP (мВ·мс) = вес × τm клетки — как у Potjans
                    # относительно расстояния до порога.
                    tau = float(self.tau_m[pop.start])
                    gain = dict(zip(("E", "PV", "SOM", "VIP"), cfg.column_gain_external))[kind]
                    self.noise_gain[pop.slice] = gain * charge * gaps[kind] / cfg.potjans_gap_mv / tau

        self.noise_std = self.noise_mean.sqrt()
        # Постоянный ток (мВ): недостающие в малой зоне соседи колонки (см.
        # _wire_column) — средняя часть их входа.
        self.dc_mv = torch.zeros(self.n_slots, device=self.device)
        self.dc_inh_mv = torch.zeros(self.n_slots, device=self.device)
        # Торможение проводимостью — у клеток колонок (cfg.layers).
        self.conductance_cells = torch.zeros(self.n_slots, dtype=torch.bool, device=self.device)

        for area in self.areas.values():
            if area.layers is not None:
                self.conductance_cells[area.cells] = True

        self.conductance = bool(self.conductance_cells.any())

    def _topographic_pre(self, post: Population, k: int, pre_pop: Population, field: float) -> torch.Tensor:
        """
        Входы с круга на карте источника: клетки зон лежат сеткой
        ceil(√n) столбцов (как места слов в Concepts); клетка-получатель
        проецируется в ту же долю карты источника и берёт k случайных
        клеток круга площадью field · k вокруг этой точки.
        """

        def grid(size: int) -> tuple[int, int]:
            cols = max(1, math.ceil(math.sqrt(size)))
            return cols, max(1, math.ceil(size / cols))

        post_cols, post_rows = grid(post.size)
        pre_cols, pre_rows = grid(pre_pop.size)
        cell = torch.arange(post.size, device=self.device)
        cx = ((cell % post_cols).float() + 0.5) / post_cols * pre_cols
        cy = ((cell // post_cols).float() + 0.5) / post_rows * pre_rows
        radius = math.sqrt(field * k / math.pi)
        # Равномерно по кругу: r = R·√u, угол равномерный.
        r = radius * torch.rand((post.size, k), device=self.device, generator=self.generator).sqrt()
        angle = 2 * math.pi * torch.rand((post.size, k), device=self.device, generator=self.generator)
        x = (cx[:, None] + r * torch.cos(angle)).round().long().clamp(0, pre_cols - 1)
        y = (cy[:, None] + r * torch.sin(angle)).round().long().clamp(0, pre_rows - 1)

        return pre_pop.start + (y * pre_cols + x) % pre_pop.size

    def _random_pre(self, post: Population, k: int, pre_pop: Population) -> torch.Tensor:
        pre = torch.randint(
            pre_pop.start,
            pre_pop.stop,
            (post.size, k),
            device=self.device,
            generator=self.generator,
        )

        # Без автапсов.
        own = torch.arange(post.start, post.stop, device=self.device)[:, None]
        clash = pre == own
        pre[clash] = pre_pop.start + (pre[clash] - pre_pop.start + 1) % pre_pop.size

        return pre

    def _uniform(self, shape, low: float, high: float) -> torch.Tensor:
        # Значения, точные в fp16: вес при рождении хранится в fp16, и
        # «выученное = вес − вес при рождении» должно считаться без ошибки.
        return exact16(low + (high - low) * torch.rand(
            shape, device=self.device, generator=self.generator
        ))

    def _potjans_charge(self) -> float:
        """Заряд PSP 0.15 мВ клетки Potjans (τm 10, τsyn 0.5 мс), мВ·мс."""

        cfg = self.cfg
        tau_m, tau_syn, dt = cfg.potjans_tau_m, cfg.potjans_tau_syn, 0.01
        current, v, peak = tau_m / tau_syn, 0.0, 0.0

        for _ in range(int(10 * tau_m / dt)):
            v += (current - v) * dt / tau_m
            current *= math.exp(-dt / tau_syn)
            peak = max(peak, v)

        # Вес 1 даёт заряд τm; вес, дающий пик ee_psp_mv, — ee_psp_mv / пик.
        return cfg.ee_psp_mv / peak * tau_m

    def _psp_peak(self, pop: Population) -> float:
        """
        Пик PSP (мВ) клетки популяции от веса 1 — тем же шагом, что в _step:
        ток прыгает на τm/τsyn и спадает за τsyn, мембрана копит его за τm.
        Вес связи = измеренный PSP / пик.
        """

        tau_m = float(self.tau_m[pop.start])
        tau_syn = float(self.tau_syn[pop.start])
        dt = self.cfg.dt_ms
        current, v, peak = tau_m / tau_syn, 0.0, 0.0

        for _ in range(int(10 * max(tau_m, tau_syn) / dt) + 1):
            v += (current - v) * dt / tau_m
            current *= math.exp(-dt / tau_syn)
            peak = max(peak, v)

        return peak

    def _laminar(self, path, src: Area, dst: Area) -> tuple[Population, Population, Population, bool]:
        """
        Из какого слоя в какой идёт путь (ARCHITECTURE 10, «Связи между зонами
        по слоям»; cfg.pathway_directions): прямой — из L2/3 в L4 (E и PV —
        прямое торможение), у агранулярной зоны — в L2/3; обратный — из L5 на
        дендриты L2/3 (слой 1) и VIP; боковой — из L2/3 в L2/3; к ядрам — из
        L5. Без слоёв — как раньше: E → E и I.
        Возвращает (пре, пост, тормозные получатели, на дендриты ли).
        """

        direction = next(
            (d for s, t, d in getattr(self.cfg, "pathway_directions", ()) if (s, t) == (path.src, path.dst)),
            "lateral",
        )
        pre, post, inhibitory, top_down = src.E, dst.E, dst.I, False

        if src.layers is not None and (direction == "fb" or dst.layers is None and dst.SOM is None):
            # Обратные пути и выход к ядрам — из L5.
            pre = src.layers["L5"]["E"]

        if dst.layers is not None:
            if direction in ("ff", "thalamic") and "L4" in dst.layers:
                post, inhibitory = dst.layers["L4"]["E"], dst.layers["L4"]["PV"]
            elif direction == "fb":
                inhibitory, top_down = dst.VIP, True

        return pre, post, inhibitory, top_down

    def _place_maps(self) -> None:
        """
        Места клеток колонок-карт (cfg.map_areas): у V1 — карта сетчатки
        (место и ориентация, как у простых клеток), у следующих ступеней —
        вдоль поля по порядку (как _retinotopic_pre), у руки — по порядку
        клавиш. Все популяции зоны — в одних единицах: местные связи идут к
        ближайшим (_nearest_pre).
        """

        self.map_pos: dict[str, torch.Tensor] = {}

        for name in getattr(self.cfg, "map_areas", ()):
            area = self.areas.get(name)

            if area is None or area.layers is None:
                continue

            for _, pop in area.populations:
                if name == "visual" and self.eye_sensor is not None:
                    layout = self.field_layout.get(pop.name) or self._oriented_layout(self.eye_sensor, pop.size)
                    self.field_layout.setdefault(pop.name, layout)
                    rows, cols = layout[1], layout[2]
                else:
                    width = self.field_width if name != "hand" and self.field_width else 1.0
                    cols = (torch.arange(pop.size, device=self.device).float() + 0.5) * width / pop.size
                    rows = torch.zeros_like(cols)

                self.field_cols.setdefault(pop.name, cols)
                self.map_pos[pop.name] = torch.stack([rows, cols], 1)

    def _nearest_pre(self, post: Population, k: int, pre_pop: Population) -> torch.Tensor:
        """k ближайших по карте клеток источника (ничья — случайно)."""

        a, b = self.map_pos[post.name], self.map_pos[pre_pop.name]
        dist = torch.cdist(a, b)
        dist = dist + torch.rand(dist.shape, device=self.device, generator=self.generator) * 1e-3

        if post is pre_pop:
            dist.fill_diagonal_(float("inf"))

        return pre_pop.start + dist.topk(k, dim=1, largest=False).indices

    def _stp_steady(self, src: str, dst: str, hz: float) -> float:
        """
        Средняя сила синапса пары типов (к первому PSP) при ровной частоте hz
        — по cfg.column_stp (Tsodyks–Markram), для тока недостающих соседей.
        """

        params = {(a, b): (U, rec, fac) for a, b, U, rec, fac in getattr(self.cfg, "column_stp", ())}

        if (src, dst) not in params or hz <= 0:
            return 1.0

        U, rec, fac = params[(src, dst)]
        isi = 1000.0 / hz
        u, x = 0.0, 1.0

        for _ in range(200):
            u = u * (math.exp(-isi / fac) if fac > 0 else 0.0)
            x = 1.0 - (1.0 - x) * math.exp(-isi / rec)
            u = u + U * (1.0 - u)
            amplitude = u * x
            x -= u * x

        return amplitude / U

    def _wire_column(self, spec, area: Area) -> None:
        """
        Связи внутри колонки (ARCHITECTURE 10, «Слои колонки»): вероятность E/I
        между слоями — Potjans Table 5; тормозная доля — по типам в отношениях
        Campagnola 2022 так, что среднее по клеткам слоя равно числу Potjans.
        Входов k = вероятность × клеток-источников в зоне. Зона меньше колонки
        1 мм² — у пути k_data / k раз меньше входов: вес × √(k_data / k), а
        недостающая средняя часть — постоянный ток dc_mv по частотам полной
        покоя in vivo (van Albada, Helias & Diesmann 2015). E → E слоя 2/3 — обучаемая
        связь зоны (AreaSpec: k_ee заменяет вероятность, веса — её).
        """

        cfg = self.cfg
        order = [layer for layer, _, _ in cfg.column_cells]
        column = {layer: (e, i) for layer, e, i in cfg.column_cells}
        shares = {layer: dict(zip(("PV", "SOM", "VIP"), v)) for layer, *v in cfg.layer_interneurons}
        # Соседи стреляют, как в покое у бодрствующих (column_target_hz).
        rates = {layer: dict(zip(("E", "PV", "SOM", "VIP"), v)) for layer, *v in cfg.column_target_hz}
        ee_layer = dict(cfg.layer_ee_psp_mv)
        ei_psp = {layer: dict(zip(("PV", "SOM", "VIP"), v)) for layer, *v in cfg.layer_ei_psp_mv}
        ii_psp = {src: dict(zip(("E", "PV", "SOM", "VIP"), v)) for src, *v in cfg.inhibitory_psp_mv}
        to_e = {"PV": 0.35, "SOM": 0.23, "VIP": 0.06}
        from_e = {"PV": 0.42, "SOM": 0.30, "VIP": 0.20}
        among = {(src, dst): p for src, dst, p, _ in cfg.local_links if src != "E" and dst != "E"}
        types = ("PV", "SOM", "VIP")

        def potjans(src_layer, src_ei, dst_layer, dst_ei) -> float:
            row = 2 * order.index(dst_layer) + (dst_ei == "I")
            col = 2 * order.index(src_layer) + (src_ei == "I")
            return cfg.column_probability[row][col]

        def mean(weights, layer) -> float:
            share = shares[layer]
            return sum(share[t] * weights[t] for t in types) / sum(share[t] for t in types)

        def label(layer, kind) -> str:
            return ({"E": "E", "PV": "I"}.get(kind, kind)) if layer == "L23" else f"{layer}.{kind}"

        for dst_layer, dst_group in area.layers.items():
            for dst_kind, post in dst_group.items():
                for src_layer, src_group in area.layers.items():
                    for src_kind, pre in src_group.items():
                        src_ei = "E" if src_kind == "E" else "I"
                        dst_ei = "E" if dst_kind == "E" else "I"
                        p = potjans(src_layer, src_ei, dst_layer, dst_ei)

                        # Тормозная доля — по типам (среднее по слою = Potjans).
                        if src_ei == "E" and dst_ei == "I":
                            p *= from_e[dst_kind] / mean(from_e, dst_layer)
                        elif src_ei == "I" and dst_ei == "E":
                            p *= to_e[src_kind] / mean(to_e, src_layer)
                        elif src_ei == "I":
                            pair = among.get((src_kind, dst_kind), 0.0)
                            norm = sum(
                                shares[src_layer][a] * shares[dst_layer][b] * among.get((a, b), 0.0)
                                for a in types for b in types
                            ) / (sum(shares[src_layer].values()) * sum(shares[dst_layer].values()))
                            p *= pair / norm if norm > 0 else 0.0

                        if p <= 0:
                            continue

                        # Согласование с частотами in vivo (column_gain_*).
                        if src_ei == "E" and dst_ei == "I":
                            gain = cfg.column_gain_from_e[types.index(dst_kind)]
                        elif src_ei == "I" and dst_ei == "E":
                            gain = cfg.column_gain_to_e[types.index(src_kind)]
                        elif src_ei == "I":
                            gain = cfg.column_gain_ii
                        else:
                            gain = 1.0

                        # PSP пары, мВ.
                        if src_ei == "E" and dst_ei == "E":
                            if (src_layer, dst_layer) == ("L4", "L23"):
                                psp = cfg.l4_to_l23_psp_mv
                            elif src_layer == dst_layer:
                                psp = ee_layer.get(src_layer, cfg.ee_psp_mv)
                            else:
                                psp = cfg.ee_psp_mv
                        elif src_ei == "E":
                            psp = ei_psp[src_layer][dst_kind]
                        else:
                            psp = ii_psp[src_kind][dst_kind]

                        n_src = column[src_layer][0] if src_ei == "E" else column[src_layer][1] * shares[src_layer][src_kind]
                        k_data = p * n_src
                        k = min(round(p * pre.size), pre.size)
                        weight = gain * psp / self._psp_peak(post)
                        rate = rates[src_layer][src_kind] / 1000.0
                        tau = float(self.tau_m[post.start])

                        dc = self.dc_inh_mv if weight < 0 else self.dc_mv
                        # Синапсы соседей на частоте покоя уже ослабли / усилились.
                        weight_dc = weight * self._stp_steady(src_kind, dst_kind, rate * 1000.0)

                        if k < 1:
                            # Связь целиком — средним током.
                            dc[post.slice] += k_data * rate * weight_dc * tau
                            continue

                        root = math.sqrt(max(1.0, k_data / k))
                        dc[post.slice] += k_data * rate * weight_dc * tau * (1.0 - 1.0 / root)
                        name = f"{spec.name}: {label(src_layer, src_kind)}->{label(dst_layer, dst_kind)}"

                        pre_index = None

                        if spec.name in getattr(cfg, "map_areas", ()):
                            if spec.name == "visual" and (src_layer, src_kind, dst_layer, dst_kind) == ("L4", "E", "L23", "E"):
                                # Сложные клетки L2/3 собирают простые L4 своей
                                # ориентации рядом (Alonso & Martinez 1998).
                                pre_index = self._complex_pre(post, k, pre, 5.0)
                            else:
                                pre_index = self._nearest_pre(post, min(k, pre.size - (pre is post)), pre)
                            k = pre_index.shape[1]

                        if (src_layer, src_kind, dst_layer, dst_kind) == ("L23", "E", "L23", "E"):
                            # Обучаемая связь зоны: веса и правила — AreaSpec.
                            proj = self._connect(
                                name, pre, post, k, spec.ee_weight, plastic=spec.ee_plastic,
                                cap=spec.ee_cap, modulator=spec.ee_modulator,
                                ach_suppressed=spec.ee_ach_suppressed, pre_index=pre_index,
                            )
                            if proj is not None:
                                proj.scaled = getattr(spec, "ee_scaled", True)

                            if spec.columns > 1:
                                self._wire_columns(spec, area)
                            continue

                        proj = self._connect(name, pre, post, k, (weight * root, weight * root), pre_index=pre_index)

                        if proj is not None and src_kind == "SOM" and dst_kind == "E" and dst_layer in ("L23", "L5"):
                            # Мартинотти — на пучки дендритов в слое 1 (L2/3 и L5).
                            proj.dendrite = True

    def _wire_microcircuit(self, spec, area: Area) -> None:
        """
        Связи внутри зоны новой коры между E, PV, SOM и VIP — по таблице
        измерений (cfg.local_links; ARCHITECTURE 10). Входов k = вероятность
        × клеток-источников в досягаемости (2πσ² × плотность); в зоне их
        меньше — k срезается, вес растёт, общий вход тот же (масштаб). SOM → E
        — на ветки дендрита. Старые множители зоны (feedback — на вход E → PV,
        inhibition — на PV → E) сохранены: сверить с данными.
        """

        cfg = self.cfg
        pops = {"E": area.E, "PV": area.I, "SOM": area.SOM, "VIP": area.VIP}
        names = {"E": "E", "PV": "I", "SOM": "SOM", "VIP": "VIP"}
        density = {"E": cfg.density_e_mm2, "PV": cfg.density_i_mm2 * cfg.pv_share,
                   "SOM": cfg.density_i_mm2 * cfg.som_share, "VIP": cfg.density_i_mm2 * cfg.vip_share}

        for src, dst, probability, psp in cfg.local_links:
            if src == "E":
                sigma = cfg.sigma_ei_um
            elif dst == "E":
                sigma = cfg.sigma_ie_um
            else:
                sigma = cfg.sigma_ii_um
            reach = 2 * math.pi * (sigma / 1000.0) ** 2 * density[src]
            k_data = probability * reach
            pre, post = pops[src], pops[dst]
            k = max(1, min(round(k_data), pre.size))

            if k_data < 0.5:
                continue

            weight = psp / self._psp_peak(post) * (k_data / k)

            if (src, dst) == ("E", "PV"):
                weight *= spec.feedback
            elif (src, dst) == ("PV", "E"):
                weight *= spec.inhibition

            proj = self._connect(
                f"{spec.name}: {names[src]}->{names[dst]}", pre, post, k, (weight, weight),
            )

            if proj is not None and (src, dst) == ("SOM", "E"):
                # Мартинотти — на дальние дендриты: гасят плато веток.
                proj.dendrite = True

    def _connect(
        self,
        name: str,
        pre_pop: Population,
        post: Population,
        k: int,
        weight: tuple[float, float],
        plastic: bool = False,
        cap: float = 0.0,
        modulator: str = "dopamine",
        ach_suppressed: bool = False,
        topographic: float = 0.0,
        blocks: int = 0,
        shift: int = 0,
        retinotopic: float = 0.0,
        complex_cells: bool = False,
        pre_index: torch.Tensor | None = None,
    ) -> Projection | None:
        if blocks > 0:
            k = blocks

        k = min(k, pre_pop.size)

        if k <= 0:
            return None

        # Врождённый путь с постоянным весом — не храним, а вычисляем
        # (топографический хранится: хеш получателей не знает карты).
        if self.cfg.procedural_wiring and pre_index is None and not plastic and weight[0] == weight[1] != 0 and not topographic and not blocks and not retinotopic:
            proj = ProceduralProjection(
                name, pre_pop, post, k, weight[0],
                seed=self.cfg.seed * 1009 + len(self.projections), device=self.device,
            )
            proj.source_kind = next(
                a.kind for a in self.areas.values() if any(pre_pop is pop for _, pop in a.populations)
            )
            self.projections.append(proj)
            return proj

        if pre_index is not None:
            # Готовые источники (колонка-карта: ближайшие по месту).
            pre = pre_index
        elif blocks > 0:
            # Блок в блок: клетка получателя — от всех клеток блока того же
            # номера у источника (колонка → колонка, как столбики артикуляций).
            # shift — со сдвигом: блок b получает от блока b − shift (цепочка
            # часов: группа → следующая группа).
            first = (torch.arange(post.size, device=self.device) // blocks - shift) * blocks
            pre = first[:, None] + torch.arange(blocks, device=self.device)[None, :]
            pre = pre_pop.start + pre.clamp(min=0, max=pre_pop.size - 1)
        elif retinotopic > 0 and complex_cells:
            pre = self._complex_pre(post, k, pre_pop, retinotopic)
        elif retinotopic > 0:
            pre = self._retinotopic_pre(post, k, pre_pop, retinotopic)
        elif topographic > 0:
            pre = self._topographic_pre(post, k, pre_pop, topographic)
        else:
            pre = self._random_pre(post, k, pre_pop)
        initial = self._uniform((post.size, k), *weight)

        if blocks > 0 and shift > 0:
            # Первым блокам цепочки предшественника нет — входа нет.
            row_block = torch.arange(post.size, device=self.device) // blocks
            initial[row_block < shift] = 0.0

        proj = Projection(
            name,
            "net",
            pre,
            post,
            initial,
            plastic=plastic,
            cap=cap,
            innate_max=weight[1],
        )
        proj.pre_pop = pre_pop
        proj.innate_range = weight
        proj.modulator = modulator
        proj.ach_suppressed = ach_suppressed
        proj.source_kind = next(
            a.kind for a in self.areas.values() if any(pre_pop is pop for _, pop in a.populations)
        )
        self.projections.append(proj)

        return proj

    def _wire_columns(self, spec, area) -> None:
        """
        Врождённые колонки: E-клетки зоны идут группами по spec.columns,
        каждая клетка получает входы от всех соседей по колонке весом
        spec.column_weight (вместо своих первых случайных входов). Колонка —
        аттрактор: разбуженная, горит сама на медленном токе, пока её не
        погасит торможение зоны.
        """

        proj = self.projections[-1]
        size, k = proj.pre.shape
        mates = spec.columns - 1

        if k < mates:
            return

        cell = torch.arange(size, device=self.device)
        first = cell - cell % spec.columns
        others = first[:, None] + torch.arange(spec.columns, device=self.device)[None, :]
        others = others[others != cell[:, None]].view(size, mates)
        whole = others < size
        pre = torch.where(whole, others + area.E.start, proj.pre[:, :mates].long())
        proj.pre[:, :mates] = pre.to(proj.pre.dtype)
        proj.weight[:, :mates] = torch.where(
            whole, torch.tensor(spec.column_weight, device=self.device), proj.weight[:, :mates]
        )

    def _init_projections(self) -> None:
        cfg = self.cfg
        self.projections: list[Projection] = []

        # Места в поле зрения (столбец, px) у клеток зрительных зон: V1 — из
        # карты сетчатки, следующие ступени — при проводке (_retinotopic_pre).
        self.field_cols: dict[str, torch.Tensor] = {}
        self.field_width = 0

        self.field_layout: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.eye_sensor = None

        for sensor in self.sensors:
            if sensor.wiring == "oriented":
                area = self.areas[sensor.area]
                E = area.layers["L4"]["E"] if area.layers is not None else area.E
                self.field_layout[E.name] = self._oriented_layout(sensor, E.size)
                self.field_cols[E.name] = self.field_layout[E.name][2]
                self.field_width = sensor.cols
                self.eye_sensor = sensor

        self._place_maps()

        # ---- локальные микросхемы зон ----
        for spec in cfg.areas:
            area = self.areas[spec.name]

            if area.layers is not None:
                self._wire_column(spec, area)
                continue

            self._connect(
                f"{spec.name}: E->E", area.E, area.E, spec.k_ee,
                spec.ee_weight, plastic=spec.ee_plastic, cap=spec.ee_cap,
                modulator=spec.ee_modulator, ach_suppressed=spec.ee_ach_suppressed,
            )
            if self.projections and self.projections[-1].name == f"{spec.name}: E->E":
                self.projections[-1].scaled = getattr(spec, "ee_scaled", True)
            if spec.columns > 1:
                self._wire_columns(spec, area)

            if area.SOM is not None:
                self._wire_microcircuit(spec, area)
                continue

            self._connect(
                f"{spec.name}: E->I", area.E, area.I, cfg.k_ei,
                (cfg.ei_weight * spec.feedback, cfg.ei_weight * spec.feedback),
            )
            self._connect(
                f"{spec.name}: I->E", area.I, area.E, cfg.k_ie,
                (cfg.ie_weight * spec.inhibition, cfg.ie_weight * spec.inhibition),
            )
            self._connect(
                f"{spec.name}: I->I", area.I, area.I, cfg.k_ii,
                (cfg.ii_weight, cfg.ii_weight),
            )

        # ---- проводящие пути между зонами ----
        for path in cfg.pathways:
            # С понятиями слово будит свои клетки смысла напрямую, а
            # «слух → смысл» — слабый обучаемый путь (звучание -> место):
            # случайная сильная проводка будила бы посторонние клетки.
            if cfg.concepts_enabled and (path.src, path.dst) == ("auditory", "association"):
                path = cfg.form_pathway

            # Память хранит места слов — случайная проводка в неё не нужна.
            if cfg.memory_places and (path.src, path.dst) == ("association", "prefrontal"):
                continue

            src = self.areas[path.src]
            dst = self.areas[path.dst]
            pre, post, inhibitory, top_down = self._laminar(path, src, dst)

            proj = self._connect(
                getattr(path, "name", "") or f"{path.src} -> {path.dst}", pre, post, path.k,
                path.weight, plastic=path.plastic, cap=path.cap,
                modulator=path.modulator, ach_suppressed=path.ach_suppressed,
                topographic=path.topographic, blocks=getattr(path, "blocks", 0),
                shift=getattr(path, "shift", 0), retinotopic=getattr(path, "retinotopic", 0.0),
                complex_cells=getattr(path, "complex_cells", False),
            )

            if proj is not None:
                proj.predictive = path.predictive
                proj.scaled = getattr(path, "scaled", True)
                proj.dendrite = getattr(path, "dendrite", False) or top_down
                proj.vogels = getattr(path, "vogels", False)
                proj.scale_up = getattr(path, "scale_up", False)
                proj.cap = path.cap
            self._connect(
                f"{path.src} -> {path.dst}.I", pre, inhibitory, path.k_inhibitory,
                (path.inhibitory_weight, path.inhibitory_weight),
            )

        # ---- органы-датчики: рецепторы -> своя зона ----
        # У каждого органа свой участок общего массива рецепторов: ухо —
        # первым (как всегда), дальше остальные.
        for sensor, offset in zip(self.sensors, self.sensor_offsets):
            area = self.areas[sensor.area]
            # Вход органа (через таламус) — в L4 колонки (Potjans: таламус → L4).
            target = area.layers.get("L4", area.layers["L23"]) if area.layers is not None else None
            excitatory = target["E"] if target else area.E
            inhibitory = target["PV"] if target else area.I

            for pop, gain in ((excitatory, 1.0), (inhibitory, sensor.inhibitory_gain)):
                if sensor.wiring == "oriented" and pop is excitatory:
                    pre, weight = self._wire_oriented(sensor, pop.size)
                else:
                    # Случайные рецепторы своего органа (торможение сетчатки
                    # в V1 — тоже широкое, без формы поля).
                    k = min(sensor.k, sensor.size)
                    pre = torch.randint(
                        0, sensor.size, (pop.size, k), device=self.device, generator=self.generator,
                    )
                    weight = torch.full((pop.size, k), sensor.weight, device=self.device)

                self.projections.append(
                    Projection(
                        f"{sensor.name} -> {pop.name}", "sensory", pre + offset, pop, weight * gain,
                    )
                )

        self._pack()

    def _oriented_layout(self, sensor, cells: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Карта V1: клетка → ориентация и место в поле (строка, столбец, px).
        4 ориентации в каждом месте (гиперколонка), места — ровной сеткой с
        тем же соотношением сторон, что поле.
        """

        rows, cols = sensor.rows, sensor.cols
        places = max(1, cells // 4)
        grid_r = max(1, round((places * rows / cols) ** 0.5))
        grid_c = max(1, places // grid_r)
        cell = torch.arange(cells, device=self.device)
        angle = (cell % 4).float() * math.pi / 4
        place = (cell // 4) % (grid_r * grid_c)
        r0 = (place // grid_c).float().add(0.5).mul(rows / grid_r).sub(0.5)
        c0 = (place % grid_c).float().add(0.5).mul(cols / grid_c).sub(0.5)
        return angle, r0, c0

    def _retinotopic_pre(self, post: Population, k: int, pre_pop: Population, window: float) -> torch.Tensor:
        """
        Ступень зрительного пути: клетки получателя стоят вдоль поля по
        порядку, клетка берёт k случайных клеток источника, чьё место — в
        окне шириной window px вокруг её места (поле шире, чем у источника:
        V1 → V4 — в 2–3 раза, Dehaene et al. 2005). Места получателя
        запоминаются — следующая ступень строится так же.
        """

        src = self.field_cols[pre_pop.name]
        width = self.field_width
        cell = torch.arange(post.size, device=self.device).float()
        centre = (cell + 0.5) * width / post.size
        order = torch.argsort(src)
        sorted_cols = src[order]
        lo = torch.searchsorted(sorted_cols, centre - window / 2)
        hi = torch.searchsorted(sorted_cols, centre + window / 2).clamp(min=1)
        lo = torch.minimum(lo, hi - 1)
        pick = lo[:, None] + (
            torch.rand((post.size, k), device=self.device, generator=self.generator) * (hi - lo)[:, None].float()
        ).long()
        self.field_cols[post.name] = centre
        return pre_pop.start + order[pick.clamp(max=len(order) - 1)]

    def _complex_pre(self, post: Population, k: int, pre_pop: Population, window: float) -> torch.Tensor:
        """
        Сложные клетки V1 (Hubel & Wiesel 1962): клетка той же карты, что
        простые (ориентация и место), собирает простые клетки своей
        ориентации из квадрата window × window px вокруг своего места —
        штрих где угодно в этом квадрате. Берёт до k ближайших таких
        (остальные входы — повтор ближайшего).
        """

        angle, rows, cols = self.field_layout[pre_pop.name]
        my_angle, my_rows, my_cols = self._oriented_layout(self.eye_sensor, post.size)
        same = my_angle[:, None] == angle[None, :]
        dist = torch.maximum((my_rows[:, None] - rows[None, :]).abs(), (my_cols[:, None] - cols[None, :]).abs())
        dist = torch.where(same & (dist <= window / 2), dist, torch.full_like(dist, float("inf")))
        order = dist.argsort(dim=1)[:, :k]
        nearest = order[:, :1].expand_as(order)
        found = torch.gather(dist, 1, order).isfinite()
        self.field_layout[post.name] = (my_angle, my_rows, my_cols)
        self.field_cols[post.name] = my_cols
        return pre_pop.start + torch.where(found, order, nearest)

    def _wire_oriented(self, sensor, cells: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Сетчатка → V1: простая клетка видит штрих одной ориентации в своём
        месте. Центральная полоса OFF (тёмный штрих на светлом) шириной в
        шаг между местами — 2 × 5 рецепторов, по бокам вплотную по 5 ON
        (светлый фон рядом со штрихом): входы, выстроенные по полю простой
        клетки (Hubel & Wiesel 1962; Reid & Alonso 1995). Поля соседей
        перекрываются: штрих в пиксель не проходит между местами. [замер]
        rf-расчёт (scratchpad): с полосой в пиксель на сетке через 2 px
        горизонтальная линия будила клетки 0° слабее всех (109 против 482 у
        45°) — центр стоял рядом со штрихом. 4 ориентации (0°, 45°, 90°,
        135°) в каждом месте, места — ровной сеткой по полю. Рецепторы:
        слой ON [0, rows·cols), слой OFF [rows·cols, 2·rows·cols).
        """

        rows, cols = sensor.rows, sensor.cols
        layer = rows * cols
        angle, r0, c0 = self._oriented_layout(sensor, cells)
        dr, dc = -torch.sin(angle), torch.cos(angle)  # вдоль штриха
        t = torch.arange(-2, 3, device=self.device).float()
        pre, sign = [], []

        # Тяни-толкай (push-pull; Hirsch et al. 1998, Ferster 1988): в каждом
        # подполе свой знак возбуждает, противоположный — тормозит (через
        # местные тормозные клетки; здесь — отрицательный вес, упрощение).
        # Центр: OFF +, ON −; бока: ON +, OFF −.
        for side, own, other in ((-0.5, layer, 0), (0.5, layer, 0), (-1.5, 0, layer), (1.5, 0, layer)):
            # Сдвиг поперёк штриха (перпендикуляр: (dc, -dr)).
            r = r0[:, None] + t[None, :] * dr[:, None] + side * dc[:, None]
            c = c0[:, None] + t[None, :] * dc[:, None] - side * dr[:, None]
            r = (r + 0.5).floor().long().clamp(0, rows - 1)
            c = (c + 0.5).floor().long().clamp(0, cols - 1)
            pre += [own + r * cols + c, other + r * cols + c]
            sign += [torch.ones_like(r, dtype=torch.float), -torch.ones_like(r, dtype=torch.float)]

        pre = torch.cat(pre, dim=1)
        weight = torch.cat(sign, dim=1) * sensor.weight
        return pre, weight

    def _pack(self) -> None:
        """
        Все синапсы мозга — в один плоский массив (пластичные первыми),
        Projection хранит view в него.

        Для event-driven симуляции строятся таблицы:

            out_all[pre]      все исходящие синапсы нейрона
            out_plastic[pre]  исходящие пластичные (LTD)
            in_plastic[post]  входящие пластичные (LTP)

        Пустые ячейки указывают на фиктивный синапс.
        """

        self.procedural = [p for p in self.projections if isinstance(p, ProceduralProjection)]
        net = sorted(
            (p for p in self.projections if p.source == "net" and not isinstance(p, ProceduralProjection)),
            key=lambda p: not p.plastic,
        )
        sensory = [p for p in self.projections if p.source == "sensory"]
        self.projections = net + self.procedural + sensory
        self.proc_rows = (
            ProceduralRows(self.procedural, self.n, self.n_slots, self.DUMMY_NEURONS, self.device)
            if self.procedural else None
        )
        self.plastic = [p for p in net if p.plastic]

        def flatten(group, dummy: bool):
            pre = [p.pre.flatten() for p in group]
            post = [
                torch.arange(p.post.start, p.post.stop, device=self.device)[:, None]
                .expand_as(p.pre).flatten()
                for p in group
            ]
            weight = [p.weight.flatten() for p in group]

            if dummy:
                # Фиктивные синапсы: фиктивный нейрон -> фиктивный нейрон, вес 0.
                ids = self.n + torch.arange(
                    self.DUMMY_SYNAPSES, device=self.device
                ) % self.DUMMY_NEURONS
                pre.append(ids)
                post.append(ids)
                weight.append(torch.zeros(self.DUMMY_SYNAPSES, device=self.device))

            pre = torch.cat(pre).to(torch.int32)
            post = torch.cat(post).to(torch.int32)
            weight = torch.cat(weight)

            offset = 0

            for p in group:
                n = p.weight.numel()
                shape = p.weight.shape
                p.pre = pre[offset: offset + n].view(shape)
                p.weight = weight[offset: offset + n].view(shape)
                p.offset = offset
                offset += n

            return pre, post, weight

        self.net_pre, self.net_post, self.net_weight = flatten(net, dummy=True)
        # Ветка дендрита каждого синапса (имеет смысл для синапсов памяти и
        # цели в смысловую кору): врождённые — случайно; свой генератор,
        # чтобы не сдвигать остальную проводку.
        branches = torch.Generator(device=self.device).manual_seed(self.cfg.seed + 7)
        self.net_branch = torch.randint(
            0, max(1, self.cfg.dendrite_branches), self.net_weight.shape,
            dtype=torch.int8, device=self.device, generator=branches,
        )

        for p in net:
            p.branch = self.net_branch[p.offset: p.offset + p.weight.numel()].view(p.weight.shape)
        self.ear_pre, self.ear_post, self.ear_weight = flatten(sensory, dummy=False)

        # Задержка каждого синапса — лишние шаги сверху обычного одного (0 —
        # приходит на следующем шаге, как всегда). Между зонами — axon_delay_ms
        # ±50%; свой генератор — проводка не сдвигается.
        self.net_delay = torch.zeros(self.net_weight.shape, dtype=torch.int16, device=self.device)

        if getattr(self.cfg, "axon_delays", False):
            delays = torch.Generator(device=self.device).manual_seed(self.cfg.seed + 11)
            steps = self.cfg.axon_delay_ms / self.cfg.dt_ms

            for p in net:
                src = getattr(p, "pre_pop", None)

                if src is None or src.name.split(".")[0] == p.post.name.split(".")[0]:
                    continue

                n = p.weight.numel()
                jitter = 1.0 + torch.rand(n, device=self.device, generator=delays) - 0.5
                extra = (steps * jitter).round().clamp(min=1) - 1
                self.net_delay[p.offset: p.offset + n] = extra.to(torch.int16)

        self.delay_slots = int(self.net_delay.max()) + 1

        self.n_synapses = self.net_weight.numel() - self.DUMMY_SYNAPSES
        self.n_plastic = sum(p.weight.numel() for p in self.plastic)

        # Следы и свежая часть веса — в половинной точности (fp16): 2 Б
        # вместо 4 на каждый пластичный синапс; сам вес — fp32 (мелкие
        # шаги обучения в fp16 терялись бы).
        # Хвост — фиктивные ячейки для пустых мест таблиц.
        self.eligibility = torch.zeros(
            self.n_plastic + self.DUMMY_SYNAPSES, dtype=torch.float16, device=self.device
        )

        # Вес каждого пластичного синапса при рождении: выученное —
        # это отклонение от него, а не от общего врождённого максимума.
        # fp16 без потерь: веса при рождении создаются точными в fp16
        # (exact16). [замер] простое округление fp32 → fp16 ломало
        # «выученное = вес − вес при рождении»: ошибка сравнима с мелкими
        # выученными шагами (тест «скука → скучно»).
        self.birth_weight = self.net_weight[: self.n_plastic].to(torch.float16)

        # Свежая (ранняя, early-LTP/LTD) часть выученного: входит в вес,
        # но затухает к закреплённому весу, пока сон её не закрепит.
        self.early = torch.zeros(self.n_plastic, dtype=torch.float16, device=self.device)

        for p in self.plastic:
            p.eligibility = self.eligibility[
                p.offset: p.offset + p.weight.numel()
            ].view(p.weight.shape)
            p.birth = self.birth_weight[
                p.offset: p.offset + p.weight.numel()
            ].view(p.weight.shape)
            p.early = self.early[
                p.offset: p.offset + p.weight.numel()
            ].view(p.weight.shape)

            # Врождённо сильнее потолка (колонки рта): учёба врождённое не
            # срезает — потолок синапса не ниже его веса при рождении.
            if bool((p.birth.float() > p.cap).any()):
                p.ceiling = p.birth.float().clamp(min=p.cap)

        # Пути гиппокампа учатся от ацетилхолина (encode), а не от дофамина.
        # Пути в стриатум D2 («стой») — от провала дофамина: знак −1
        # (D2-рецептор: LTP при паузе дофамина, Frank 2005; Shen 2008).
        self.encoding = [p for p in self.plastic if p.modulator == "acetylcholine"]
        inverted = [p for p in self.plastic if p.modulator == "dopamine_d2"]
        # Пути из зон, стреляющих непрерывно: след STDP там ни о чём не
        # говорит (LTP ≈ LTD), а во сне раздувал общее слово — учит их
        # совпадение (language._learn_context).
        coincidence = [p for p in self.plastic if p.modulator == "coincidence"]
        # Чистый STDP без третьего фактора (цепочка звуков рта): след сам
        # становится весом на каждом услышанном слове (hebbian()).
        self.hebbian_paths = [p for p in self.plastic if p.modulator == "hebbian"]
        self.dopamine_mask = None

        if self.encoding or inverted or coincidence or self.hebbian_paths:
            self.dopamine_mask = torch.ones(self.n_plastic, dtype=torch.float16, device=self.device)

            for p in self.encoding + coincidence + self.hebbian_paths:
                self.dopamine_mask[p.offset: p.offset + p.weight.numel()] = 0.0

            for p in inverted:
                self.dopamine_mask[p.offset: p.offset + p.weight.numel()] = -1.0

        # Пути на дендриты из любой зоны (PathwaySpec.dendrite): часы → рот.
        dendritic = [p for p in net if getattr(p, "dendrite", False)]
        self.dendrite_syn = None

        if dendritic:
            self.dendrite_syn = torch.zeros(self.net_weight.numel(), dtype=torch.bool, device=self.device)

            for p in dendritic:
                self.dendrite_syn[p.offset: p.offset + p.weight.numel()] = True

        # Пути, передачу по которым подавляет ацетилхолин: доля подавления
        # на каждом синапсе (0 — не подавляется).
        self.ach_mask = None
        suppressed = [p for p in net if getattr(p, "ach_suppressed", False)]

        if suppressed:
            self.ach_mask = torch.zeros(self.net_weight.numel(), dtype=torch.float16, device=self.device)

            for p in suppressed:
                self.ach_mask[p.offset: p.offset + p.weight.numel()] = self.cfg.ach_suppression

        self._build_tables()

    def set_acetylcholine(self, level: float) -> None:
        """Уровень ацетилхолина (0 — сон и вспоминание, 1 — бодрая запись нового)."""

        self.acetylcholine.fill_(level)

    # Нейромодуляторы: имя → ядро-источник (зона, чьи E-клетки выбрасывают);
    # None — все нейроны (аденозин — продукт обмена веществ любой клетки).
    # Галанин (с ГАМК) — выброс VLPO: сон, когда его больше норадреналина.
    MODULATORS = {"dopamine": "vta", "noradrenaline": "lc", "adenosine": None, "galanin": "vlpo"}

    def _init_chemistry(self) -> None:
        """
        Химия как концентрации (объёмная передача): каждый спайк клетки
        ядра выбрасывает вещество, оно убывает со своим временем
        (обратный захват). Концентрация дофамина и норадреналина — «спайков
        на клетку ядра за последнее τ»: залп, где выстрелила половина ядра,
        — 0.5. Аденозин — средняя частота всех нейронов плюс обмен веществ
        покоя, усреднённая за τ (Гц): копится, пока мозг бодрствует, и
        вымывается во сне. Действие на клетки — через занятость рецепторов
        c / (c + K), а не через формулы на события.
        """

        names = list(self.MODULATORS)
        self.chemistry_names = names
        # float64: аденозин копится часами — прибавки за шаг ~1e-10.
        self.chemistry = torch.zeros(len(names), device=self.device, dtype=torch.float64)
        self.chemistry_decay = torch.zeros(len(names), device=self.device, dtype=torch.float64)
        # Выброс на спайк (доля клеток источника) и постоянный выброс за шаг.
        self.chemistry_release = torch.zeros(len(names), device=self.device, dtype=torch.float64)
        self.chemistry_basal = torch.zeros(len(names), device=self.device, dtype=torch.float64)
        self.chemistry_source = torch.zeros(len(names), self.n_slots, device=self.device)

        for i, name in enumerate(names):
            area = self.areas.get(self.MODULATORS[name])
            cells = area.E.slice if area is not None else slice(0, self.n)
            size = area.E.size if area is not None else self.n
            self.chemistry_source[i, cells] = 1.0 / size

        # Рецепторы: клетки зоны чувствуют вещество — ток (мВ) при полной
        # занятости, знак — возбуждает или тормозит (cfg.receptors).
        self.chemistry_half = torch.tensor(
            [getattr(self.cfg, f"{n}_half") for n in names], device=self.device, dtype=torch.float64,
        )
        self.receptor_mv = torch.zeros(len(names), self.n_slots, device=self.device)

        for name, area_name, mv in self.cfg.receptors:
            area = self.areas.get(area_name)

            if area is not None and name in names:
                self.receptor_mv[names.index(name), area.E.slice] = mv

        # Часы (SCN): ток клеткам часов по фазе суток — молекулярный цикл
        # клетки, выставляется set_clock (copy_ — для CUDA Graph).
        self.clock_mv = torch.zeros(self.n_slots, device=self.device)
        self.sleep_chemistry(False)

    def _chemistry_rates(self, asleep: bool) -> tuple[list[float], list[float], list[float]]:
        """τ (мс), выброс на долю выстреливших, постоянный выброс за шаг — по веществам."""

        cfg, dt = self.cfg, self.cfg.dt_ms
        tau, release, basal = [], [], []

        for name in self.chemistry_names:
            if name == "adenosine":
                # Сон у нас сжат, как проигрывание у грызунов: τ вымывания
                # делится на sleep_compression (DESIGN «Сон по аденозину»).
                t = cfg.adenosine_sleep_tau_ms / cfg.sleep_compression if asleep else cfg.adenosine_tau_ms
                hz = cfg.adenosine_sleep_basal_hz if asleep else cfg.adenosine_basal_hz
                # Среднее за τ в Гц: доля за шаг × 1000 / dt, вес dt / τ.
                tau.append(t)
                release.append((dt / t) * 1000.0 / dt)
                basal.append((dt / t) * hz)
            else:
                tau.append(getattr(cfg, f"{name}_tau_ms"))
                release.append(1.0)
                basal.append(0.0)

        return tau, release, basal

    def sleep_chemistry(self, asleep: bool) -> None:
        """Сон и бодрствование: аденозин вымывается во сне (copy_ — для CUDA Graph)."""

        tau, release, basal = self._chemistry_rates(asleep)
        f64 = lambda xs: torch.tensor(xs, device=self.device, dtype=torch.float64)
        self.chemistry_decay.copy_(torch.exp(-self.cfg.dt_ms / f64(tau)))
        self.chemistry_release.copy_(f64(release))
        self.chemistry_basal.copy_(f64(basal))
        self._asleep = asleep

    def set_clock(self) -> None:
        """
        Суточные часы: клетки SCN получают ток по фазе суток (днём выше —
        у SCN пик активности в середине дня). Время суток — время мозга
        от рождения плюс час рождения.
        """

        area = self.areas.get("scn")

        if area is None:
            return

        hour = (self.cfg.birth_hour + self.time_ms / 3_600_000.0) % 24.0
        day = 0.5 * (1.0 + math.cos(2.0 * math.pi * (hour - self.cfg.scn_peak_hour) / 24.0))
        self.clock_mv[area.E.slice] = self.cfg.scn_mv * day

    def hour(self) -> float:
        """Время суток мозга, ч."""

        return (self.cfg.birth_hour + self.time_ms / 3_600_000.0) % 24.0

    def level(self, name: str) -> float:
        """Концентрация вещества (см. _init_chemistry)."""

        return float(self.chemistry[self.chemistry_names.index(name)])

    def occupancy(self, name: str) -> float:
        """Занятость рецепторов: c / (c + K), K — полунасыщение вещества."""

        c = self.level(name)
        k = getattr(self.cfg, f"{name}_half")

        return c / (c + k)

    def pass_time(self, seconds: float) -> None:
        """
        Время без симуляции (тишина): мозг живёт — часы идут, свежее
        забывается, вещества убывают к уровню покоя (нейроны в тишине
        молчат — остаётся обмен веществ покоя).
        """

        if seconds <= 0:
            return

        self.time_ms += seconds * 1000.0
        tau, release, basal = self._chemistry_rates(self._asleep)
        dt = self.cfg.dt_ms

        for i in range(len(tau)):
            fade = math.exp(-seconds * 1000.0 / tau[i])
            # Постоянный выброс за шаг в равновесии: basal / (1 − decay).
            rest = basal[i] / (1.0 - math.exp(-dt / tau[i]))
            self.chemistry[i] = self.chemistry[i] * fade + rest * (1.0 - fade)

    def _build_tables(self, refill: bool = False) -> None:
        P, S = self.n_plastic, self.n_synapses
        table = lambda keys, pad: EventTable(keys, self.n_slots, pad, self.DUMMY_SYNAPSES)
        out_all = table(self.net_pre[:S], S)
        out_plastic = PrefixTable(out_all, self.net_pre[:P], self.n_slots, P, self.DUMMY_SYNAPSES)

        # Рост синапсов меняет только пре-нейроны (post и число синапсов те
        # же): таблицы той же формы переписываются на месте, и CUDA Graph
        # не записывается заново. [замер] 20 тыс., чтение: 12 ростов на 364
        # слова — 2.9 с из 11 (перестройка, запись графа, gc).
        # Входящие (in_plastic) по post — от роста не меняются.
        if refill and hasattr(self, "out_all"):
            # Начала строк префикса — вид на ptr общей таблицы: сперва общая.
            if self.out_all.refill(out_all) and self.out_plastic.refill(out_plastic):
                return

        self.out_all, self.out_plastic = out_all, out_plastic
        self.in_plastic = table(self.net_post[:P], P)
        # Таблицы — новые тензоры: старые CUDA Graph на них не смотрят.
        self._graphs = {}

        # Отдать кэш освобождённых временных массивов: иначе аллокатор
        # держит резерв больше видеокарты, и WSL уходит в системную память.
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _init_state(self) -> None:
        cfg = self.cfg
        z = lambda: torch.zeros(self.n_slots, device=self.device)

        self.v = cfg.v_rest + self._uniform((self.n_slots,), 0.0, 10.0)
        # Спайки в пути (задержки аксонов): кольцо [шагов вперёд, клеток] — что
        # придёт к сому и к веткам через столько-то шагов; tick — номер шага.
        branches = max(1, cfg.dendrite_branches)
        self.pending = torch.zeros(self.delay_slots, self.n_slots, device=self.device)
        self.pending_dend = torch.zeros(self.delay_slots, self.n_slots * branches, device=self.device)
        self.tick = torch.zeros(1, dtype=torch.long, device=self.device)
        self.i_syn = z()
        # Тормозной ток (проводимость; cfg.layers): отдельно от возбуждающего.
        self.i_inh = z()
        self.i_slow = z()
        # Дендритный ток (контекст из рабочей памяти, см. dendritic_context).
        self.i_dend = torch.zeros(self.n_slots * max(1, cfg.dendrite_branches), device=self.device)
        # Пул торможения смысловой коры: отфильтрованное число её спайков.
        self.pool = torch.zeros(1, device=self.device)
        # Пулы торможения отдельных зон (AreaSpec.pool_mv): срез E и мВ на спайк.
        self.area_pools = [
            (self.areas[a.name].E.slice, float(a.pool_mv))
            for a in cfg.areas if getattr(a, "pool_mv", 0.0) > 0
        ]
        self.area_pool = torch.zeros(max(1, len(self.area_pools)), device=self.device)
        self.adapt = z()
        self.adapt_slow = z()
        self.refractory = z()
        self.spikes = z()
        # STDP-следы: пре (LTP) и пост (LTD).
        self.trace_pre = z()
        self.trace_pre_rise = z()
        self.ltp_kernel = z()
        self.trace_post = z()
        # Счётчик спайков для окна наблюдения.
        self.counts = z()
        # Счёт спайков с весом, убывающим от начала звука (рот по буквам):
        # сколько передал бы устающий синапс на момент спайка (кратковременная
        # депрессия, Tsodyks & Markram 1997) — ранние буквы слова весят больше.
        self.track_primacy = bool(getattr(cfg, "mouth_letters", False))
        self.primacy_counts = z()
        self.primacy_gain = torch.ones(1, device=self.device)
        self.primacy_decay = math.exp(-cfg.dt_ms / max(1e-6, getattr(cfg, "primacy_tau_ms", 50.0)))
        # Ток покоя (мВ): гомеостаз возбудимости зоны.
        self.bias = z()

        for area in self.areas.values():
            self.bias[area.I.slice] = cfg.i_excitability

            if area.SOM is not None:
                # До порога — как у клетки этого типа (реобаза × сопротивление).
                gap = cfg.v_threshold - cfg.v_rest

                for _, group in self._typed_groups(area):
                    for kind, data_gap in (("PV", cfg.pv_gap_mv), ("SOM", cfg.som_gap_mv), ("VIP", cfg.vip_gap_mv)):
                        self.bias[group[kind].slice] = gap - data_gap

        # Развитие до рождения (калибровка): гомеостаз быстрый везде.
        self.developing = False
        # Ацетилхолин (тензор — его читает записанный шаг CUDA Graph).
        self.acetylcholine = torch.zeros(1, device=self.device)
        # Сигнал «изнутри тела» (мВ): чувство возбуждает свой набор
        # нейронов лимбической зоны — как ухо возбуждает слуховую кору.
        self.interoception = z()
        # Короткий толчок (kick): ток на несколько шагов — сигнал начала
        # звука или команда «начать»; гаснет сам.
        self.kick_current = z()
        self.kick_left = torch.zeros(1, device=self.device)
        # Истинная eligibility = eligibility * eligibility_scale:
        # затухание всех следов — одно умножение скаляра.
        self.eligibility_scale = torch.ones(1, device=self.device)
        # Спайки, не влезшие в таблицу шага (должно быть 0).
        self.dropped = torch.zeros(1, device=self.device)
        self._init_chemistry()
        # Базовый контур (base_circuit): ресурс пресинапса (истощение,
        # 1 — полный) и усиление торможения, приходящего к клетке (1 — норма).
        self.stp_x = torch.ones(self.n_slots, device=self.device)
        # Кратковременная пластичность клеток с типами: u (до первого спайка
        # 0) и ресурс x к каждому типу получателя.
        self.stp_u4 = torch.zeros(self.n_slots, 4, device=self.device)
        self.stp_x4 = torch.ones(self.n_slots, 4, device=self.device)
        self.inh_gain = torch.ones(self.n_slots, device=self.device)
        # У маленьких мозгов зоны малы, и случайные пики занимают
        # большую долю нейронов — им нужен запас не по доле, а в штуках.
        self.max_spikes = max(
            int(self.n * cfg.max_spike_fraction),
            min(int(self.n * 0.2), cfg.min_spike_table),
        )
        # Пустые строки event-таблицы -> разные фиктивные нейроны.
        self.dummy_rows = (
            self.n + torch.arange(self.max_spikes, device=self.device) % self.DUMMY_NEURONS
        )


        # Вход органов-датчиков на CHUNK шагов вперёд.
        self.ear_input = torch.zeros(
            self.CHUNK, self.n_receptors, device=self.device
        )

        # Структурная пластичность: сколько синапсов отмерло и выросло.
        self.synapses_grown = 0

    # ==============================================================
    # SIMULATION
    # ==============================================================

    # Шагов в одном запуске. На GPU CHUNK шагов записаны
    # в CUDA Graph и запускаются одной командой.
    CHUNK = 10

    def _step(self, heard_now: torch.Tensor, plastic: bool) -> None:
        """
        Один шаг dt, event-driven: обрабатываются только синапсы
        нейронов, которые выстрелили. Всё обновляется на месте,
        без синхронизаций с CPU, — иначе шаг нельзя записать
        в CUDA Graph.
        """

        cfg = self.cfg

        # ---- фоновый шум: Poisson ≈ Гаусс с той же дисперсией ----
        drive = torch.randn(self.n_slots, device=self.device)
        drive.mul_(self.noise_std).add_(self.noise_mean).clamp_(min=0.0)
        drive.mul_(self.noise_gain)

        # ---- синаптический вход от спайков прошлого шага ----
        fired_before = self._fired()

        synapses, pre_of = self.out_all.events(fired_before)
        post = self.net_post.index_select(0, synapses)
        weight = self.net_weight.index_select(0, synapses)

        if self.ach_mask is not None:
            # Ацетилхолин подавляет передачу по отмеченным путям.
            weight = weight * (1.0 - self.acetylcholine * self.ach_mask.index_select(0, synapses))

        if cfg.dendritic_context:
            # Рабочая память -> смысл: на дальний дендрит, не в сому.
            to_dendrite = (
                self.dendrite_pre.index_select(0, pre_of)
                & self.dendrite_post.index_select(0, post)
            )

            if self.dendrite_syn is not None:
                to_dendrite = to_dendrite | self.dendrite_syn.index_select(0, synapses)
            branches = max(1, cfg.dendrite_branches)
            slot = post.long() * branches + self.net_branch.index_select(0, synapses).long()

            if self.delay_slots > 1:
                # В путь: придёт к ветке через свою задержку.
                ahead = (self.tick + self.net_delay.index_select(0, synapses).long()) % self.delay_slots
                self.pending_dend.view(-1).index_add_(0, ahead * self.i_dend.numel() + slot, weight * to_dendrite)
                now = self.tick % self.delay_slots
                dendrite = self.pending_dend.index_select(0, now).squeeze(0)
                self.pending_dend.index_fill_(0, now, 0.0)
            else:
                dendrite = torch.zeros_like(self.i_dend).index_add_(0, slot, weight * to_dendrite)
            self.i_dend.mul_(self.decay_dend).add_(dendrite)
            weight = weight * ~to_dendrite

        if self.has_som:
            # Синапсы клеток с типами: свои U, ресурс и облегчение к каждому
            # типу получателя (cfg.column_stp). Сила спайка = u·x до выброса /
            # U: состояние уже после спайка — x_до = x / (1 − u).
            typed = self.typed_pre.index_select(0, pre_of)
            slot = pre_of.long() * 4 + self.cell_type.index_select(0, post).clamp(min=0)
            u = self.stp_u4.view(-1).index_select(0, slot)
            x = self.stp_x4.view(-1).index_select(0, slot)
            gain = u * x / (1.0 - u).clamp(min=1e-3) / self.stp_U4.view(-1).index_select(0, slot)
            weight = torch.where(typed, weight * gain, weight)

        if cfg.base_circuit:
            # Истощение: синапс отдаёт долю оставшегося ресурса пресинапса
            # (у клеток с типами — своё, выше).
            depletion = self.stp_x.index_select(0, pre_of)

            if self.has_som:
                depletion = torch.where(typed, 1.0, depletion)
            weight = weight * depletion
            # Торможение к клетке × её усиление (растёт, если клетка
            # стреляет выше нормы).
            weight = torch.where(weight < 0, weight * self.inh_gain.index_select(0, post), weight)

        if self.conductance:
            # Торможение — проводимость (cfg.layers): копится отдельно, к соме
            # приходит × движущая сила (см. ниже). Задержка у тормозных путей
            # между зонами не учитывается — местные идут через шаг.
            inhibition = torch.zeros_like(drive).index_add_(0, post, weight.clamp(max=0.0))
            weight = weight.clamp(min=0.0)

        if self.delay_slots > 1:
            # В путь: каждый синапс доставит спайк через свою задержку; к соме
            # приходит то, что было в пути и пришло сейчас.
            ahead = (self.tick + self.net_delay.index_select(0, synapses).long()) % self.delay_slots
            self.pending.view(-1).index_add_(0, ahead * self.n_slots + post.long(), weight)
            now = self.tick % self.delay_slots
            drive.add_(self.pending.index_select(0, now).squeeze(0))
            self.pending.index_fill_(0, now, 0.0)
            self.tick.add_(1)
        else:
            drive.index_add_(0, post, weight)

        if self.proc_rows is not None:
            # Врождённые пути с постоянным весом: получатели — из хеша.
            proc_post, proc_weight = self.proc_rows.rows(fired_before)
            drive.index_add_(0, proc_post.flatten(), proc_weight.flatten())

        heard = heard_now.index_select(0, self.ear_pre)
        drive.index_add_(0, self.ear_post, heard * self.ear_weight)

        if plastic:
            # LTD: пре пришёл после поста. Нейроны рабочей памяти
            # (ковариационное правило) этой ветки не имеют.
            synapses, pre_of = self.out_plastic.events(fired_before)
            post = self.net_post.index_select(0, synapses)
            ltd = self.trace_post.index_select(0, post) * self.ltd_mask.index_select(0, pre_of)
            sign = -cfg.stdp_a_minus

            if self.symmetric_area is not None:
                # Симметричное окно: пре после поста — тоже потенциация.
                zone = self.symmetric_area.index_select(0, post)
                same = (zone > 0) & (zone == self.symmetric_area.index_select(0, pre_of))
                sign = torch.where(same, cfg.stdp_a_plus, -cfg.stdp_a_minus)

            self.eligibility.index_add_(
                0,
                synapses,
                (ltd * (sign / self.eligibility_scale)).to(self.eligibility.dtype),
            )

        self.i_syn.mul_(self.decay_syn).add_(drive * self.syn_scale)

        if self.conductance:
            self.i_inh.mul_(self.decay_syn).add_(inhibition * self.syn_scale)
        self.i_slow.mul_(self.decay_slow).add_(drive.clamp(min=0.0) * self.slow_scale)
        self.i_slow.clamp_(max=self.slow_max)

        # ---- мембрана (AdEx / LIF) ----
        exp_term = torch.where(
            self.is_adex,
            self.delta_t
            * torch.exp(
                ((self.v - cfg.v_threshold) / self.safe_delta_t).clamp(max=10.0)
            ),
            0.0,
        )

        synaptic = self.i_syn + self.i_slow

        if self.conductance:
            # Движущая сила ГАМК-А: ток ∝ (v − E_ГАМК), нормирован к −55 мВ,
            # где мерили IPSP (Campagnola et al. 2022); у реверсии торможение
            # только шунтирует, ниже неё не тянет (Gulledge & Stuart 2003).
            force = ((self.v - cfg.gaba_reversal_mv) / (cfg.ipsp_holding_mv - cfg.gaba_reversal_mv)).clamp(min=0.0)
            force = torch.where(self.conductance_cells, force, 1.0)
            synaptic = synaptic + (self.i_inh + self.dc_inh_mv) * force

        if cfg.dendritic_context:
            # Плато на дендрите усиливает возбуждающий вход сомы:
            # контекст выбирает среди того, что предлагает текущее слово.
            if cfg.dendrite_branches > 1:
                # Плато — на самой горячей ветке, если её ток выше порога.
                branch = self.i_dend.view(self.n_slots, cfg.dendrite_branches)
                plateau = torch.sigmoid(
                    (branch - cfg.dendrite_threshold_mv) / cfg.dendrite_slope_mv
                ).amax(1)
                # Плато ветки деполяризует и сому (см. dendrite_soma_mv; у
                # зоны может быть своя сила — AreaSpec.dendrite_soma_mv).
                somatic = self.dendrite_soma * plateau
            else:
                plateau = self.i_dend / (self.i_dend + cfg.dendrite_half_mv)
                somatic = 0.0
            synaptic = synaptic + cfg.dendrite_gain * plateau * synaptic.clamp(min=0.0) + somatic

        if cfg.pool_inhibition_mv > 0:
            synaptic = synaptic - cfg.pool_inhibition_mv * self.pool * self.pool_mask

        if self.area_pools:
            synaptic = synaptic.clone()

            for i, (cells, mv) in enumerate(self.area_pools):
                synaptic[cells] = synaptic[cells] - mv * self.area_pool[i]

        # Вещества через рецепторы: занятость c / (c + K) × ток клетки.
        occupancy = (self.chemistry / (self.chemistry + self.chemistry_half)).to(torch.float32)
        modulated = occupancy @ self.receptor_mv

        dv = (
            cfg.v_rest - self.v + exp_term - self.adapt - self.adapt_slow
            + synaptic + self.bias + self.dc_mv + self.interoception + modulated + self.clock_mv
            + self.kick_current * (self.kick_left > 0)
        ) * self.decay_m
        v = torch.where(self.refractory <= 0, self.v + dv, self.v)
        self.refractory.sub_(cfg.dt_ms)

        fired = v >= self.v_peak
        self.spikes.copy_(fired)
        # Выброс веществ ядрами: доля выстреливших клеток каждого ядра.
        released = (self.chemistry_source @ self.spikes).to(torch.float64)
        self.chemistry.mul_(self.chemistry_decay).add_(released * self.chemistry_release + self.chemistry_basal)

        self.v.copy_(torch.where(fired, cfg.v_reset, v))
        self.refractory.copy_(torch.where(fired, cfg.refractory_ms, self.refractory))
        self.adapt.mul_(self.decay_adapt).add_(self.adapt_b * self.spikes)
        # Ацетилхолин блокирует sAHP (Madison & Nicoll 1984).
        sahp = self.slow_ahp_b * (1.0 - cfg.ach_blocks_sahp * self.acetylcholine)
        self.adapt_slow.mul_(self.decay_slow_ahp).add_(sahp * self.spikes)

        if cfg.base_circuit:
            # Ресурс восстанавливается к 1, спайк тратит долю stp_u.
            self.stp_x.add_((1.0 - self.stp_x) * self.stp_recover)
            # Истощаются синапсы E зон с обучаемыми связями и все E колонок
            # (E → E и E → PV ослабевают при повторе — Markram et al. 1998;
            # Campagnola et al. 2022).
            self.stp_x.sub_(cfg.stp_u * self.stp_x * self.spikes * self.cortex_e)
            # Усиление торможения: каждый спайк E коры поднимает, медленно
            # возвращается к 1 (в равновесии 1 + шаг · τ · частота).
            self.inh_gain.sub_(1.0).mul_(self.decay_inh_gain).add_(1.0)
            self.inh_gain.add_(cfg.inh_gain_per_spike * self.spikes * self.inh_adapts)

        if self.has_som:
            # u спадает к 0 за τ облегчения, спайк поднимает его на U·(1 − u);
            # ресурс x восстанавливается за τ восстановления, спайк тратит u·x
            # (Tsodyks & Markram 1997; Markram et al. 1998).
            spiked = (self.spikes * self.typed_pre)[:, None]
            self.stp_u4.mul_(self.stp_fac4)
            self.stp_x4.add_((1.0 - self.stp_x4) * self.stp_rec4)
            self.stp_u4.add_(self.stp_U4 * (1.0 - self.stp_u4) * spiked)
            self.stp_x4.sub_(self.stp_u4 * self.stp_x4 * spiked)

        if cfg.pool_inhibition_mv > 0:
            self.pool.mul_(self.decay_pool).add_((self.spikes * self.pool_mask).sum())

        for i, (cells, _mv) in enumerate(self.area_pools):
            self.area_pool[i] = self.area_pool[i] * self.decay_pool + self.spikes[cells].sum()
        self.counts.add_(self.spikes)
        self.kick_left.sub_(1.0).clamp_(min=0.0)

        if self.track_primacy:
            self.primacy_counts.add_(self.spikes * self.primacy_gain)
            self.primacy_gain.mul_(self.primacy_decay)
        self.dropped.add_((self.spikes.sum() - self.max_spikes).clamp(min=0.0))

        # ---- STDP -> eligibility ----
        if plastic:
            # LTP: пост выстрелил после пре.
            fired_now = self._fired()
            synapses, _ = self.in_plastic.events(fired_now)
            pre = self.net_pre.index_select(0, synapses)
            self.eligibility.index_add_(
                0,
                synapses,
                (
                    self.ltp_kernel.index_select(0, pre)
                    * (cfg.stdp_a_plus / self.eligibility_scale)
                ).to(self.eligibility.dtype),
            )
            self.eligibility_scale.mul_(self.decay_eligibility)

        self.trace_pre.mul_(self.decay_plus).add_(self.spikes)
        self.trace_pre_rise.mul_(self.decay_rise).add_(self.spikes)
        torch.sub(self.trace_pre, self.trace_pre_rise, out=self.ltp_kernel)
        self.ltp_kernel.mul_(self.ltp_gain)

        if self.cfg.context_rule == "covariance":
            # Рабочая память: вклад нейрона = насколько он сейчас
            # активнее среднего по зоне (ковариационное правило, BCM).
            for pop in self.memory_pops:
                trace = self.trace_pre[pop.slice]
                torch.sub(trace, trace.mean(), out=self.ltp_kernel[pop.slice])
                self.ltp_kernel[pop.slice].mul_(self.cfg.context_hebb_gain)
        self.trace_post.mul_(self.decay_minus).add_(self.spikes)

    def _fired(self) -> torch.Tensor:
        """Индексы выстреливших нейронов, дополненные фиктивными до max_spikes."""

        fired = torch.nonzero_static(
            self.spikes, size=self.max_spikes, fill_value=self.n
        ).squeeze(1)

        return torch.where(fired == self.n, self.dummy_rows, fired)

    def _chunk(self, plastic: bool) -> None:
        step = self._compiled_step()

        for t in range(self.CHUNK):
            step(self.ear_input[t], plastic)

    def _compiled_step(self):
        """
        Шаг, собранный torch.compile: десятки мелких операций шага
        сливаются в несколько ядер (время шага — запуск ядер, а не счёт).
        """

        if not self.cfg.compile_step or self.device.type != "cuda":
            return self._step

        if getattr(self, "_compiled", None) is None:
            # Пересборка — при новой форме таблиц и для каждого мозга
            # (тесты создают десятки); по умолчанию лимит 8, дальше молча
            # медленный режим.
            torch._dynamo.config.cache_size_limit = 256
            torch._dynamo.config.recompile_limit = 256
            self._compiled = torch.compile(self._step, dynamic=False)

        return self._compiled

    def recompile(self) -> None:
        """Сбросить собранный шаг и CUDA Graph (поменялась настройка шага)."""

        self._compiled = None
        self._graphs = {}

    def _capture(self, plastic: bool) -> torch.cuda.CUDAGraph:
        """Записать CHUNK шагов в CUDA Graph (один раз на режим)."""

        saved = self._snapshot()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(stream):
            for _ in range(2):
                self._chunk(plastic)

        torch.cuda.current_stream().wait_stream(stream)

        graph = torch.cuda.CUDAGraph()
        # Сборщик мусора во время записи может удалить граф прошлого мозга
        # (тесты, /load) — запись рушится («operation failed due to a
        # previous error during capture»). Убрать мусор до, не собирать во время.
        gc.collect()
        gc.disable()

        try:
            with torch.cuda.graph(graph):
                self._chunk(plastic)
        finally:
            gc.enable()

        # Прогрев и запись не должны оставить след в мозге.
        self._restore(saved)

        return graph

    def run(self, duration_ms: float, sensory: torch.Tensor | None = None) -> None:
        """
        Прожить duration_ms. sensory: [steps, sensory_size] —
        растр sensory-спайков (0/1) с начала окна.
        Длительность округляется до CHUNK шагов.
        """

        steps = int(round(duration_ms / self.cfg.dt_ms))
        chunks = max(1, round(steps / self.CHUNK))
        use_graph = self.device.type == "cuda"
        self.set_clock()

        key = self.plasticity

        if use_graph and key not in self._graphs:
            self._graphs[key] = self._capture(self.plasticity)

        for c in range(chunks):
            begin = c * self.CHUNK

            if sensory is not None and begin < sensory.shape[0]:
                # Растр может быть уже массива (только ухо: его участок первый).
                part = sensory[begin: begin + self.CHUNK]
                self.ear_input.zero_()
                self.ear_input[: len(part), : part.shape[1]].copy_(part)
            else:
                self.ear_input.zero_()

            if self.on_chunk is not None:
                before = self.counts.clone()

            if use_graph:
                self._graphs[key].replay()
            else:
                self._chunk(self.plasticity)


            if self.on_chunk is not None:
                fired = (self.counts[: self.n] > before[: self.n]).nonzero().squeeze(1)
                self.on_chunk(fired.cpu(), self.time_ms + (c + 1) * self.CHUNK * self.cfg.dt_ms)

        self.time_ms += chunks * self.CHUNK * self.cfg.dt_ms

        # Свернуть масштаб eligibility обратно в сами следы.
        self.eligibility.mul_(self.eligibility_scale)
        self.eligibility_scale.fill_(1.0)

    def observe(
        self,
        duration_ms: float,
        sensory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Прожить окно и вернуть частоты (Hz) всех нейронов."""

        self.counts.zero_()

        if self.track_primacy:
            self.primacy_counts.zero_()
            self.primacy_gain.fill_(1.0)

        self.run(duration_ms, sensory)

        steps = max(1, round(duration_ms / self.cfg.dt_ms / self.CHUNK)) * self.CHUNK

        return self.counts[: self.n] * (1000.0 / (steps * self.cfg.dt_ms))

    # ==============================================================
    # DOPAMINE
    # ==============================================================

    def reward(self, amplitude: float) -> None:
        """
        Dopamine-залп: eligibility traces -> изменение весов.

            Δw = lr * dopamine * C

        Отрицательный dopamine (провал ниже фона) ослабляет
        то, что было активно недавно. modulation (норадреналин)
        усиливает изменение: пережитое в возбуждении запоминается крепче.
        """

        if not self.plasticity or amplitude == 0:
            return

        gain = self.cfg.dopamine_learning_rate * amplitude * self.modulation

        plastic = self.net_weight[: self.n_plastic]
        eligibility = self.eligibility[: self.n_plastic]

        if self.dopamine_mask is not None:
            # Пути гиппокампа дофамин не трогает — их пишет encode.
            eligibility = eligibility * self.dopamine_mask

        before = plastic.clone()
        plastic.add_(eligibility, alpha=gain)

        for proj in self.plastic:
            ceiling = getattr(proj, "ceiling", None)

            if ceiling is None:
                proj.weight.clamp_(0.0, proj.cap)
            else:
                proj.weight.clamp_(min=0.0)
                torch.minimum(proj.weight, ceiling, out=proj.weight)

        # Изменение сначала свежее (early) — см. forget() и сон.
        self.early.add_(plastic - before)

        capture = self.cfg.sleep_capture

        if self.consolidating and capture > 0:
            # Synaptic tagging & capture: синапсы, участвовавшие в
            # проигрывании (след eligibility), закрепляют свою свежую
            # часть — пропорционально участию.
            # В fp32: в fp16 граница 1e-12 — ноль, и при нулевых следах
            # (кора молчит во вспышке гиппокампа) выходило 0/0 = NaN.
            tag = eligibility.abs().float()
            tag = tag / tag.max().clamp(min=1e-12)
            self.early.mul_(1.0 - capture * tag)

    def encode(self, gain: float) -> None:
        """
        Запись в гиппокамп (ацетилхолин бодрствования): следы STDP путей
        с modulator = "acetylcholine" сразу становятся весом — эпизод
        запоминается с одного раза (Hasselmo 2006; быстрая пластичность
        CA3, BTSP — Bittner 2017). Во сне ацетилхолина нет — не вызывается.
        """

        if not self.plasticity or gain <= 0 or not self.encoding:
            return

        for proj in self.encoding:
            proj.weight.add_(proj.eligibility, alpha=gain)
            proj.weight.clamp_(0.0, proj.cap)

    def hebbian(self, rate: float) -> None:
        """
        STDP без третьего фактора (пути modulator="hebbian"): накопленный
        след пар спайков сразу становится весом и обнуляется — учит каждое
        услышанное, а не только неожиданное (связи коры, Markram 1997;
        Bi & Poo 1998). Свежее — как всё выученное (early).
        """

        if not self.plasticity or rate <= 0 or not self.hebbian_paths:
            return

        for proj in self.hebbian_paths:
            before = proj.weight.clone()
            proj.weight.add_(proj.eligibility, alpha=rate)
            ceiling = getattr(proj, "ceiling", None)

            if ceiling is None:
                proj.weight.clamp_(0.0, proj.cap)
            else:
                proj.weight.clamp_(min=0.0)
                torch.minimum(proj.weight, ceiling, out=proj.weight)

            proj.early.add_((proj.weight - before).to(proj.early.dtype))
            proj.eligibility.zero_()

    def instar(self, name: str, rates: torch.Tensor, rate: float, covariance: bool = False) -> None:
        """
        Связь по совпадению за окно (instar, Grossberg): у клеток, стрелявших
        в окне, входы тянутся к узору активных пресинапсов —
        Δw = rate · post · (cap · pre − w). Хеббовское правило с насыщением:
        сильнее то, что совпадало чаще.

        covariance — ковариационное правило (Sejnowski 1977; BCM —
        Bienenstock, Cooper & Munro 1982, в зрительной коре — Kirkwood,
        Rioult & Bear 1996): Δw = rate · cap · (post − θ) · pre, θ — свой
        у клетки скользящий порог, средняя её активность за окна (BCM):
        вход, активный при молчащей клетке, слабеет — остаётся то, что
        отличает, а не то, что активно всегда.
        """

        if not self.plasticity:
            return

        proj = self.projection(name)
        pre = (rates.index_select(0, proj.pre.flatten().long()).view(proj.pre.shape) > 0).float()
        post = (rates[proj.post.slice] > 0).float()[:, None]

        if covariance:
            theta = getattr(proj, "theta", None)

            if theta is None:
                theta = proj.theta = torch.zeros_like(post)

            # Скользящее среднее за ~100 окон [не подбиралось].
            theta.mul_(0.99).add_(0.01 * post)
            delta = rate * proj.cap * (post - theta) * pre
            delta = (proj.weight + delta).clamp(0.0, proj.cap) - proj.weight
        else:
            delta = rate * post * (proj.cap * pre - proj.weight)

        proj.weight.add_(delta)
        proj.early.add_(delta.to(proj.early.dtype))

    def error_learning(
        self, names, rates: torch.Tensor, target: torch.Tensor, rate: float, goal_mv: float = 6.0,
    ) -> None:
        """
        Обучение по ошибке (мозжечок: лазающее волокно несёт ошибку, синапс
        меняется на её величину — Marr 1969, Albus 1971, Ito; внутренние
        модели «вижу → двигаю» — Wolpert, Miall & Kawato 1998). Учитель —
        копия моторной команды target (локальные клетки получателя, что
        действительно работали), предсказание — вход по пути от активных
        входов в долях goal_mv (вход, который вместе с «давай» доводит до
        порога): Δw = rate · (команда − предсказание) · pre — дельта-правило.
        names — один путь или несколько к той же зоне (возбуждающий и
        тормозной): предсказание — их сумма, каждый держит свой знак.
        """

        if not self.plasticity:
            return

        projs = [self.projection(n) for n in ([names] if isinstance(names, str) else names)]
        # Средний ток входа, мВ: вес × частота × длительность синаптического
        # тока (e_tau_syn) — столько вход поднимает клетку в среднем.
        tau = self.cfg.e_tau_syn / 1000.0
        pres = [rates.index_select(0, p.pre.flatten().long()).view(p.pre.shape) * tau for p in projs]
        drive = sum((pre * p.weight).sum(1) for pre, p in zip(pres, projs))
        predicted = (drive / goal_mv).clamp(0.0, 1.0)[:, None]
        wanted = torch.zeros_like(predicted)
        wanted[target] = 1.0

        for pre, proj in zip(pres, projs):
            # Шаг — по доле участия входа (pre / сумма), как у дельта-правила
            # с нормировкой: ошибка распределяется по активным входам.
            share = pre / pre.sum(1, keepdim=True).clamp(min=1e-9)
            delta = rate * goal_mv * (wanted - predicted) * share / tau
            # Возбуждающий путь — [0, cap], тормозной (врождённо 0, не
            # пластичный по STDP) — [−cap, 0].
            low, high = (0.0, proj.cap) if proj.plastic else (-proj.cap, 0.0)
            delta = (proj.weight + delta).clamp(low, high) - proj.weight
            proj.weight.add_(delta)

            if proj.plastic:
                proj.early.add_(delta.to(proj.early.dtype))

    def inhibitory_plasticity(self, rates: torch.Tensor) -> None:
        """
        Пластичность торможения (Vogels et al. 2011) на путях vogels: после
        окна (фиксации) сила торможения от соседа j к клетке i меняется на
        rate · r_j · (r_i − норма) (частоты — в долях 100 Гц): клетка,
        стрелявшая вместе с соседом выше нормы, получает от него больше
        торможения, ниже нормы — меньше. Клетки, отвечающие на одно и то
        же, расходятся по разным признакам (декорреляция).
        """

        if not self.plasticity:
            return

        target = self.cfg.inhibitory_target_hz

        for proj in self.projections:
            if not getattr(proj, "vogels", False):
                continue

            pre = rates.index_select(0, proj.pre.flatten().long()).view(proj.pre.shape) / 100.0
            post = (rates[proj.post.slice] - target)[:, None] / 100.0
            proj.weight.sub_(self.cfg.inhibitory_rate * pre * post).clamp_(-proj.cap, 0.0)

    def forget(self) -> None:
        """
        Свежая память затухает: ранняя часть изменений уходит к
        закреплённому весу с полураспадом early_ltp_half_life_ms времени
        мозга, прожитого с прошлого затухания (а не по числу пауз: короткая
        вспышка сна — не фраза). Закреплённое сном (capture) не затухает.
        """

        half_life = self.cfg.early_ltp_half_life_ms

        if not self.plasticity or not self.cfg.forgetting_enabled or half_life <= 0:
            return

        elapsed = self.time_ms - self.forgotten_at_ms
        self.forgotten_at_ms = self.time_ms

        if elapsed <= 0:
            return

        fade = 1.0 - 0.5 ** (elapsed / half_life)
        self.net_weight[: self.n_plastic].sub_(self.early, alpha=fade)
        self.early.mul_(1.0 - fade)

    def rewire(self) -> int:
        """
        Структурная пластичность (synaptic turnover).

        Синапс, который опыт ослабил почти до нуля, отмирает,
        а нейрон отращивает вместо него новый — к случайному
        нейрону той же зоны-источника, со слабым врождённым весом.
        Так сеть со временем меняет саму схему связей,
        а не только их силу.

        Возвращает число выросших синапсов.
        """

        if not self.plasticity:
            return 0

        grown = 0

        for proj in self.plastic:
            # Порог относителен врождённому весу пути: синапс отмирает,
            # только если опыт ослабил его заметно ниже рождения.
            weak = proj.weight < proj.innate_range[0] * self.cfg.prune_fraction
            count = int(weak.sum())

            if not count:
                continue

            pre = proj.pre_pop
            new_pre = torch.randint(
                pre.start, pre.stop, (count,),
                device=self.device, generator=self.generator,
            ).to(torch.int32)
            own = torch.arange(proj.post.start, proj.post.stop, device=self.device)
            own = own[:, None].expand_as(weak)[weak].to(torch.int32)
            new_pre = torch.where(
                new_pre == own,
                pre.start + (new_pre - pre.start + 1) % pre.size,
                new_pre,
            ).to(torch.int32)

            low, high = proj.innate_range
            proj.pre[weak] = new_pre
            proj.weight[weak] = exact16(low + (high - low) * torch.rand(
                count, device=self.device, generator=self.generator
            ))
            proj.birth[weak] = proj.weight[weak].to(proj.birth.dtype)
            proj.early[weak] = 0.0
            proj.eligibility[weak] = 0.0
            grown += count

        if grown:
            self._build_tables(refill=True)
            self.synapses_grown += grown

        return grown

    def reset_cells(self, pop: Population, cells: torch.Tensor) -> int:
        """
        Клетки освободились (место забытого слова отдают новому): их
        выученные синапсы — входящие и исходящие — возвращаются к весу при
        рождении, иначе новое слово унаследует чужие связи. Схема связей
        (кто к кому) остаётся. Возвращает число сброшенных синапсов.
        """

        reset = 0
        ids = (cells.to(self.device) + pop.start).to(torch.int32)

        for proj in self.plastic:
            mask = None

            if proj.post is pop:
                mask = torch.zeros(proj.weight.shape[0], dtype=torch.bool, device=self.device)
                mask[cells.to(self.device)] = True
                mask = mask[:, None].expand_as(proj.weight)

            if proj.pre_pop is pop:
                outgoing = torch.isin(proj.pre, ids)
                mask = outgoing if mask is None else mask | outgoing

            if mask is None:
                continue

            proj.weight[mask] = proj.birth[mask].to(proj.weight.dtype)
            proj.early[mask] = 0.0
            proj.eligibility[mask] = 0.0
            reset += int(mask.sum())

        return reset

    def downscale(self, fraction: float) -> None:
        """
        Сон ослабляет выученное (Tononi): прирост каждого пластичного
        синапса над весом при рождении × (1 − fraction). Сильное остаётся
        сильным относительно слабого, общий уровень падает.
        """

        if not self.plasticity or fraction <= 0:
            return

        plastic = self.net_weight[: self.n_plastic]
        birth = self.birth_weight.to(plastic.dtype)
        gained = (plastic - birth).clamp(min=0.0)
        plastic.sub_(gained * fraction)
        self.early.mul_(1.0 - fraction)

    def homeostasis(self, rates: torch.Tensor) -> None:
        """
        Гомеостаз возбудимости (intrinsic plasticity): если зона
        ответила на слово шире цели — она становится менее
        возбудимой, если уже — более. Так каждая зона держит
        разреженный код при любом размере мозга.
        """

        if not self.plasticity:
            return

        low, high = self.cfg.homeostasis_bias_range

        for spec in self.cfg.areas:
            # target_activity = 0: активность зоны задаёт её вход
            # (лимбическая — само чувство), а не гомеостаз.
            if spec.target_activity <= 0:
                continue

            # С понятиями активность смысла (и памяти с местами слов)
            # задают сами места, а не гомеостаз.
            if self.cfg.concepts_enabled and spec.name == "association":
                continue

            if self.cfg.memory_places and spec.name == "prefrontal":
                continue

            # Своя у клетки подстройка — не до рождения: вход такой зоны
            # (слоговарь — повтор про себя) появляется лишь после лепета, на
            # калибровке она молчит, и подстройка выкручивала бы возбудимость
            # всех клеток (+10 мВ) под вход, которого не будет.
            if spec.cell_homeostasis and self.developing:
                continue

            E = self.areas[spec.name].E
            active = (rates[E.slice] > 0).float().mean()
            error = (spec.target_activity - active) / spec.target_activity
            # До рождения (калибровка) — критический период: подстройка
            # быстрая во всех зонах; потом у зоны своя скорость.
            speed = 1.0 if self.developing else spec.homeostasis
            step = (self.cfg.homeostasis_rate * speed * error).clamp(-1.0, 1.0)

            if spec.cell_homeostasis:
                # У каждой клетки своя норма: выстрелившая теряет
                # возбудимость, молчащая набирает; в среднем клетка
                # стреляет на доле target_activity слов.
                fired = (rates[E.slice] > 0).float()
                step = self.cfg.homeostasis_rate * speed * (spec.target_activity - fired)

            self.bias[E.slice] = (self.bias[E.slice] + step).clamp(low, high)

    def grow(self, links: list[tuple[str, torch.Tensor, torch.Tensor]], count: int) -> int:
        """
        Вырастить синапсы: для каждой связи (проекция, pre, post) —
        pre и post локальные индексы клеток своих зон — count синапсов от
        случайных pre к случайным post, взамен самого слабого входа клетки
        post. Новый синапс слабый (нижний врождённый вес) и учится как все.
        """

        if not self.plasticity or not links:
            return 0

        grown = 0
        # Связи одной проекции растут одним проходом: по связи на вызов —
        # ~1.3 мс, а их сотни за раз. [замер] 20 тыс., чтение быта: 400
        # связей — 500 мс на рост, рост — четверть всего чтения.
        groups: dict[str, list] = {}

        for link in links:
            if len(link[1]) and len(link[2]):
                groups.setdefault(link[0], []).append(link)

        for name, group in groups.items():
            proj = next((p for p in self.projections if p.name == name), None)

            if proj is None:
                continue

            # count случайных пар (pre, post) на каждую связь: номер связи
            # у каждой пары и случайная доля длины её pre и post.
            n_pre = torch.tensor([len(link[1]) for link in group], device=self.device)
            n_post = torch.tensor([len(link[2]) for link in group], device=self.device)
            pre_at = torch.cumsum(n_pre, 0) - n_pre
            post_at = torch.cumsum(n_post, 0) - n_post
            all_pre = torch.cat([link[1] for link in group])
            all_post = torch.cat([link[2] for link in group])
            link_of = torch.arange(len(group), device=self.device).repeat_interleave(count)
            u = torch.rand((2, len(link_of)), device=self.device, generator=self.generator)
            pick = post_at[link_of] + (u[0] * n_post[link_of]).long().clamp(max=n_post[link_of] - 1)
            which = pre_at[link_of] + (u[1] * n_pre[link_of]).long().clamp(max=n_pre[link_of] - 1)
            rows = all_post[pick]
            new_pre = all_pre[which] + proj.pre_pop.start
            # Четвёртое в связи — ветка дендрита: выросшие на одной
            # неожиданности синапсы ложатся на одну ветку.
            # Нет ветки — -1: такой синапс ветку не меняет.
            branches = [int(link[3]) if len(link) > 3 and link[3] is not None else -1 for link in group]
            branch = (
                torch.tensor(branches, device=self.device)[link_of]
                if hasattr(proj, "branch") and max(branches) >= 0 else None
            )
            # Уже есть такая связь — не дублировать.
            fresh = ~(proj.pre[rows] == new_pre[:, None].to(proj.pre.dtype)).any(1)
            rows, new_pre = rows[fresh], new_pre[fresh]
            branch = branch[fresh] if branch is not None else None

            if not len(rows):
                continue

            # Одна клетка может получить несколько синапсов за раз: i-й
            # занимает её i-й по слабости вход (иначе все писались в один
            # самый слабый, и выживал последний).
            order = torch.argsort(rows, stable=True)
            rows, new_pre = rows[order], new_pre[order]
            branch = branch[order] if branch is not None else None
            pair = rows.long() * (1 << 32) + new_pre.long()
            first = torch.ones_like(rows, dtype=torch.bool)
            first[1:] = pair[1:] != pair[:-1]
            rows, new_pre = rows[first], new_pre[first]
            branch = branch[first] if branch is not None else None
            _, counts = torch.unique_consecutive(rows, return_counts=True)
            starts = torch.repeat_interleave(torch.cumsum(counts, 0) - counts, counts)
            nth = torch.arange(len(rows), device=self.device) - starts
            keep = nth < proj.weight.shape[1]
            rows, new_pre, nth = rows[keep], new_pre[keep], nth[keep]
            branch = branch[keep] if branch is not None else None
            cols = torch.argsort(proj.weight[rows].float(), dim=1, stable=True).gather(1, nth[:, None]).squeeze(1)
            proj.pre[rows, cols] = new_pre.to(proj.pre.dtype)
            low = float(exact16(torch.tensor(proj.innate_range[0] if proj.plastic else 0.0)))
            proj.weight[rows, cols] = low

            if branch is not None:
                on = branch >= 0
                proj.branch[rows[on], cols[on]] = branch[on].to(proj.branch.dtype)

            if proj.plastic:
                proj.birth[rows, cols] = low
                proj.early[rows, cols] = 0.0
                proj.eligibility[rows, cols] = 0.0

            grown += len(rows)

        if grown:
            self._build_tables(refill=True)
            self.synapses_grown += grown

        return grown

    def synaptic_scaling(self) -> int:
        """
        Homeostatic synaptic scaling (Turrigiano): суммарный
        входной пластичный вес нейрона не растёт бесконечно,
        иначе «хабы» размывают все ассоциации.
        """

        if not self.plasticity:
            return 0

        scaled = 0

        for proj in self.plastic:
            if not getattr(proj, "scaled", True):
                continue

            total = proj.weight.sum(1)
            budget = (
                proj.weight.shape[1]
                * proj.innate_mean
                * self.cfg.synaptic_scaling_budget
            )
            factor = budget / total.clamp(min=1e-9)

            if getattr(proj, "scale_up", False):
                # В обе стороны — только у клеток, что уже чему-то научились
                # (у молчащей клетки нечего масштабировать).
                learned = total > proj.innate_mean * proj.weight.shape[1]
                factor = torch.where(learned, factor, factor.clamp(max=1.0))
            else:
                factor = factor.clamp(max=1.0)
            proj.weight.mul_(factor[:, None])
            proj.early.mul_(factor[:, None])
            scaled += int((factor < 1.0).sum())

        return scaled

    def presynaptic_scaling(self, names: tuple[str, ...], budget: float) -> int:
        """
        Гетеросинаптическая пластичность (Chistiakova et al. 2014): ресурс
        исходящих синапсов клетки ограничен — суммарный выход клетки по пути
        не больше budget × врождённой суммы. Клетка, связанная со многими
        продолжениями («столица» → Берлин, Мадрид, Москва), даёт каждому
        меньше; связанная с одним («франции» → Париж) — полный вес (эффект
        веера в коре). Возвращает число урезанных клеток.
        """

        if not self.plasticity:
            return 0

        scaled = 0

        for proj in self.plastic:
            if proj.name not in names:
                continue

            pre = proj.pre.flatten().long()
            weight = proj.weight.float().flatten()
            total = torch.zeros(self.n_slots, device=self.device).index_add_(0, pre, weight)
            count = torch.zeros(self.n_slots, device=self.device).index_add_(0, pre, torch.ones_like(weight))
            limit = count * proj.innate_mean * budget
            factor = (limit / total.clamp(min=1e-9)).clamp(max=1.0)
            per_synapse = factor[pre].view_as(proj.weight).to(proj.weight.dtype)
            proj.weight.mul_(per_synapse)
            proj.early.mul_(per_synapse.to(proj.early.dtype))
            scaled += int((factor < 1.0).sum())

        return scaled

    # ==============================================================
    # ЧТЕНИЕ СОСТОЯНИЯ
    # ==============================================================

    def learned_drive(
        self, rates: torch.Tensor, area: str, source: str | None = None
    ) -> torch.Tensor:
        """
        Куда текущее состояние толкает E-нейроны зоны через
        ВЫУЧЕННЫЕ синапсы (вес выше, чем был при рождении):

            drive[post] = Σ rate[pre] * max(0, w - w_при_рождении)

        Это ожидание следующего состояния. Источники — все
        пластичные пути в зону. Вклад рабочей памяти либо
        складывается (context_mode="add"), либо умножает отклик
        на текущее слово (context_mode="gain": контекст выбирает
        среди кандидатов, которых предлагает текущее слово).

        Модулирующие пути (predictive=False, например чувство -> смысл)
        в ожидание не входят; source="limbic" читает только путь из
        этой зоны — куда толкает само чувство.
        """

        target = self.areas[area].E
        local = torch.zeros(target.size, device=self.device)
        context = torch.zeros(target.size, device=self.device)

        for proj in self.plastic:
            if proj.post is not target:
                continue

            if source is not None:
                if proj.pre_pop is not self.areas[source].E:
                    continue
            elif not getattr(proj, "predictive", True):
                continue

            reference = proj.birth if self.cfg.learned_from_birth else proj.innate_max
            learned = (proj.weight - reference).clamp(min=0.0)
            got = rates.index_select(0, proj.pre.flatten()).view_as(learned)
            part = (got * learned).sum(1)

            if proj.source_kind == "memory":
                context += part
            else:
                local += part

        # Один источник — сырой вход от него (без смешения с контекстом).
        if source is not None:
            return local + context

        gain = self.cfg.context_gain

        if self.cfg.context_mode == "gain":
            scale = context.mean().clamp(min=1e-9)
            return local * (1.0 + gain * context / scale)

        return local + gain * context

    def projection(self, name: str) -> Projection:
        return next(p for p in self.projections if p.name == name)

    def weights_between(self, proj: Projection, pre_ids: torch.Tensor, post_ids: torch.Tensor) -> torch.Tensor:
        """Веса синапсов pre_ids -> post_ids (индексы глобальные)."""

        if not len(pre_ids) or not len(post_ids):
            return torch.empty(0, device=self.device)

        rows = post_ids - proj.post.start
        sub_pre = proj.pre[rows].long()
        mask = torch.isin(sub_pre, pre_ids)

        return proj.weight[rows][mask]

    # ==============================================================
    # READ-ONLY РЕЖИМ
    # ==============================================================

    # Динамическое состояние: буферы, которые CUDA Graph
    # читает и пишет по фиксированным адресам. Поэтому их
    # всегда восстанавливают через copy_, а не присваиванием.
    _DYNAMIC = (
        "v", "i_syn", "i_inh", "i_slow", "i_dend", "pool", "area_pool", "adapt", "adapt_slow", "refractory", "spikes",
        "trace_pre", "trace_pre_rise", "ltp_kernel", "trace_post", "counts", "eligibility",
        "eligibility_scale", "dropped", "bias", "interoception", "kick_current", "kick_left", "stp_x", "stp_u4", "stp_x4", "inh_gain",
        "chemistry", "pending", "pending_dend", "tick",
    )

    def _snapshot(self) -> dict:
        return {name: getattr(self, name).clone() for name in self._DYNAMIC}

    def _restore(self, saved: dict) -> None:
        for name, value in saved.items():
            target = getattr(self, name)

            # Старое сохранение другой формы (дендрит без веток) — с нуля.
            if target.shape != value.shape:
                target.zero_()
                continue

            target.copy_(value)

    @contextmanager
    def frozen(self):
        """
        Диагностика без следа: пластичность выключена,
        после выхода состояние нейронов возвращается как было.
        Веса не меняются вообще.
        """

        saved = self._snapshot()
        time_ms = self.time_ms
        plasticity = self.plasticity
        self.plasticity = False

        try:
            yield
        finally:
            self._restore(saved)
            self.time_ms = time_ms
            self.plasticity = plasticity

    def feel(self, area: str, neurons: torch.Tensor | None, amplitude) -> None:
        """
        Подать внутренний сигнал: neurons — индексы E-нейронов зоны
        (локальные); amplitude — число или тензор (своя сила каждой клетке).
        """

        if area not in self.areas:
            return

        E = self.areas[area].E
        self.interoception[E.slice] = 0.0

        if neurons is not None:
            self.interoception[E.start + neurons] = amplitude

    def kick(self, area: str, neurons: torch.Tensor, amplitude: float, ms: float) -> None:
        """Толчок E-клеткам зоны на ms миллисекунд (сигнал начала)."""

        if area not in self.areas:
            return

        E = self.areas[area].E
        self.kick_current.zero_()
        self.kick_current[E.start + neurons] = amplitude
        self.kick_left.fill_(max(1.0, round(ms / self.cfg.dt_ms)))

    def silence(self, area: str) -> None:
        """
        Погасить устойчивую активность зоны (и её ток CAN): медленная волна
        сна — кора целиком замолкает (down-state).
        """

        if area not in self.areas:
            return

        cells = self.areas[area].cells
        self.i_syn[cells] = 0.0
        self.i_slow[cells] = 0.0
        self.adapt[cells] = 0.0
        self.adapt_slow[cells] = 0.0
        self.v[cells] = self.cfg.v_rest

    def quiet(self, area: str) -> None:
        """Погасить активность зоны (только внутри frozen())."""

        cells = self.areas[area].cells
        self.i_syn[cells] = 0.0
        self.i_slow[cells] = 0.0
        self.v[cells] = self.cfg.v_rest

    # ==============================================================
    # СОХРАНЕНИЕ
    # ==============================================================

    def state_dict(self) -> dict:
        return {
            "dynamic": self._snapshot(),
            "time_ms": self.time_ms,
            "forgotten_at_ms": self.forgotten_at_ms,
            "synapses_grown": self.synapses_grown,
            "synapses": {
                "net_pre": self.net_pre,
                "net_weight": self.net_weight,
                "ear_pre": self.ear_pre,
                "birth_weight": self.birth_weight,
                "early": self.early,
                "net_branch": self.net_branch,
                "innate_mean": {p.name: p.innate_mean for p in self.plastic},
            },
            "generator": self.generator.get_state(),
        }

    def load_state_dict(self, state: dict) -> None:
        self._restore(state["dynamic"])
        self.time_ms = state["time_ms"]
        self.forgotten_at_ms = state.get("forgotten_at_ms", self.time_ms)
        self.synapses_grown = state.get("synapses_grown", 0)

        synapses = state["synapses"]

        # copy_ в те же тензоры: view проекций остаются валидными.
        for name in ("net_pre", "net_weight", "ear_pre"):
            getattr(self, name).copy_(synapses[name])

        if "birth_weight" in synapses:
            self.birth_weight.copy_(synapses["birth_weight"])

        if "net_branch" in synapses and synapses["net_branch"].shape == self.net_branch.shape:
            self.net_branch.copy_(synapses["net_branch"])

        # Старые сохранения: всё выученное считается закреплённым.
        if "early" in synapses:
            self.early.copy_(synapses["early"])
        else:
            self.early.zero_()

        for proj in self.plastic:
            proj.innate_mean = synapses["innate_mean"][proj.name]

        # Схема связей могла измениться (rewire) — таблицы заново.
        self._build_tables()

        self.generator.set_state(state["generator"].cpu())

    # ==============================================================
    # STATS
    # ==============================================================

    def synapse_count(self) -> int:
        return (
            self.n_synapses + self.ear_weight.numel()
            + sum(p.n_synapses for p in self.procedural)
        )

    def memory_bytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for t in (
                self.net_pre, self.net_post, self.net_weight,
                self.ear_pre, self.ear_post, self.ear_weight,
                self.eligibility, self.early, self.birth_weight,
            )
        ) + sum(t.nbytes() for t in (self.out_all, self.out_plastic, self.in_plastic)) + (
            self.proc_rows.nbytes() if self.proc_rows is not None else 0
        )
