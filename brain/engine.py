from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import gc
import math

import torch

from brain import kernels as _kernels

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


# Типы клеток колонки: главные и тормозные (у человека — PV, SST, VIP и
# LAMP5/нейроглиаформные; Hodge et al. 2019). Номер — cell_type.
KINDS = ("E", "PV", "SOM", "VIP", "LAMP5")
INHIBITORY_KINDS = KINDS[1:]


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
    # Колонка по слоям (cfg.layers): слой -> {"E", "PV", "SOM", "VIP", "LAMP5"}.
    # E, I, SOM, VIP, LAMP5 — популяции слоя 2/3.
    layers: dict | None = None
    LAMP5: Population | None = None

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

    def prepare_compact(self, capacity: int, n: int, dummy_neurons: int) -> None:
        """
        Буферы плотного списка событий (ядро brain/csrc/events.cu): ёмкость
        постоянна; пустые ячейки — разные фиктивные синапсы и нейроны (как в
        events, чтобы не драться за один адрес).
        """

        device = self.perm.device
        cells = torch.arange(capacity, device=device)
        self.capacity, self.n = capacity, n
        self.empty_synapses = (self.pad + cells % self.dummies).long()
        self.empty_pre = (n + cells % dummy_neurons).long()
        self.out_synapses = self.empty_synapses.clone()
        self.out_pre = self.empty_pre.clone()
        self.total = torch.zeros(1, dtype=torch.long, device=device)

    def compact(self, neurons: torch.Tensor, dropped: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Синапсы строк neurons подряд, без пустых ячеек; лишнее — в dropped."""

        neurons = neurons.long()
        lengths = torch.where(neurons < self.n, self.count.index_select(0, neurons), 0).long()
        offsets = torch.cumsum(lengths, 0) - lengths
        self.out_synapses.copy_(self.empty_synapses)
        self.out_pre.copy_(self.empty_pre)
        _kernels.compact_events(neurons, self.n, self.start, self.count, self.perm, offsets,
                                self.out_synapses, self.out_pre)
        dropped.add_((lengths.sum() - self.capacity).clamp(min=0))
        return self.out_synapses, self.out_pre

    def compact_head(self, neurons: torch.Tensor, dropped: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Как compact, но без заполнения пустых ячеек: ядра шага (deliver,
        ltd, ltp) читают только первые total событий. Сброс всего буфера
        (два копирования по ёмкости) стоил больше, чем сами события.
        """

        neurons = neurons.long()
        lengths = torch.where(neurons < self.n, self.count.index_select(0, neurons), 0).long()
        offsets = torch.cumsum(lengths, 0) - lengths
        _kernels.compact_events(neurons, self.n, self.start, self.count, self.perm, offsets,
                                self.out_synapses, self.out_pre)
        torch.sum(lengths, 0, keepdim=True, out=self.total)
        dropped.add_((self.total - self.capacity).clamp(min=0).squeeze(0))
        return self.out_synapses, self.out_pre, self.total

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

            fraction = cfg.inhibitory_fraction if getattr(spec, "local_inhibitory", -1.0) < 0 else spec.local_inhibitory

            if fraction != cfg.inhibitory_fraction:
                # Своя доля местных тормозных (0 — только 2 немых места: у
                # ядра их нет, но зона всегда с E и I).
                n_i = max(2, int(n * fraction))
                n_e = n - n_i

            if getattr(spec, "cells", 0) > 0:
                # Ядро с заданным числом главных клеток (реле — по рецепторам).
                n_e = spec.cells
                n_i = max(2, round(n_e * fraction / (1.0 - fraction))) if fraction < 1 else 2
                n = n_e + n_i

            if self._has_layers(spec.name):
                area, start = self._column(spec, max(n, getattr(cfg, "min_column_neurons", 0)), start)
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

        cfg = self.cfg
        cells = list(cfg.column_cells)

        if name in cfg.agranular_areas:
            cells = [c for c in cells if c[0] != "L4"]
        elif name in getattr(cfg, "thin_l4_areas", ()):
            # Тонкий L4 — доля клеток колонки thin_l4_share (Yamawaki 2014).
            rest = sum(e + i for layer, e, i in cells if layer != "L4")
            l4 = next(e + i for layer, e, i in cells if layer == "L4")
            scale = cfg.thin_l4_share / (1.0 - cfg.thin_l4_share) * rest / l4
            cells = [(layer, e * scale, i * scale) if layer == "L4" else (layer, e, i) for layer, e, i in cells]

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
        shares = self._layer_shares()
        layers = {}

        for layer, n_e, n_i in cells:
            group = {}
            sizes = [("E", max(4, round(n * n_e / total)))]
            inhibitory = n * n_i / total
            sizes += [(kind, max(1, round(inhibitory * share))) for kind, share in shares[layer].items()]

            for kind, size in sizes:
                group[kind] = Population(f"{spec.name}.{layer}.{kind}", start, size)
                start += size

            layers[layer] = group

        top = layers["L23"]
        area = Area(spec.name, spec.title, spec.kind, top["E"], top["PV"], top.get("SOM"), top.get("VIP"), layers, top.get("LAMP5"))
        return area, start

    def _layer_shares(self) -> dict[str, dict[str, float]]:
        """Доли тормозных типов слоя среди построенных (cfg.column_kinds), в сумме 1."""

        kinds = getattr(self.cfg, "column_kinds", KINDS)
        out = {}

        for layer, *v in self.cfg.layer_interneurons:
            share = {kind: s for kind, s in zip(INHIBITORY_KINDS, v) if kind in kinds}
            # Полная колонка — доли как в данных (сумма ≈ 1 с округлением).
            total = sum(share.values()) if len(share) < len(INHIBITORY_KINDS) else 1.0
            out[layer] = {kind: s / total for kind, s in share.items()}

        return out

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
            if area.SOM is None and area.layers is None:
                continue

            for _, group in self._typed_groups(area):
                if area.layers is not None:
                    # E колонки — клетки человека (Allen; config column_e_*).
                    self.tau_m[group["E"].slice] = cfg.column_e_tau_m
                    self.adapt_b[group["E"].slice] = cfg.e_adapt_b * cfg.column_e_adaptation_index / cfg.e_adaptation_index

                for kind, tau, index in (
                    ("PV", cfg.pv_tau_m, cfg.pv_adaptation_index),
                    ("SOM", cfg.som_tau_m, cfg.som_adaptation_index),
                    ("VIP", cfg.vip_tau_m, cfg.vip_adaptation_index),
                    ("LAMP5", cfg.lamp5_tau_m, cfg.lamp5_adaptation_index),
                ):
                    if kind not in group:
                        continue
                    pop = group[kind]
                    self.tau_m[pop.slice] = tau
                    self.adapt_b[pop.slice] = cfg.e_adapt_b * index / cfg.e_adaptation_index

                if "SOM" in group:
                    self.is_som[group["SOM"].slice] = True

                for number, kind in enumerate(KINDS):
                    if kind in group:
                        self.cell_type[group[kind].slice] = number

        # Кратковременная пластичность по типам (column_stp) и ядра шага —
        # у всех клеток с типом (E, PV, SOM, VIP, LAMP5), не только при SOM:
        # с колонкой E + PV (column_kinds) прежнее условие «есть SOM»
        # выключало пластичность синапсов колонок и быстрые ядра.
        self.has_som = bool((self.cell_type >= 0).any())
        self.typed_pre = self.cell_type >= 0
        # Кратковременная пластичность по парам типов (cfg.column_stp): у
        # клетки-источника — U, восстановление и облегчение к каждому типу
        # получателя (столбцы KINDS).
        kinds = KINDS
        n = len(KINDS)
        self.stp_U4 = torch.ones(self.n_slots, n, device=self.device)
        self.stp_rec4 = torch.zeros(self.n_slots, n, device=self.device)
        self.stp_fac4 = torch.zeros(self.n_slots, n, device=self.device)

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
            if getattr(spec, "tau_m", None) is not None:
                self.tau_m[self.areas[spec.name].E.slice] = spec.tau_m
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
        # Медленное торможение LAMP5 (нейроглиаформные): свой ток со своим
        # спадом; скачок τm/τ — тот же интеграл PSP, что у быстрого синапса.
        lamp5_tau = getattr(cfg, "lamp5_tau_syn", 100.0)
        self.decay_ngf = math.exp(-dt / lamp5_tau)
        self.ngf_scale = self.tau_m / lamp5_tau
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
        specs = {spec.name: spec for spec in cfg.areas}

        for area in self.areas.values():
            if area.layers is None:
                continue

            gain_external = self._column_gains(specs[area.name])[0]
            external = {layer: (e, i) for layer, e, i in cfg.external_inputs}
            gaps = {"E": cfg.column_e_gap_mv, "PV": cfg.pv_gap_mv, "SOM": cfg.som_gap_mv, "VIP": cfg.vip_gap_mv,
                    "LAMP5": cfg.lamp5_gap_mv}
            charge = self._potjans_charge()

            for layer, group in area.layers.items():
                for kind, pop in group.items():
                    inputs = external[layer][0 if kind == "E" else 1]
                    self.noise_mean[pop.slice] = inputs * cfg.external_rate_hz * dt / 1000.0
                    # Заряд PSP (мВ·мс) = вес × τm клетки — как у Potjans
                    # относительно расстояния до порога.
                    tau = float(self.tau_m[pop.start])
                    gain = dict(zip(KINDS, gain_external))[kind]
                    # Своё у зоны (зона, слой, тип, ×) — иначе общее (слой, тип, ×).
                    own = [e[-1] for e in cfg.column_gain_external_pop if len(e) == 4 and tuple(e[:3]) == (area.name, layer, kind)]
                    gain *= own[0] if own else next(
                        (e[-1] for e in cfg.column_gain_external_pop if len(e) == 3 and (e[0], e[1]) == (layer, kind)), 1.0,
                    )
                    self.noise_gain[pop.slice] = gain * charge * gaps[kind] / cfg.potjans_gap_mv / tau

        self.noise_std = self.noise_mean.sqrt()
        # Постоянный ток (мВ): недостающие в малой зоне соседи колонки (см.
        # _wire_column) — средняя часть их входа.
        self.dc_mv = torch.zeros(self.n_slots, device=self.device)
        self.dc_inh_mv = torch.zeros(self.n_slots, device=self.device)
        # Торможение проводимостью — у клеток колонок (cfg.layers).
        self.conductance_cells = torch.zeros(self.n_slots, dtype=torch.bool, device=self.device)

        for area in self.areas.values():
            # Анатомический мозг — у всех клеток: реверсия ГАМК одна для всех
            # нейронов, ниже неё торможение клетку не тянет.
            if area.layers is not None or getattr(cfg, "anatomy", False):
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

    def _psp_peak(self, pop: Population, tau_syn: float | None = None) -> float:
        """
        Пик PSP (мВ) клетки популяции от веса 1 — тем же шагом, что в _step:
        ток прыгает на τm/τsyn и спадает за τsyn, мембрана копит его за τm.
        Вес связи = измеренный PSP / пик. tau_syn — у медленного синапса
        (LAMP5), иначе — синапс клетки.
        """

        tau_m = float(self.tau_m[pop.start])
        tau_syn = float(self.tau_syn[pop.start]) if tau_syn is None else tau_syn
        dt = self.cfg.dt_ms
        current, v, peak = tau_m / tau_syn, 0.0, 0.0

        for _ in range(int(10 * max(tau_m, tau_syn) / dt) + 1):
            v += (current - v) * dt / tau_m
            current *= math.exp(-dt / tau_syn)
            peak = max(peak, v)

        return peak

    def _data_path(self, path) -> bool:
        """Врождённый путь между колонками анатомического мозга — веса по данным."""

        src, dst = self.areas.get(path.src), self.areas.get(path.dst)
        return (
            bool(getattr(self.cfg, "anatomy", False)) and (not path.plastic or getattr(path, "learned", False))
            and src is not None and dst is not None
            and src.layers is not None and dst.layers is not None and src is not dst
            and getattr(path, "psp_mv", None) is None and path.k > 0
        )

    def _external_minus_modeled(self) -> None:
        """
        Внешний вход Potjans — всё, что приходит в клетку извне колонки,
        включая таламус и другие зоны (Table 3). Явные пути модели — уже
        его часть: внешних входов остаётся K_ext − явные (так у Schmidt et al.
        2018 в мультиобластной модели). Анатомический мозг.
        """

        cfg = self.cfg

        if not getattr(cfg, "anatomy", False):
            return

        external = {layer: (e, i) for layer, e, i in cfg.external_inputs}

        for area in self.areas.values():
            if area.layers is None:
                continue

            for layer, group in area.layers.items():
                for kind, pop in group.items():
                    total = external[layer][0 if kind == "E" else 1]
                    left = max(0.0, total - self.modeled_inputs.get(pop.name, 0.0)) / total
                    self.noise_mean[pop.slice] *= left

        self.noise_std = self.noise_mean.sqrt()

    def _thalamic(self, path, dst: Area) -> bool:
        """Путь таламуса в колонку анатомического мозга."""

        direction = getattr(path, "direction", "") or next(
            (d for s, t, d in getattr(self.cfg, "pathway_directions", ()) if (s, t) == (path.src, path.dst)), "",
        )
        return bool(getattr(self.cfg, "anatomy", False)) and direction == "thalamic" and dst.layers is not None

    def _thalamic_path(self, path, pre: Population, post: Population, inhibitory: bool) -> tuple[int, float]:
        """
        Таламус → колонка по данным: ~85 таламических клеток на клетку L4
        (Bruno & Sakmann 2006), единичный ВПСП у взрослых in vivo 1.0 ± 0.2 мВ
        (Schoonover et al. 2014, n 11; у молодых — ~0.6, Bruno & Sakmann).
        Синапс депрессирует (PPR 0.67 при 10–20 Гц, там же) — пока без
        кратковременной пластичности [долг: τ восстановления]; на тормозные
        L4 — ~56 входов (Potjans: 0.062 × 902), синапс на быстрые тормозные в
        2–3 раза сильнее (Cruikshank, Lewis & Connors 2007) — ×2. Агранулярная
        зона — в L2/3 (моторный таламус → L2/3 и L5, Hooks et al. 2013). Масштаб —
        как у межзонных путей: вес × √(k_data / k), недостающее — ток по частоте
        ядра in vivo.
        """

        from brain import anatomy

        cfg = self.cfg
        k_data = 56.0 if inhibitory else 85.0

        if post.name.split(".")[0] in getattr(cfg, "thin_l4_areas", ()):
            # Моторная кора: таламических синапсов меньше, на тормозные — нет
            # (Bopp et al. 2017; config thin_l4_*).
            if inhibitory:
                return 0, 0.0

            k_data *= cfg.thin_l4_thalamic_ratio

        self.modeled_inputs[post.name] = self.modeled_inputs.get(post.name, 0.0) + k_data
        psp = 2.0 if inhibitory else 1.0
        k = max(1, min(round(k_data), pre.size))
        weight = psp / self._psp_peak(post)
        root = math.sqrt(max(1.0, k_data / k))
        rate = anatomy.RATES_HZ.get(path.src, 10.0) / 1000.0
        tau = float(self.tau_m[post.start])
        self._missing(post, pre, k_data * rate * weight * tau * (1.0 - 1.0 / root), rate)
        return k, weight * root

    def _missing(self, post: Population, pre: Population, mv: float, rate_khz: float, ach: bool = False,
                 slow: bool = False) -> None:
        """
        Входы недостающих соседей (клеток в модели меньше, чем в мозге):
        средний ток mv при частоте источника rate_khz. Он живой — следует
        текущей частоте построенной популяции-источника (недостающие клетки
        той же популяции стреляют так же, как построенные; среднее поле), а
        не постоянный по частоте in vivo: постоянный (70–90% входа) рвал
        обратные связи — колонки и CA3 застревали в «горящем» состоянии.
        ach — путь, который ацетилхолин подавляет (PathwaySpec.ach_suppressed):
        недостающие синапсы подавляются так же, как построенные (иначе у
        CA3 → CA3 подавлялось бы 18% входа из 6 тыс. связей, а 82% шли бы
        наяву в полную силу, как во сне).
        slow — объёмное торможение LAMP5: ток идёт через медленный фильтр
        (lamp5_tau_syn), как облако GABA вокруг аксона (Oláh et al. 2009).
        """

        if mv == 0.0:
            return

        if rate_khz <= 0:
            # Источник по данным молчит — средний вход ноль.
            return

        self.mean_field.append((post, pre, mv / rate_khz, rate_khz, ach, slow))

    def _init_nmda(self) -> None:
        """
        NMDA у клеток E колонок (cfg.nmda): медленный ток от тех же
        возбуждающих входов (построенных и среднего поля), с блоком магнием.
        Сила — по пику: при −70 мВ NMDA даёт nmda_share_at_rest пика EPSP
        (Hunt et al. 2023), то есть r · B(−70) · пик(τ_NMDA) = доля · пик(τ_AMPA).
        """

        cfg = self.cfg
        self.nmda_r = torch.zeros(self.n_slots, device=self.device)
        self.i_nmda = torch.zeros(self.n_slots, device=self.device)
        self.nmda_mf = torch.zeros(self.n_slots, device=self.device)
        self.nmda_on = bool(getattr(cfg, "nmda", False) and getattr(cfg, "anatomy", False) and self.conductance)
        tau = getattr(cfg, "nmda_tau_ms", 100.0)
        self.decay_nmda = math.exp(-cfg.dt_ms / tau)

        if not self.nmda_on:
            return

        block = 1.0 / (1.0 + cfg.mg_mm / 3.57 * math.exp(0.062 * 70.0))

        for area in self.areas.values():
            if area.layers is None:
                continue

            for group in area.layers.values():
                pop = group["E"]
                r = cfg.nmda_share_at_rest * self._psp_peak(pop) / (block * self._psp_peak(pop, tau))
                self.nmda_r[pop.slice] = r

        # Скачок τm/τ — тот же интеграл, что у быстрого синапса того же веса.
        self.nmda_scale = self.nmda_r * self.tau_m / tau

    def _build_mean_field(self) -> None:
        """Матрицы среднего поля: популяция-получатель ← популяция-источник (мВ на кГц)."""

        terms = getattr(self, "mean_field", [])
        pres = sorted({(p.start, p.size) for _, p, *_ in terms})
        posts = sorted({(p.start, p.size) for p, *_ in terms})
        pre_index = {key: i for i, key in enumerate(pres)}
        post_index = {key: i for i, key in enumerate(posts)}
        P, Q = len(pres), len(posts)
        dev = self.device
        self.mf_pre_id = torch.full((self.n_slots,), P, dtype=torch.long, device=dev)
        self.mf_post_id = torch.full((self.n_slots,), Q, dtype=torch.long, device=dev)
        self.mf_size = torch.ones(P + 1, device=dev)
        self.mf_rate = torch.zeros(P + 1, device=dev)
        self.mf_counts = torch.zeros(P + 1, device=dev)
        self.mf_exc = torch.zeros(Q + 1, P + 1, device=dev)
        self.mf_inh = torch.zeros(Q + 1, P + 1, device=dev)
        # Доля возбуждения, которую снимает ацетилхолин (пути ach_suppressed).
        self.mf_ach = torch.zeros(Q + 1, P + 1, device=dev)
        # Объёмное торможение LAMP5 — отдельно: через медленный фильтр.
        self.mf_slow = torch.zeros(Q + 1, P + 1, device=dev)
        # Тип клетки получателя (столбец кратковременной пластичности;
        # len(KINDS) — клетка без типа: синапс без пластичности).
        n = len(KINDS)
        self.mf_post_type = torch.full((Q + 1,), n, dtype=torch.long, device=dev)

        for (start, size), q in post_index.items():
            kind = int(self.cell_type[start])
            self.mf_post_type[q] = kind if kind >= 0 else n

        self.mf_gain = torch.ones(P + 1, n + 1, device=dev)

        for (start, size), i in pre_index.items():
            self.mf_pre_id[start: start + size] = i
            self.mf_size[i] = size

        for (start, size), q in post_index.items():
            self.mf_post_id[start: start + size] = q

        for post, pre, coeff, rate, ach, slow in terms:
            q, i = post_index[(post.start, post.size)], pre_index[(pre.start, pre.size)]
            (self.mf_slow if slow else self.mf_exc if coeff > 0 else self.mf_inh)[q, i] += coeff

            if ach and coeff > 0:
                self.mf_ach[q, i] += coeff * self.cfg.ach_suppression
            # Старт — частота по данным (как прежний постоянный ток).
            self.mf_rate[i] = rate

        # Сглаживание частоты источника: недостающие соседи передают
        # медленную среднюю частоту своей популяции, а не её быстрые
        # колебания — их колебания у каждой клетки свои, не общие. С 10 мс
        # флуктуации построенных клеток рассылались всем клеткам как общий вход,
        # и кора колебалась синхронно (выброс — тишина каждые 20–40 мс).
        self.mf_decay = math.exp(-self.cfg.dt_ms / getattr(self.cfg, "mean_field_tau_ms", 50.0))
        self.mf_on = bool(terms)
        self.mf_slow_on = bool(self.mf_slow.any())
        self.ngf_mf = torch.zeros(self.n_slots, device=dev)

    def _cortical_path(self, path, pre: Population, post: Population) -> tuple[int, float]:
        """
        Путь между колонками по данным: входов из других зон у клетки — 21%
        всех (Markov et al. 2011, FLN; так же у Schmidt et al. 2018); все —
        местные колонки плюс внешние Potjans (K_ext — всё, что не из колонки
        1 мм²). Прежде — 0.21 / 0.79 от местных: 79% у Markov — вся своя зона,
        а не колонка, и у L4 человека (местных связей почти нет, L4 → L4 p
        0.007) межзонных выходило ~260 — путь V1 → буквы был немым. Поровну
        между путями, входящими в популяцию (доли по парам зон у нас не
        измерены). PSP — как E → E у Potjans (0.15 мВ). Масштаб — как
        внутри колонки: вес × √(k_data / k), недостающее — ток по частоте
        слоя-источника in vivo. Возвращает (k, вес).
        """

        cfg = self.cfg
        local = self.local_e_inputs.get(post.name, 0.0)
        parts = post.name.split(".")
        layer_post = parts[1] if len(parts) == 3 else "L23"
        external = dict((l, (e, i)) for l, e, i in cfg.external_inputs).get(layer_post, (0, 0))
        outside = external[0] if parts[-1] == "E" else external[1]
        k_data = 0.21 * (local + outside) / max(1, self.cortical_inputs.get(post.name, 1))
        self.modeled_inputs[post.name] = self.modeled_inputs.get(post.name, 0.0) + k_data
        k = max(1, min(round(k_data), pre.size))
        weight = cfg.ee_psp_mv / self._psp_peak(post)
        root = math.sqrt(max(1.0, k_data / k))
        layer = pre.name.split(".")[1] if pre.name.count(".") == 2 else "L23"
        rate = dict((l, e) for l, e, *_ in cfg.column_target_hz).get(layer, 1.0) / 1000.0
        tau = float(self.tau_m[post.start])
        self._missing(post, pre, k_data * rate * weight * tau * (1.0 - 1.0 / root), rate)
        return k, weight * root

    def _laminar(self, path, src: Area, dst: Area) -> tuple[Population, Population, Population, bool]:
        """
        Из какого слоя в какой идёт путь (ARCHITECTURE 10, «Связи между зонами
        по слоям»; cfg.pathway_directions): прямой — из L2/3 в L4 (E и PV —
        прямое торможение), у агранулярной зоны — в L2/3; обратный — из L5 на
        дендриты L2/3 (слой 1) и VIP; боковой — из L2/3 в L2/3; к ядрам — из
        L5. Без слоёв — как раньше: E → E и I.
        Возвращает (пре, пост, тормозные получатели, на дендриты ли).
        """

        direction = getattr(path, "direction", "") or next(
            (d for s, t, d in getattr(self.cfg, "pathway_directions", ()) if (s, t) == (path.src, path.dst)),
            "lateral",
        )
        pre, post, inhibitory, top_down = src.E, dst.E, dst.I, False

        if direction == "ct" and src.layers is not None:
            # Кортикоталамический путь — из L6 (Sherman & Guillery).
            return src.layers["L6"]["E"], dst.E, dst.I, False

        if src.layers is not None and (direction == "fb" or dst.layers is None and dst.SOM is None):
            # Обратные пути и выход к ядрам — из L5.
            pre = src.layers["L5"]["E"]

        if dst.layers is not None:
            if direction in ("ff", "thalamic") and "L4" in dst.layers:
                post, inhibitory = dst.layers["L4"]["E"], dst.layers["L4"]["PV"]
            elif direction == "fb":
                # Обратный путь — на ветки E и на VIP; без VIP в колонке
                # (cfg.column_kinds) тормозной ветки нет.
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

    def _column_gains(self, spec) -> tuple:
        """Множители согласования колонки: свои у зоны (стенд) или общие."""

        cfg = self.cfg
        own = tuple(getattr(spec, "column_gains", ()) or ())

        if len(own) == 14:
            return own[0:5], own[5:9], own[9:13], own[13]

        return cfg.column_gain_external, cfg.column_gain_from_e, cfg.column_gain_to_e, cfg.column_gain_ii

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
        self.local_e_inputs = getattr(self, "local_e_inputs", {})
        order = [layer for layer, _, _ in cfg.column_cells]
        # Клеток в слоях колонки этой зоны (тонкий L4 — меньше источников).
        column = {layer: (e, i) for layer, e, i in self._column_layers(spec.name)}
        shares = {layer: dict(zip(INHIBITORY_KINDS, v)) for layer, *v in cfg.layer_interneurons}
        # Соседи стреляют, как в покое у бодрствующих (column_target_hz).
        rates = {layer: dict(zip(KINDS, v)) for layer, *v in cfg.column_target_hz}
        ee_layer = dict(cfg.layer_ee_psp_mv)
        ei_psp = {layer: dict(zip(INHIBITORY_KINDS, v)) for layer, *v in cfg.layer_ei_psp_mv}
        ii_psp = {src: dict(zip(KINDS, v)) for src, *v in cfg.inhibitory_psp_mv}
        # Вероятности по типам — Campagnola (local_links).
        to_e = {src: p for src, dst, p, _ in cfg.local_links if dst == "E"}
        from_e = {dst: p for src, dst, p, _ in cfg.local_links if src == "E"}
        among = {(src, dst): p for src, dst, p, _ in cfg.local_links if src != "E" and dst != "E"}
        # Нормировка к Potjans — по синаптическим типам; LAMP5 действует
        # объёмно (Oláh et al. 2009), его вероятность — по данным как есть.
        types = ("PV", "SOM", "VIP")
        _, gain_from_e, gain_to_e, gain_ii = self._column_gains(spec)

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

                        if src_ei == dst_ei == "E":
                            # Где у человека измерено иначе (L4 → L4 почти нет).
                            p = next((q for a, b, q in cfg.column_ee_probability_human
                                      if (a, b) == (src_layer, dst_layer)), p)

                        # Тормозная доля — по типам (среднее по слою = Potjans).
                        if src_kind == "LAMP5" and src_layer == dst_layer:
                            # Выход LAMP5 внутри слоя — как измерено: объёмное
                            # действие на соседей (Oláh et al. 2009, L2/3). Вход
                            # в LAMP5 — обычные синапсы: вероятность близких пар
                            # (E → LAMP5 0.36, 4 из 11) нормируется к Potjans,
                            # как у PV/SOM/VIP (ниже). Без нормировки у LAMP5
                            # было в 2.4 раза больше входов E, чем у SOM, — 50–95 Гц
                            # при цели 3.5.
                            p = to_e[src_kind] if dst_ei == "E" else among.get((src_kind, dst_kind), 0.0)
                        elif src_kind == "LAMP5":
                            # Облако аксона нейроглиаформной клетки — плотное и
                            # местное, в своём слое (Oláh et al. 2009: соседи
                            # ближе 100 мкм): между слоями объёмного действия нет.
                            # (Число Potjans × 0.87/среднее давало 0.35–0.47 —
                            # слои гасили друг друга по очереди.)
                            p = 0.0
                        elif src_ei == "E" and dst_ei == "I":
                            p *= from_e[dst_kind] / mean(from_e, dst_layer)
                        elif src_ei == "I" and dst_ei == "E":
                            p *= to_e[src_kind] / mean(to_e, src_layer)
                        elif src_ei == "I":
                            pair = among.get((src_kind, dst_kind), 0.0)
                            # Между слоями LAMP5 — то же отношение к среднему.
                            norm = sum(
                                shares[src_layer][a] * shares[dst_layer][b] * among.get((a, b), 0.0)
                                for a in types for b in types
                            ) / (sum(shares[src_layer][t] for t in types) * sum(shares[dst_layer][t] for t in types))
                            p *= pair / norm if norm > 0 else 0.0

                        if p <= 0:
                            continue

                        # Согласование с частотами in vivo (column_gain_*).
                        if src_ei == "E" and dst_ei == "I":
                            gain = gain_from_e[INHIBITORY_KINDS.index(dst_kind)]
                        elif src_ei == "I" and dst_ei == "E":
                            gain = gain_to_e[INHIBITORY_KINDS.index(src_kind)]
                        elif src_ei == "I":
                            # LAMP5 — без поправки на срезанные аксоны (его
                            # вероятность — объёмная, Oláh 2009).
                            gain = 1.0 if src_kind == "LAMP5" else gain_ii
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

                        if cfg.column_contacts_on and not (src_kind == "LAMP5" and src_layer == dst_layer):
                            # Синапсы Potjans → связанные клетки (config column_contacts_*);
                            # выход LAMP5 в слое — доля соседей под облаком ГАМК, не синапсы.
                            if src_ei == dst_ei == "E":
                                contacts = next((m for a, b, m in cfg.column_contacts_ee if (a, b) == (src_layer, dst_layer)),
                                                cfg.column_contacts_ee_default)
                            elif src_ei == "E":
                                contacts = cfg.column_contacts_ei
                            elif dst_ei == "E":
                                contacts = dict(cfg.column_contacts_ie).get(src_kind, cfg.column_contacts_ie_default)
                            else:
                                contacts = cfg.column_contacts_ii
                            k_data /= contacts
                            p /= contacts

                        if src_ei == dst_ei == "E" and src_layer == dst_layer == "L23":
                            # Местных партнёров у человека — подбор (config column_ee_partners_l23).
                            k_data *= cfg.column_ee_partners_l23
                            p *= cfg.column_ee_partners_l23

                        if src_ei == "E":
                            # Местных возбуждающих входов клетки — для доли
                            # входов из других зон (_cortical_path).
                            self.local_e_inputs[post.name] = self.local_e_inputs.get(post.name, 0.0) + k_data
                        k = min(round(p * pre.size), pre.size)
                        # Синапс LAMP5 медленный (lamp5_tau_syn): тот же пик
                        # PSP — меньший вес, но заряд больше.
                        slow = cfg.lamp5_tau_syn if src_kind == "LAMP5" else None
                        weight = gain * psp / self._psp_peak(post, slow)
                        rate = rates[src_layer][src_kind] / 1000.0
                        tau = float(self.tau_m[post.start])

                        # Сила синапсов соседей — живая (кратковременная
                        # пластичность их популяции, _step), не устоявшаяся
                        # при частоте покоя.
                        weight_dc = weight

                        if k < 1 or src_kind == "LAMP5":
                            # Связь целиком — средним током (у LAMP5 — всегда:
                            # объёмное действие облака GABA, не синапсы).
                            self._missing(post, pre, k_data * rate * weight_dc * tau, rate, slow=src_kind == "LAMP5")
                            continue

                        root = math.sqrt(max(1.0, k_data / k))
                        self._missing(post, pre, k_data * rate * weight_dc * tau * (1.0 - 1.0 / root), rate)
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
                            # Обучаемая связь зоны: правила — AreaSpec; веса в
                            # анатомическом мозге — по данным (средний PSP пары,
                            # потолок learned_cap_psp средних), иначе — AreaSpec.
                            ee_weight, ee_cap = spec.ee_weight, spec.ee_cap

                            if getattr(cfg, "anatomy", False) and getattr(cfg, "learned_ee_data", True):
                                w = weight * math.sqrt(max(1.0, k_data / k))
                                ee_weight = (0.0, 2.0 * w) if spec.ee_plastic else (w, w)
                                ee_cap = cfg.learned_cap_psp * w

                            proj = self._connect(
                                name, pre, post, k, ee_weight, plastic=spec.ee_plastic,
                                cap=ee_cap, modulator=spec.ee_modulator,
                                ach_suppressed=spec.ee_ach_suppressed, pre_index=pre_index,
                            )
                            if proj is not None:
                                proj.scaled = getattr(spec, "ee_scaled", True)
                                proj.data_weight = bool(getattr(cfg, "anatomy", False)) and getattr(cfg, "learned_ee_data", True)

                            if spec.columns > 1:
                                self._wire_columns(spec, area)
                            continue

                        proj = self._connect(name, pre, post, k, (weight * root, weight * root), pre_index=pre_index)

                        if proj is not None and spec.name in getattr(cfg, "vogels_areas", ()) and (src_kind, dst_kind) == ("PV", "E"):
                            # Пластичность торможения PV → E (Vogels et al. 2011;
                            # D'amour & Froemke 2015) — декорреляция признаков.
                            proj.vogels = True
                            proj.cap = cfg.vogels_cap

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
        # Входы недостающих соседей — живое среднее поле (_missing).
        self.mean_field: list = []

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

            if getattr(spec, "local_inhibitory", -1.0) == 0:
                # Местных тормозных нет — нет и местного контура.
                continue

            if getattr(spec, "local_psp", ()):
                # Местный контур по данным (anatomy.NUCLEI): входов (по данным)
                # и PSP; клеток в модели меньше — недостающие связи средним
                # током по частоте источника in vivo (как у путей, k_data):
                # главные — anatomy.RATES_HZ, тормозные — частота из таблицы.
                from brain import anatomy

                k_ei, ei, k_ie, ie, k_ii, ii, i_hz = spec.local_psp
                e_hz = anatomy.RATES_HZ.get(spec.name, 1.0)

                for name, pre, post, k, psp, hz in (("E->I", area.E, area.I, k_ei, ei, e_hz),
                                                     ("I->E", area.I, area.E, k_ie, ie, i_hz),
                                                     ("I->I", area.I, area.I, k_ii, ii, i_hz)):
                    if k <= 0:
                        continue

                    w = psp / self._psp_peak(post)
                    built = max(1, min(k, pre.size))
                    root = math.sqrt(k / built)
                    tau = float(self.tau_m[post.start])
                    dc = self.dc_inh_mv if w < 0 else self.dc_mv
                    self._missing(post, pre, k * hz / 1000.0 * w * tau * (1.0 - 1.0 / root), hz / 1000.0)
                    local = (spec.name == "lgn" and getattr(cfg, "eye", False)) or (
                        spec.name == "mgn" and getattr(cfg, "relay_local_inhibition", False)) or (
                        spec.name in ("cn", "ic"))
                    index = self._lgn_local(area, pre, post, built) if local else None
                    self._connect(f"{spec.name}: {name}", pre, post, built, (w * root, w * root), pre_index=index)

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

        # Входы клетки колонки, которые в модели есть явно (таламус, другие
        # колонки): их вычитают из внешнего входа (_external_minus_modeled).
        self.modeled_inputs: dict[str, float] = {}

        # ---- проводящие пути между зонами ----
        # Сколько врождённых путей между колонками входит в каждую популяцию:
        # доля входов из других зон делится между ними (_cortical_path).
        self.cortical_inputs: dict[str, int] = {}

        for path in cfg.pathways:
            if self._data_path(path):
                _, post, inhibitory, _ = self._laminar(path, self.areas[path.src], self.areas[path.dst])

                for pop in (post, inhibitory):
                    if pop is None:
                        continue
                    self.cortical_inputs[pop.name] = self.cortical_inputs.get(pop.name, 0) + 1

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
            weight = path.weight

            k = path.k
            k_inhibitory, inhibitory_weight = path.k_inhibitory, path.inhibitory_weight

            if getattr(path, "psp_mv", None) is not None:
                # Вес из измеренного PSP (анатомический мозг); обучаемый — от 0
                # до 2 средних, потолок learned_cap_psp средних.
                w = path.psp_mv / self._psp_peak(post)
                weight = (w, w)

                if getattr(path, "k_data", 0.0) > 0:
                    # Входов по данным больше, чем строим (PathwaySpec.k_data).
                    from brain import anatomy

                    k = max(1, min(path.k, pre.size))
                    root = math.sqrt(max(1.0, path.k_data / k))
                    rate = anatomy.RATES_HZ.get(path.src, 1.0) / 1000.0
                    tau = float(self.tau_m[post.start])
                    dc = self.dc_inh_mv if w < 0 else self.dc_mv
                    self._missing(post, pre, path.k_data * rate * w * tau * (1.0 - 1.0 / root), rate,
                                  ach=getattr(path, "ach_suppressed", False))
                    w = w * root
                    weight = (w, w)

                if path.plastic:
                    weight = (0.0, 2.0 * w)
                    path = replace(path, cap=cfg.learned_cap_psp * abs(w))

                if getattr(path, "k_data_inhibitory", 0.0) > 0 and inhibitory is not None:
                    # Прямое торможение: путь бьёт и по местным тормозным
                    # клеткам получателя (входов и PSP — по данным).
                    from brain import anatomy

                    wi = path.psp_inhibitory_mv / self._psp_peak(inhibitory)
                    k_inhibitory = max(1, min(round(path.k_data_inhibitory), pre.size))
                    root = math.sqrt(max(1.0, path.k_data_inhibitory / k_inhibitory))
                    rate = anatomy.RATES_HZ.get(path.src, 1.0) / 1000.0
                    tau = float(self.tau_m[inhibitory.start])
                    self._missing(inhibitory, pre, path.k_data_inhibitory * rate * wi * tau * (1.0 - 1.0 / root), rate,
                                  ach=getattr(path, "ach_suppressed", False))
                    inhibitory_weight = wi * root
                else:
                    k_inhibitory = 0

            if self._data_path(path):
                # Врождённый путь между колонками — по данным (_cortical_path).
                k, w = self._cortical_path(path, pre, post)
                weight = (w, w)
                if inhibitory is not None:
                    k_inhibitory, inhibitory_weight = self._cortical_path(path, pre, inhibitory)

                if path.plastic:
                    # Обучаемый: врождённо — от 0 до 2 средних PSP (среднее —
                    # по данным), потолок — сильнейшие связи коры ~10 средних
                    # (логнормальный хвост, Song et al. 2005).
                    weight = (0.0, 2.0 * w)
                    path = replace(path, cap=cfg.learned_cap_psp * w)
            elif self._thalamic(path, dst):
                # Таламус → колонка — по данным (_thalamic_path).
                k, w = self._thalamic_path(path, pre, post, inhibitory=False)
                weight = (w, w)
                if inhibitory is not None:
                    k_inhibitory, inhibitory_weight = self._thalamic_path(path, pre, inhibitory, inhibitory=True)

            proj = self._connect(
                getattr(path, "name", "") or f"{path.src} -> {path.dst}", pre, post, k,
                weight, plastic=path.plastic, cap=path.cap,
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
                # Вес по данным (средний PSP пути — cap / learned_cap_psp):
                # шаг STDP — в долях этой средней связи (chemical_plasticity).
                proj.data_weight = path.plastic and (self._data_path(path) or getattr(path, "psp_mv", None) is not None)
                sigma = getattr(cfg, "interareal_weight_sigma", 0.0)

                if proj.data_weight and sigma > 0:
                    # Врождённые веса обучаемого межзонного пути — логнормально
                    # (cfg.interareal_weight_sigma), среднее то же.
                    mean = 0.5 * (weight[0] + weight[1])
                    mu = math.log(mean) - 0.5 * sigma ** 2
                    z = torch.randn(proj.weight.shape, device=self.device, generator=self.generator)
                    proj.weight.copy_(exact16(torch.exp(mu + sigma * z).clamp(max=path.cap)))

            if inhibitory is None:
                continue

            self._connect(
                f"{path.src} -> {path.dst}.I", pre, inhibitory, k_inhibitory,
                (inhibitory_weight, inhibitory_weight),
                # Ацетилхолин подавляет выброс на окончаниях пути (Hasselmo
                # 1995) — и на тормозных клетках тоже.
                ach_suppressed=path.ach_suppressed,
                # Прямое торможение — из той же окрестности, что вход на E
                # (та же карта поля): без неё ветка на PV шла со всего V1, и
                # любая активность V1 гасила всю зону букв (ответ ×0.98).
                topographic=path.topographic, retinotopic=getattr(path, "retinotopic", 0.0),
            )

        # ---- органы-датчики: рецепторы -> своя зона ----
        # У каждого органа свой участок общего массива рецепторов: ухо —
        # первым (как всегда), дальше остальные.
        for sensor, offset in zip(self.sensors, self.sensor_offsets):
            area = self.areas[sensor.area]
            relay = self.areas.get(getattr(sensor, "relay", "") or "")

            if relay is not None and getattr(sensor, "brainstem", ()):
                # Слух: нерв → ствол → реле (_brainstem), не прямо в таламус.
                self._brainstem(sensor, offset, relay)
            elif relay is not None:
                # Реле таламуса (LGN, MGN): рецептор → своя клетка реле (у
                # клетки LGN 1–3 сильных входа сетчатки, Usrey, Reppas & Reid
                # 1999), тормозные реле — случайные рецепторы. Дальше реле
                # входит в L4 зоны той же проводкой, что раньше рецепторы.
                cells = torch.arange(relay.E.size, device=self.device) % sensor.size
                driver = cfg.relay_driver_psp_mv / self._psp_peak(relay.E)
                self.projections.append(Projection(
                    f"{sensor.name} -> {relay.E.name}", "sensory", (cells + offset)[:, None], relay.E,
                    torch.full((relay.E.size, 1), driver, device=self.device),
                ))
                # Местные тормозные клетки реле получают сетчатку так же (1–3
                # входа, триады; ~25% клеток таламуса приматов — Arcelli et al.
                # 1997).
                k = 2

                if getattr(cfg, "relay_local_inhibition", False):
                    # Вход тормозной клетки — рецепторы своего места на
                    # карте (рядом с её клетками реле), а не со всего органа:
                    # у MGN — своя полоса улитки ±1, у LGN — то же место
                    # сетчатки ±2 столбца (слой ON или OFF — случайно).
                    own = self._relay_place(relay, relay.I)
                    g = self.generator

                    if relay.name == "mgn":
                        per = sensor.size // cfg.cochlea_bands
                        band = (own[:, 0:1] + torch.randint(-1, 2, (relay.I.size, k), device=self.device, generator=g)).clamp(0, cfg.cochlea_bands - 1)
                        pre = band.long() * per + torch.randint(0, per, (relay.I.size, k), device=self.device, generator=g)
                    else:
                        R, C = cfg.retina_rows, cfg.retina_cols
                        col = (own[:, 1:2] + torch.randint(-2, 3, (relay.I.size, k), device=self.device, generator=g)).clamp(0, C - 1)
                        layer = torch.randint(0, 2, (relay.I.size, k), device=self.device, generator=g)
                        pre = layer * R * C + own[:, 0:1].long() * C + col.long()
                else:
                    pre = torch.randint(0, sensor.size, (relay.I.size, k), device=self.device, generator=self.generator)

                self.projections.append(Projection(
                    f"{sensor.name} -> {relay.I.name}", "sensory", pre + offset,
                    relay.I, torch.full((relay.I.size, k), driver, device=self.device),
                ))
            # Вход органа (через таламус) — в L4 колонки (Potjans: таламус → L4).
            target = area.layers.get("L4", area.layers["L23"]) if area.layers is not None else None
            excitatory = target["E"] if target else area.E
            inhibitory = target["PV"] if target else area.I

            for pop, gain in ((excitatory, 1.0), (inhibitory, sensor.inhibitory_gain)):
                if sensor.wiring == "oriented" and pop is excitatory:
                    pre, weight = self._wire_oriented(sensor, pop.size)
                elif sensor.wiring == "tonotopic":
                    # Слух: клетка L4 — своё место на оси частот (по порядку);
                    # входы — волокна (клетки реле) своей полосы ± a1_thalamic_bands.
                    k = min(sensor.k, sensor.size)
                    bands = cfg.cochlea_bands
                    per = sensor.size // bands
                    g = self.generator
                    own = ((torch.arange(pop.size, device=self.device).float() + 0.5) * bands / pop.size).long()
                    band = (own[:, None] + torch.randint(-cfg.a1_thalamic_bands, cfg.a1_thalamic_bands + 1, (pop.size, k),
                                                         device=self.device, generator=g)).clamp(0, bands - 1)
                    pre = band * per + torch.randint(0, per, (pop.size, k), device=self.device, generator=g)
                    weight = torch.full((pop.size, k), sensor.weight / self._psp_peak(pop), device=self.device)
                elif sensor.wiring == "map":
                    # По положению: клетка — своё место поля, k рецепторов
                    # (ON и OFF, все строки) из соседних столбцов.
                    k = min(sensor.k, sensor.size)
                    centre = (torch.arange(pop.size, device=self.device).float() + 0.5) * sensor.cols / pop.size
                    col = centre[:, None] + torch.rand((pop.size, k), device=self.device, generator=self.generator) * 3 - 1.5
                    col = col.long().clamp(0, sensor.cols - 1)
                    row = torch.randint(0, sensor.rows, (pop.size, k), device=self.device, generator=self.generator)
                    layer = torch.randint(0, 2, (pop.size, k), device=self.device, generator=self.generator)
                    pre = layer * sensor.rows * sensor.cols + row * sensor.cols + col
                    weight = torch.full((pop.size, k), sensor.weight / self._psp_peak(pop), device=self.device)
                elif (relay is not None and getattr(cfg, "anatomy", False) and sensor.wiring == "oriented"
                      and pop is inhibitory):
                    # Быстрые тормозные L4 V1: поле шире, чем у E, но местное
                    # (не всё поле зрения) — своё место и входы из окрестности
                    # ~вдвое шире поля простой клетки (±3 строки, ±4 столбца;
                    # поле E — ~5 × 4 px). Входы со всего поля (прежнее «широкое
                    # торможение») гасили весь L4 на букву в одном месте.
                    k = min(sensor.k, sensor.size)
                    _, r0, c0 = self._oriented_layout(sensor, pop.size)
                    g = self.generator
                    r = (r0[:, None] + (torch.rand((pop.size, k), device=self.device, generator=g) * 6 - 3)).round().long()
                    c = (c0[:, None] + (torch.rand((pop.size, k), device=self.device, generator=g) * 8 - 4)).round().long()
                    r = r.clamp(0, sensor.rows - 1)
                    c = c.clamp(0, sensor.cols - 1)
                    layer = torch.randint(0, 2, (pop.size, k), device=self.device, generator=g)
                    pre = layer * sensor.rows * sensor.cols + r * sensor.cols + c
                    weight = torch.full((pop.size, k), sensor.weight, device=self.device)
                else:
                    # Случайные рецепторы своего органа (торможение сетчатки
                    # в V1 — тоже широкое, без формы поля).
                    k = min(sensor.k, sensor.size)
                    pre = torch.randint(
                        0, sensor.size, (pop.size, k), device=self.device, generator=self.generator,
                    )
                    weight = torch.full((pop.size, k), sensor.weight, device=self.device)

                    if getattr(sensor, "psp_mv", None) is not None:
                        weight = torch.full((pop.size, k), sensor.psp_mv / self._psp_peak(pop), device=self.device)

                if relay is not None and getattr(cfg, "anatomy", False):
                    # Анатомический мозг: таламус → L4 по данным, как у
                    # остальных путей таламус → кора (_thalamic_path): на E —
                    # 85 таламических клеток по 1.0 мВ, на PV — 56 по 2.0
                    # (Bruno & Sakmann 2006; Schoonover et al. 2014;
                    # Cruikshank et al. 2007). Синапсы таламуса только
                    # возбуждают: у ориентированного поля — лишь «+» подполя
                    # (форма поля Хьюбела — Визеля), «толкай» дают PV L4 со
                    # своим входом таламуса, а не отрицательные веса (прежнее
                    # упрощение гасило слово: «+» и «−» поровну, L4 E молчали).
                    from brain import anatomy

                    if sensor.wiring == "oriented" and pop is excitatory:
                        pre = pre[:, weight[0] > 0]
                    k_data = 56.0 if pop is inhibitory else 85.0
                    psp = 2.0 if pop is inhibitory else 1.0
                    built = pre.shape[1]
                    w = psp / self._psp_peak(pop)

                    if sensor.wiring in ("oriented", "tonotopic"):
                        # Все таламические входы клетки — из её места поля (у слуха — её полосы)
                        # (ретинотопия): недостающие видят то же, что её
                        # построенные, — их несут сами построенные (вес ×
                        # k_data / k). Среднее поле от всего LGN гасило слово:
                        # зрячие клетки LGN 14 Гц, прочие 9.5 → 3.6 (через
                        # ретикулярное), вход L4 на слово падал 15 → 12 мВ.
                        scale = k_data / built
                    else:
                        root = math.sqrt(max(1.0, k_data / built))
                        rate = anatomy.RATES_HZ.get(relay.name, 10.0) / 1000.0
                        tau = float(self.tau_m[pop.start])
                        self._missing(pop, relay.E, k_data * rate * w * tau * (1.0 - 1.0 / root), rate)
                        scale = root
                    proj = Projection(
                        f"{relay.name} -> {pop.name}", "net", (pre + relay.E.start).to(torch.int32), pop,
                        torch.full(pre.shape, w * scale, device=self.device),
                    )
                    proj.pre_pop = relay.E
                    proj.source_kind = relay.kind
                    self.projections.append(proj)
                    self.modeled_inputs[pop.name] = self.modeled_inputs.get(pop.name, 0.0) + k_data
                    continue

                if relay is not None:
                    # Реле → L4 (таламокортикальный путь).
                    proj = Projection(
                        f"{relay.name} -> {pop.name}", "net", (pre + relay.E.start).to(torch.int32), pop, weight * gain,
                    )
                    proj.pre_pop = relay.E
                    proj.innate_range = (float(weight.min()), float(weight.max()))
                    proj.source_kind = relay.kind
                    self.projections.append(proj)
                    self.modeled_inputs[pop.name] = self.modeled_inputs.get(pop.name, 0.0) + pre.shape[1]
                    continue

                self.projections.append(
                    Projection(
                        f"{sensor.name} -> {pop.name}", "sensory", pre + offset, pop, weight * gain,
                    )
                )

        self._external_minus_modeled()
        self._build_mean_field()
        self._pack()
        self._init_nmda()

    def _relay_place(self, area, pop: Population) -> torch.Tensor:
        """
        Место клетки реле на карте органа: клетка реле j — место рецептора j,
        тормозная — место «своей» клетки реле (ровно по ядру). LGN — строка и
        столбец сетчатки; MGN — полоса улитки (тонотопия), одна ось.
        """

        cfg = self.cfg

        if pop is area.E:
            j = torch.arange(pop.size, device=self.device)
        else:
            j = ((torch.arange(pop.size, device=self.device).float() + 0.5) * area.E.size / pop.size).long()

        if area.name in ("mgn", "cn", "ic"):
            j = j % cfg.sensory_size
            band = (j // (cfg.sensory_size // cfg.cochlea_bands)).float()
            return torch.stack([band, torch.zeros_like(band)], 1)

        R, C = cfg.retina_rows, cfg.retina_cols
        j = j % (R * C)
        return torch.stack([(j // C).float(), (j % C).float()], 1)

    def _brainstem(self, sensor, offset: int, relay) -> None:
        """
        Слуховой ствол по тонотопии (cfg.brainstem_links): волокно нерва →
        кохлеарное ядро → нижнее двухолмие → реле MGN. Место клетки каждого
        ядра — полоса улитки (_relay_place); входы — случайные клетки
        источника из своей полосы ± spread.
        """

        cfg = self.cfg
        per = sensor.size // cfg.cochlea_bands
        g = self.generator

        def bands(area, pop):
            return self._relay_place(area, pop)[:, 0].long()

        def source(name):
            if name == "nerve":
                band = torch.arange(sensor.size, device=self.device) // per
                return band, torch.arange(sensor.size, device=self.device) + offset, "sensory", None
            area = self.areas[name]
            return bands(area, area.E), torch.arange(area.E.start, area.E.stop, device=self.device), "net", area

        for src, dst, k, spread, psp in cfg.brainstem_links:
            name, kind = dst.split(".")
            area = self.areas[name]
            post = area.E if kind == "E" else area.I
            src_band, src_index, table, src_area = source(src)
            order = torch.argsort(src_band)
            counts = torch.bincount(src_band, minlength=cfg.cochlea_bands)
            starts = torch.cumsum(counts, 0) - counts
            want = bands(area, post)[:, None] + torch.randint(-spread, spread + 1, (post.size, k), device=self.device, generator=g)
            want = want.clamp(0, cfg.cochlea_bands - 1)
            pick = starts[want] + (torch.rand((post.size, k), device=self.device, generator=g) * counts[want]).long()
            pre = src_index[order[pick]]
            weight = torch.full((post.size, k), psp / self._psp_peak(post), device=self.device)
            proj = Projection(f"{src} -> {post.name}", table, pre if table == "sensory" else pre.to(torch.int32), post, weight)

            if src_area is not None:
                proj.pre_pop = src_area.E
                proj.source_kind = src_area.kind

            self.projections.append(proj)

    def _lgn_local(self, area, pre: Population, post: Population, k: int) -> torch.Tensor:
        """
        Местный контур LGN — по месту в поле зрения: тормозные клетки реле
        локальны (дендриты и триады — у своих клеток реле; Sherman & Guillery
        2006). Клетка реле j — место рецептора j (ON и OFF, строка, столбец);
        тормозная — место «своей» клетки реле (ровно по ядру). Входы — k
        ближайших по полю (со случайным сдвигом). Случайные по всему ядру
        связи гасили на слово клетки, его не видящие (15 → 8 Гц).
        """

        a, b = self._relay_place(area, post), self._relay_place(area, pre)
        g = self.generator
        out = []

        for chunk in torch.split(torch.arange(post.size, device=self.device), 512):
            d = torch.cdist(a[chunk], b)
            d = d + torch.rand(d.shape, device=self.device, generator=g) * 1.5

            if pre is post:
                d[torch.arange(len(chunk), device=self.device), chunk] = float("inf")
            out.append(d.topk(min(k, pre.size - (pre is post)), dim=1, largest=False).indices)

        return pre.start + torch.cat(out).to(torch.int32)

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
        # Без задержек массив не нужен (0.56 ГБ на 279 млн синапсов).
        self.net_delay = None
        self.delay_slots = 1

        if getattr(self.cfg, "axon_delays", False):
            self.net_delay = torch.zeros(self.net_weight.shape, dtype=torch.int16, device=self.device)
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
        suppressed = [p for p in net if getattr(p, "ach_suppressed", False)]
        self.dendrite_syn = None
        self.ach_mask = None
        # Оба признака постоянны внутри пути: ядро доставки (brain/csrc)
        # находит путь синапса поиском по началам путей — масок на каждый
        # синапс не нужно (0.84 ГБ на 279 млн синапсов). Маски — только для
        # шага на PyTorch (процессор).
        paths = sorted((p for p in net if p.weight.numel()), key=lambda p: p.offset)
        flags = [int(p in dendritic) | (int(p in suppressed) << 1) for p in paths]
        self.path_start = torch.tensor([p.offset for p in paths] + [self.n_synapses], dtype=torch.long, device=self.device)
        self.path_flags = torch.tensor(flags + [0], dtype=torch.uint8, device=self.device)
        # LTD пропорциональна весу синапса (stdp_ltd_weight_dependent): на
        # путь — 1 / средний врождённый вес, 0 — шаг LTD от веса не зависит.
        self.path_ltd_ref = None

        if self.cfg.stdp_ltd_weight_dependent:
            refs = [1.0 / p.innate_mean if p.plastic and p.innate_mean > 0 else 0.0 for p in paths]
            self.path_ltd_ref = torch.tensor(refs + [0.0], dtype=torch.float32, device=self.device)
        # Получатель синапса — из места в матрице пути [получатель × входы]:
        # post.start + (синапс − начало пути) // входов. Хвост — фиктивные
        # синапсы (получатель — сам фиктивный нейрон, вес 0, сюда не доходят).
        self.path_post = torch.tensor([p.post.start for p in paths] + [self.n], dtype=torch.long, device=self.device)
        self.path_fanin = torch.tensor([p.pre.shape[1] for p in paths] + [1 << 40], dtype=torch.long, device=self.device)
        self.ach_paths = bool(suppressed)

        if self.device.type == "cuda" and _kernels.deliver is not None:
            # net_post на видеокарте не нужен (1.1 ГБ на 279 млн синапсов).
            self.net_post = None
            self._build_tables()
            return

        if dendritic:
            self.dendrite_syn = torch.zeros(self.net_weight.numel(), dtype=torch.bool, device=self.device)

            for p in dendritic:
                self.dendrite_syn[p.offset: p.offset + p.weight.numel()] = True

        # Пути, передачу по которым подавляет ацетилхолин: доля подавления
        # на каждом синапсе (0 — не подавляется).
        if suppressed:
            self.ach_mask = torch.zeros(self.net_weight.numel(), dtype=torch.float16, device=self.device)

            for p in suppressed:
                self.ach_mask[p.offset: p.offset + p.weight.numel()] = self.cfg.ach_suppression

        self._build_tables()

    def set_acetylcholine(self, level: float) -> None:
        """Уровень ацетилхолина (0 — сон и вспоминание, 1 — бодрая запись нового)."""

        if self.ach_chemistry >= 0:
            # Ацетилхолин даёт ядро Мейнерта — код его не ставит.
            return

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

        modulators = dict(self.MODULATORS)
        modulators.update(dict(getattr(self.cfg, "extra_modulators", ())))
        names = list(modulators)
        self.chemistry_names = names
        # float64: аденозин копится часами — прибавки за шаг ~1e-10.
        self.chemistry = torch.zeros(len(names), device=self.device, dtype=torch.float64)
        self.chemistry_decay = torch.zeros(len(names), device=self.device, dtype=torch.float64)
        # Выброс на спайк (доля клеток источника) и постоянный выброс за шаг.
        self.chemistry_release = torch.zeros(len(names), device=self.device, dtype=torch.float64)
        self.chemistry_basal = torch.zeros(len(names), device=self.device, dtype=torch.float64)
        self.chemistry_source = torch.zeros(len(names), self.n_slots, device=self.device)

        for i, name in enumerate(names):
            area = self.areas.get(modulators[name])
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
            # «зона» — её E; «зона:VIP» — все клетки этого типа во всех слоях.
            area_name, _, kind = area_name.partition(":")
            area = self.areas.get(area_name)

            if area is None or name not in names:
                continue

            if not kind:
                self.receptor_mv[names.index(name), area.E.slice] = mv
                continue

            for label, pop in area.populations:
                if label.split(".")[-1] == kind or (kind == "PV" and label.split(".")[-1] == "I"):
                    self.receptor_mv[names.index(name), pop.slice] = mv

        # Ацетилхолин из ядра Мейнерта (анатомический мозг): уровень, который
        # читают пути и sAHP, — занятость его рецепторов, а не число из кода.
        self.ach_chemistry = names.index("acetylcholine") if "acetylcholine" in names else -1

        # Часы (SCN): ток клеткам часов по фазе суток — молекулярный цикл
        # клетки, выставляется set_clock (copy_ — для CUDA Graph).
        self.clock_mv = torch.zeros(self.n_slots, device=self.device)
        self.sleep_chemistry(False)
        # Анатомический мозг: спит он или нет, решает переключатель (выброс
        # VLPO — галанин — больше норадреналина LC, Saper 2001), а не код —
        # по нему в каждом шаге выбираются вымывание аденозина и прочее.
        self.state_sleep = bool(getattr(self.cfg, "anatomy", False)) and {"galanin", "noradrenaline"} <= set(names)

        if self.state_sleep:
            f64 = lambda xs: torch.tensor(xs, device=self.device, dtype=torch.float64)
            rates = [self._chemistry_rates(s) for s in (False, True)]
            self.chemistry_by_state = [
                (torch.exp(-self.cfg.dt_ms / f64(tau)), f64(release), f64(basal)) for tau, release, basal in rates
            ]
            self.galanin_i, self.noradrenaline_i = names.index("galanin"), names.index("noradrenaline")

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
        self.in_plastic = table(self.net_post[:P] if self.net_post is not None else self._plastic_posts(), P)

        # Плотный список событий ядром C++/CUDA (brain/kernels.py): ёмкость —
        # доля синапсов таблицы за шаг (cfg.event_capacity).
        self.compact = _kernels.compact_events is not None and self.device.type == "cuda"

        if self.compact:
            fraction = self.cfg.event_capacity

            for t, total in ((self.out_all, S), (self.out_plastic, P), (self.in_plastic, P)):
                t.prepare_compact(max(32768, int(total * fraction)), self.n, self.DUMMY_NEURONS)
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
        # Медленное торможение от LAMP5 (см. decay_ngf).
        self.i_ngf = z()
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

            if area.SOM is not None or area.layers is not None:
                # До порога — как у клетки этого типа (реобаза × сопротивление).
                gap = cfg.v_threshold - cfg.v_rest

                for _, group in self._typed_groups(area):
                    for kind, data_gap in (("PV", cfg.pv_gap_mv), ("SOM", cfg.som_gap_mv), ("VIP", cfg.vip_gap_mv),
                                           ("LAMP5", cfg.lamp5_gap_mv)):
                        if kind in group:
                            self.bias[group[kind].slice] = gap - data_gap

                    if area.layers is not None:
                        self.bias[group["E"].slice] = gap - cfg.column_e_gap_mv

                    # Разброс между клетками (Allen; config column_gap_sd_mv),
                    # расстояние до порога не меньше 3 мВ.
                    for kind, sd in zip(KINDS, cfg.column_gap_sd_mv):
                        if (kind == "E" and area.layers is None) or kind not in group:
                            continue
                        pop = group[kind]
                        spread = torch.randn(pop.size, device=self.device, generator=self.generator) * sd
                        self.bias[pop.slice] = (self.bias[pop.slice] + spread).clamp(max=gap - 3.0)

        for spec in cfg.areas:
            hz = getattr(spec, "autonomous_hz", 0.0)

            if hz > 0:
                # Пейсмейкер: постоянный ток I, при котором клетка из сброса
                # доходит до порога за 1/hz − рефрактерность (LIF):
                # e^{T/τ} = (I − a) / (I − g), a — сброс, g — порог над покоем.
                E = self.areas[spec.name].E
                tau = float(self.tau_m[E.start])
                T = max(1.0, 1000.0 / hz - cfg.refractory_ms)
                q = math.exp(T / tau)
                g, a = cfg.v_threshold - cfg.v_rest, cfg.v_reset - cfg.v_rest
                # Средний фон клетка и так получает — его вычесть (частота в
                # срезе — без сети; фон — среднее число входов × вес × τm).
                background = float(self.noise_mean[E.start] / cfg.dt_ms * self.noise_gain[E.start] * tau)
                self.bias[E.slice] = (q * g - a) / (q - 1.0) - background

        # Фон ядер — согласование с частотами in vivo (cfg.nucleus_bias_mv).
        for name, mv in getattr(cfg, "nucleus_bias_mv", ()):
            if name in self.areas:
                self.bias[self.areas[name].E.slice] += mv

        # Разброс возбудимости между клетками ядра — как у главных клеток
        # колонок (Allen, column_gap_sd_mv E): без него все клетки ядра
        # одинаковы и включаются разом — ядро «всё или ничего» (узел сна:
        # проигравшая сторона молчала целиком, а в записях стреляет реже).
        if getattr(cfg, "anatomy", False):
            sd = cfg.column_gap_sd_mv[0]

            for area in self.areas.values():
                if area.layers is None:
                    E = area.E
                    self.bias[E.slice] += torch.randn(E.size, device=self.device, generator=self.generator) * sd

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
        self.stp_u4 = torch.zeros(self.n_slots, len(KINDS), device=self.device)
        self.stp_x4 = torch.ones(self.n_slots, len(KINDS), device=self.device)
        # Десенситизация рецепторов ГАМК-А от LAMP5 (нейроглиаформных):
        # второе, медленное состояние их синапсов (Karayannis et al. 2010).
        self.is_lamp5 = self.cell_type == KINDS.index("LAMP5")
        # Старт — устоявшаяся сила при частоте покоя LAMP5 (1/(1 + U·f·τ)):
        # восстановление идёт минуты, мозг не «рождается» с отдохнувшими
        # синапсами — иначе любой замер покоя был бы переходным процессом.
        lamp5_hz = cfg.column_target_hz[0][1 + KINDS.index("LAMP5")] if cfg.column_target_hz else 0.0
        steady = 1.0 / (1.0 + cfg.lamp5_desens_u * lamp5_hz * cfg.lamp5_desens_tau_s)
        self.ngf_d = torch.where(self.is_lamp5, steady, 1.0).to(torch.float32)
        self.ngf_rec = cfg.dt_ms / (cfg.lamp5_desens_tau_s * 1000.0)
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
        # Вероятность спонтанного спайка рецептора за шаг (органы с
        # spontaneous_hz); None — рецепторы молчат без сигнала.
        spontaneous = torch.zeros(self.n_receptors, device=self.device)

        for sensor, offset in zip(self.sensors, self.sensor_offsets):
            # Частота — одна на орган или своя у каждого рецептора (ухо).
            hz = torch.as_tensor(getattr(sensor, "spontaneous_hz", 0.0), dtype=torch.float32, device=self.device)
            spontaneous[offset: offset + sensor.size] = hz * cfg.dt_ms / 1000.0

        self.spontaneous = spontaneous if bool((spontaneous > 0).any()) else None

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

        if self._fused():
            # Доставка одним ядром (brain/csrc/events.cu) — тот же счёт, что
            # ниже, только по настоящим событиям шага.
            # По строкам таблицы выстреливших — без буфера событий (ничего не
            # теряется); список выстреливших — без предела.
            fired_all = self._fired_all()
            t = self.out_all
            self.i_dend.mul_(self.decay_dend)
            inhibition = torch.zeros_like(drive)
            ngf = torch.zeros_like(drive)
            glutamate = torch.zeros_like(drive) if self.nmda_on else None
            _kernels.deliver(
                fired_all, self.n, t.start, t.count, t.perm, self.path_post, self.path_fanin, self.net_weight,
                self.path_start, self.path_flags, self.ach_paths, self.cfg.ach_suppression, self.acetylcholine,
                self.dendrite_pre, self.dendrite_post,
                self.net_branch, max(1, cfg.dendrite_branches),
                self.typed_pre, self.cell_type, len(KINDS),
                self.stp_u4, self.stp_x4, self.stp_U4,
                self.is_lamp5, self.ngf_d, 1.0 / (1.0 - cfg.lamp5_desens_u), KINDS.index("LAMP5"),
                self.stp_x, self.inh_gain,
                self.i_dend, drive, glutamate, inhibition, ngf,
            )
        else:
            synapses, pre_of = self._events(self.out_all, fired_before)
            post = self._posts().index_select(0, synapses)
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
                slot = pre_of.long() * len(KINDS) + self.cell_type.index_select(0, post).clamp(min=0)
                u = self.stp_u4.view(-1).index_select(0, slot)
                x = self.stp_x4.view(-1).index_select(0, slot)
                gain = u * x / (1.0 - u).clamp(min=1e-3) / self.stp_U4.view(-1).index_select(0, slot)
                # У LAMP5 — ещё десенситизация рецепторов (ngf_d, до спайка).
                d = self.ngf_d.index_select(0, pre_of) / (1.0 - cfg.lamp5_desens_u)
                gain = torch.where(self.is_lamp5.index_select(0, pre_of), gain * d, gain)
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
                inh = weight.clamp(max=0.0)
                # От LAMP5 — в медленный ток (lamp5_tau_syn), остальное — в быстрый.
                slow = self.cell_type.index_select(0, pre_of) == KINDS.index("LAMP5")
                inhibition = torch.zeros_like(drive).index_add_(0, post, torch.where(slow, 0.0, inh))
                ngf = torch.zeros_like(drive).index_add_(0, post, torch.where(slow, inh, 0.0))
                weight = weight.clamp(min=0.0)

                if self.nmda_on:
                    # NMDA — от тех же возбуждающих синапсов (задержка пути для
                    # медленного тока не важна).
                    glutamate = torch.zeros_like(drive).index_add_(0, post, weight)

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

        if self.spontaneous is not None:
            # Рецепторы стреляют и в тишине (SensorSpec.spontaneous_hz).
            heard_now = (heard_now + (torch.rand_like(heard_now) < self.spontaneous).float()).clamp(max=1.0)

        heard = heard_now.index_select(0, self.ear_pre)
        drive.index_add_(0, self.ear_post, heard * self.ear_weight)

        if plastic:
            # LTD: пре пришёл после поста. Нейроны рабочей памяти
            # (ковариационное правило) этой ветки не имеют.
            if self._fused():
                t = self.out_plastic
                _kernels.ltd(fired_all, self.n, t.start, t.count, t.perm, self.path_start, self.path_post, self.path_fanin, self.trace_post, self.ltd_mask,
                             self.symmetric_area, cfg.stdp_a_plus, cfg.stdp_a_minus,
                             self.net_weight, self.path_ltd_ref, self.eligibility_scale, self.eligibility)
            else:
                synapses, pre_of = self._events(self.out_plastic, fired_before)
                post = self._posts().index_select(0, synapses)
                ltd = self.trace_post.index_select(0, post) * self.ltd_mask.index_select(0, pre_of)
                sign = -cfg.stdp_a_minus

                if self.symmetric_area is not None:
                    # Симметричное окно: пре после поста — тоже потенциация.
                    zone = self.symmetric_area.index_select(0, post)
                    same = (zone > 0) & (zone == self.symmetric_area.index_select(0, pre_of))
                    sign = torch.where(same, cfg.stdp_a_plus, -cfg.stdp_a_minus)

                if self.path_ltd_ref is not None:
                    # LTD пропорциональна весу (как в ядре ltd).
                    path = torch.searchsorted(self.path_start, synapses, right=True) - 1
                    ref = self.path_ltd_ref.index_select(0, path)
                    depress = (ref > 0) & (torch.as_tensor(sign, device=self.device) < 0)
                    sign = torch.where(depress, sign * self.net_weight.index_select(0, synapses) * ref, sign)

                self.eligibility.index_add_(
                    0,
                    synapses,
                    (ltd * (sign / self.eligibility_scale)).to(self.eligibility.dtype),
                )

        self.i_syn.mul_(self.decay_syn).add_(drive * self.syn_scale)

        if self.conductance:
            self.i_inh.mul_(self.decay_syn).add_(inhibition * self.syn_scale)
            self.i_ngf.mul_(self.decay_ngf).add_(ngf * self.ngf_scale)

            if self.nmda_on:
                self.i_nmda.mul_(self.decay_nmda).add_(glutamate * self.nmda_scale)
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
            # Возбуждение — тоже проводимость: ток ∝ (E_AMPA − v), нормирован
            # к −70 мВ, где мерили EPSP. Сильный фон открывает каналы и сам
            # укорачивает мембрану — высокопроводящее состояние коры in vivo
            # (Destexhe & Paré 1999; Destexhe, Rudolph & Paré 2003).
            excite = ((cfg.ampa_reversal_mv - self.v) / (cfg.ampa_reversal_mv - cfg.epsp_holding_mv)).clamp(min=0.0)
            excite = torch.where(self.conductance_cells, excite, 1.0)
            dc_e, dc_i = self.dc_mv, self.dc_inh_mv

            if getattr(self, "mf_on", False):
                # Недостающие соседи стреляют, как их построенная популяция
                # (частота сглажена ~10 мс): ток — матрица популяций × частоты.
                self.mf_counts.zero_().index_add_(0, self.mf_pre_id, self.spikes)
                inst = self.mf_counts / self.mf_size / cfg.dt_ms
                self.mf_rate.mul_(self.mf_decay).add_(inst * (1.0 - self.mf_decay))
                exc = self.mf_exc - self.acetylcholine * self.mf_ach
                inh = self.mf_inh

                if self.has_som:
                    # Кратковременная пластичность и у недостающих соседей:
                    # их синапсы устают так же, как у построенных клеток той
                    # же популяции (средняя сила к типу получателя). Без
                    # этого L2/3 на 330 Гц кормила себя без депрессии и
                    # держалась «горящей» при PV на 400 Гц.
                    # Сила следующего спайка: u⁺ = u + U(1 − u), выброс u⁺·x,
                    # к силе покоя U.
                    nxt = self.stp_u4 + self.stp_U4 * (1.0 - self.stp_u4)
                    gain = nxt * self.stp_x4 / self.stp_U4
                    gain = torch.where(self.is_lamp5[:, None], gain * self.ngf_d[:, None], gain)
                    gain = torch.where(self.typed_pre[:, None], gain, 1.0)
                    n = len(KINDS)
                    self.mf_gain[:, :n].zero_().index_add_(0, self.mf_pre_id, gain)
                    self.mf_gain[:, :n].div_(self.mf_size[:, None])
                    g = self.mf_gain.index_select(1, self.mf_post_type).t()
                    exc, inh = exc * g, inh * g

                dc_e = dc_e + (exc @ self.mf_rate).index_select(0, self.mf_post_id)
                dc_i = dc_i + (inh @ self.mf_rate).index_select(0, self.mf_post_id)

                if self.mf_slow_on:
                    # Облако GABA от LAMP5: средний ток через фильтр τ
                    # медленного синапса (тот же заряд, медленный ход).
                    slow = ((self.mf_slow * g) @ self.mf_rate).index_select(0, self.mf_post_id)
                    self.ngf_mf.mul_(self.decay_ngf).add_(slow * (1.0 - self.decay_ngf))
                    dc_i = dc_i + self.ngf_mf

            if self.nmda_on:
                # Недостающие соседи — тот же NMDA через фильтр τ_NMDA; блок
                # магнием (Jahr & Stevens 1990) — по текущему потенциалу.
                self.nmda_mf.mul_(self.decay_nmda).add_(dc_e * (1.0 - self.decay_nmda))
                block = 1.0 / (1.0 + cfg.mg_mm / 3.57 * torch.exp(-0.062 * self.v))
                synaptic = synaptic + (self.i_nmda + self.nmda_r * self.nmda_mf) * block

            synaptic = (synaptic + dc_e) * excite + (self.i_inh + self.i_ngf + dc_i) * force

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
            + synaptic + self.bias + self.interoception + modulated + self.clock_mv
            + self.kick_current * (self.kick_left > 0)
        ) * self.decay_m
        v = torch.where(self.refractory <= 0, self.v + dv, self.v)
        self.refractory.sub_(cfg.dt_ms)

        fired = v >= self.v_peak
        self.spikes.copy_(fired)
        # Выброс веществ ядрами: доля выстреливших клеток каждого ядра.
        released = (self.chemistry_source @ self.spikes).to(torch.float64)

        if self.state_sleep:
            (dw, rw, bw), (ds, rs, bs) = self.chemistry_by_state
            asleep = self.chemistry[self.galanin_i] > self.chemistry[self.noradrenaline_i]
            self.chemistry.mul_(torch.where(asleep, ds, dw)).add_(released * torch.where(asleep, rs, rw) + torch.where(asleep, bs, bw))
        else:
            self.chemistry.mul_(self.chemistry_decay).add_(released * self.chemistry_release + self.chemistry_basal)

        if self.ach_chemistry >= 0:
            self.acetylcholine.copy_(occupancy[self.ach_chemistry : self.ach_chemistry + 1])

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
            # Десенситизация от LAMP5: спайк снимает долю U_d, восстановление
            # — за минуты (lamp5_desens_*).
            ngf_spiked = self.spikes * self.is_lamp5
            self.ngf_d.add_((1.0 - self.ngf_d) * self.ngf_rec)
            self.ngf_d.sub_(cfg.lamp5_desens_u * self.ngf_d * ngf_spiked)

        if cfg.pool_inhibition_mv > 0:
            self.pool.mul_(self.decay_pool).add_((self.spikes * self.pool_mask).sum())

        for i, (cells, _mv) in enumerate(self.area_pools):
            self.area_pool[i] = self.area_pool[i] * self.decay_pool + self.spikes[cells].sum()
        self.counts.add_(self.spikes)
        self.kick_left.sub_(1.0).clamp_(min=0.0)

        if self.track_primacy:
            self.primacy_counts.add_(self.spikes * self.primacy_gain)
            self.primacy_gain.mul_(self.primacy_decay)
        if not self._fused():
            # Путь PyTorch: выстреливших в списке шага не больше max_spikes.
            self.dropped.add_((self.spikes.sum() - self.max_spikes).clamp(min=0.0))

        # ---- STDP -> eligibility ----
        if plastic:
            # LTP: пост выстрелил после пре.
            fired_now = self._fired()
            if self._fused():
                t = self.in_plastic
                _kernels.ltp(self._fired_all(), self.n, t.start, t.count, t.perm, self.net_pre, self.ltp_kernel, cfg.stdp_a_plus,
                             self.eligibility_scale, self.eligibility)
            else:
                synapses, _ = self._events(self.in_plastic, fired_now)
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

    def _posts(self) -> torch.Tensor:
        """Получатель каждого синапса — для шага на PyTorch (на видеокарте его считает ядро)."""

        if self.net_post is None:
            lengths = self.path_start[1:] - self.path_start[:-1]
            local = torch.arange(self.n_synapses, device=self.device) - torch.repeat_interleave(self.path_start[:-1], lengths)
            path = torch.repeat_interleave(torch.arange(len(lengths), device=self.device), lengths)
            post = self.path_post[path] + local // self.path_fanin[path]
            ids = self.n + torch.arange(self.DUMMY_SYNAPSES, device=self.device) % self.DUMMY_NEURONS
            self.net_post = torch.cat([post, ids]).to(torch.int32)

        return self.net_post

    def _plastic_posts(self) -> torch.Tensor:
        """Получатели пластичных синапсов (они первые в общем массиве) — для таблицы входов."""

        return torch.cat([
            torch.arange(p.post.start, p.post.stop, device=self.device, dtype=torch.int32)[:, None]
            .expand_as(p.pre).flatten()
            for p in self.plastic
        ] + [torch.zeros(0, dtype=torch.int32, device=self.device)])

    def _fused(self) -> bool:
        """Доставка и STDP своими ядрами: есть ядра и такой состав шага, какой они считают."""

        return (
            self.compact and _kernels.deliver is not None and self.delay_slots <= 1
            and self.cfg.dendritic_context and self.has_som and self.cfg.base_circuit and self.conductance
            and self.eligibility.dtype == torch.float16 and self.net_pre.dtype == torch.int32
            and self.cell_type.dtype == torch.long and self.net_branch.dtype == torch.int8
            and self.acetylcholine.dtype == torch.float32
        )

    def _events(self, table: EventTable, neurons: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """События шага: плотным списком (ядро C++/CUDA) или строками таблицы."""

        if self.compact:
            return table.compact(neurons, self.dropped)

        return table.events(neurons)

    def _fired_all(self) -> torch.Tensor:
        """Индексы всех выстреливших нейронов шага, дополненные n (без предела)."""

        return torch.nonzero_static(self.spikes, size=self.n, fill_value=self.n).squeeze(1)

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

        if getattr(self.cfg, "anatomy", False):
            self._plastic_ms = getattr(self, "_plastic_ms", 0.0) + chunks * self.CHUNK * self.cfg.dt_ms

            if self._plastic_ms >= self.cfg.plasticity_period_ms:
                self.chemical_plasticity(self._plastic_ms)
                self._plastic_ms = 0.0

        self._check_overflow()

    def _check_overflow(self) -> None:
        """
        События, не вместившиеся в таблицу шага (dropped), — не молча:
        cfg.event_overflow "error" останавливает прогон, "warn" — печатает
        (диагностика переполнения). Счётчик обнуляют стенды — тогда
        отсчёт заново.
        """

        lost = float(self.dropped)
        seen = getattr(self, "_dropped_seen", 0.0)

        if lost < seen:
            seen = 0.0

        if lost > seen:
            self._dropped_seen = lost
            message = (f"переполнение таблицы событий: потеряно {lost - seen:.0f} синаптических событий "
                       f"(ёмкость {getattr(self.out_all, 'capacity', 0)}; cfg.event_capacity {self.cfg.event_capacity})")

            if getattr(self.cfg, "event_overflow", "error") == "error":
                raise RuntimeError(message + " — cfg.event_overflow='warn' для диагностики")

            print("ВНИМАНИЕ:", message, flush=True)

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

    def chemical_plasticity(self, elapsed_ms: float) -> None:
        """
        Анатомический мозг: веса меняются сами, без вызовов кода. Третий
        фактор — концентрация вещества у синапса (её дают нейроны ядер):
        - кора (modulator "hebbian"): след STDP сразу в вес (Markram et al.
          1997; Bi & Poo 1998);
        - гиппокамп ("acetylcholine"): след × занятость рецепторов
          ацетилхолина — наяву пишет с одного раза, во сне нет (Hasselmo 2006);
        - стриатум ("dopamine", "dopamine_d2"): dw/dt = след × (дофамин −
          тонический) / τ следа (Izhikevich 2007): залп выше фона VTA
          усиливает D1 и ослабляет D2, провал — наоборот (Reynolds & Wickens
          2002; Shen et al. 2008).
        Шаг — в долях среднего PSP пути (A+ = 0.01 — ~1% связи на пару
        спайков, Bi & Poo 1998), у путей не по данным — как раньше.
        """

        if not self.plasticity or not self.plastic:
            return

        from brain import anatomy

        cfg = self.cfg
        names = self.chemistry_names
        ach = self.occupancy("acetylcholine") if "acetylcholine" in names else 0.0
        # Тонический дофамин: VTA наяву × время выведения, в единицах уровня.
        tonic = anatomy.RATES_HZ["vta"] * cfg.dopamine_tau_ms / 1000.0
        tonic = tonic / (tonic + cfg.dopamine_half)
        phasic = (self.occupancy("dopamine") - tonic) / (1.0 - tonic)
        integrate = elapsed_ms / cfg.eligibility_tau

        # Спайки за период (для лазающего волокна): разница счётчиков.
        # observe() обнуляет счётчики — тогда разница отрицательна, берём сами счётчики.
        counts = self.counts[: self.n].clone()
        spikes = None

        if hasattr(self, "_plastic_counts"):
            spikes = counts - self._plastic_counts
            spikes = torch.where(spikes < 0, counts, spikes)

        self._plastic_counts = counts
        fused = self.device.type == "cuda" and _kernels.plastic_update is not None

        if fused:
            # Все пути без своего потолка — одним ядром (тот же счёт, что ниже).
            steps = []

            for proj in self.plastic:
                factor = {
                    "hebbian": 1.0,
                    "acetylcholine": ach,
                    "dopamine": phasic * integrate,
                    "dopamine_d2": -phasic * integrate,
                }.get(proj.modulator, 0.0)
                simple = proj.modulator != "climbing" and getattr(proj, "ceiling", None) is None
                unit = proj.cap / cfg.learned_cap_psp if getattr(proj, "data_weight", False) else 1.0
                steps.append(float(torch.tensor(unit * factor, dtype=torch.float32)) if simple else 0.0)

            if not hasattr(self, "_plastic_paths"):
                self._plastic_paths = (
                    torch.tensor([p.offset for p in self.plastic] + [self.n_plastic], dtype=torch.long, device=self.device),
                    torch.tensor([float(p.cap) for p in self.plastic] + [0.0], dtype=torch.float32, device=self.device),
                    torch.tensor([int(p.modulator in ("hebbian", "acetylcholine")) for p in self.plastic] + [0],
                                 dtype=torch.uint8, device=self.device),
                )

            start, caps, spend = self._plastic_paths
            step = torch.tensor(steps + [0.0], dtype=torch.float32).to(self.device, non_blocking=True)
            _kernels.plastic_update(self.net_weight, self.eligibility, self.early, start, step, caps, spend, self.n_plastic)

        for proj in self.plastic:
            if fused and proj.modulator != "climbing" and getattr(proj, "ceiling", None) is None:
                continue

            if proj.modulator == "climbing":
                if spikes is not None:
                    self._climbing(proj, spikes)

                continue

            factor, spend = {
                "hebbian": (1.0, True),
                "acetylcholine": (ach, True),
                "dopamine": (phasic * integrate, False),
                "dopamine_d2": (-phasic * integrate, False),
            }.get(proj.modulator, (0.0, False))

            if factor == 0.0:
                continue

            unit = proj.cap / cfg.learned_cap_psp if getattr(proj, "data_weight", False) else 1.0
            before = proj.weight.clone()
            proj.weight.add_(proj.eligibility.float() * (unit * factor))
            ceiling = getattr(proj, "ceiling", None)

            if ceiling is None:
                proj.weight.clamp_(0.0, proj.cap)
            else:
                proj.weight.clamp_(min=0.0)
                torch.minimum(proj.weight, ceiling, out=proj.weight)

            proj.early.add_((proj.weight - before).to(proj.early.dtype))

            if spend:
                proj.eligibility.zero_()

        if cfg.heterosynaptic_conservation:
            self._conserve()

    def _conserve(self) -> None:
        """
        Гетеросинаптическое сохранение суммы входа (Royer & Paré 2003,
        вставочные клетки миндалины; обзор для коры и гиппокампа —
        Chistiakova et al. 2014): усиление части входов клетки ослабляет
        остальные её входы, ослабление — усиливает, общий вес почти не
        меняется. У каждой строки пути коры (modulator "hebbian") сумма
        возвращается к врождённой сумме этой клетки: фон не растит путь
        целиком, выученное — перераспределение веса. (Прежде цель была общей
        для пути — входов × средний врождённый вес; суммы случайных строк от
        неё отличались на ~2.5%, и первый вызов перекраивал все строки —
        скачок весов, похожий на учёбу, при выключенном STDP.)
        """

        paths = [p for p in self.plastic if p.modulator == "hebbian" and p.weight.numel()]

        if not paths:
            return

        if self.device.type == "cuda" and _kernels.conserve_rows is not None:
            if not hasattr(self, "_conserve_rows"):
                start, length, target, cap = [], [], [], []

                for p in paths:
                    rows, fanin = p.weight.shape
                    start.append(p.offset + torch.arange(rows, dtype=torch.long) * fanin)
                    length.append(torch.full((rows,), fanin, dtype=torch.int32))
                    target.append(p.birth.float().sum(1).cpu())
                    cap.append(torch.full((rows,), float(p.cap), dtype=torch.float32))

                self._conserve_rows = tuple(torch.cat(t).to(self.device) for t in (start, length, target, cap))

            _kernels.conserve_rows(self.net_weight, self.early, *self._conserve_rows)
            return

        for p in paths:
            before = p.weight.clone()
            total = p.weight.sum(1, keepdim=True).clamp(min=1e-9)
            p.weight.mul_(p.birth.float().sum(1, keepdim=True) / total).clamp_(max=p.cap)
            p.early.add_((p.weight - before).to(p.early.dtype))

    def _climbing(self, proj, spikes: torch.Tensor) -> None:
        """
        Параллельные волокна → Пуркинье (Ito 1989; Coesmans et al. 2004):
        спайк волокна вместе с лазающим волокном своей клетки Пуркинье за
        тот же период (окно ~50–200 мс, Wang, Denk & Häusser 2000) — LTD,
        без него — слабая LTP. Равновесие при фоновых частотах: лазающее
        волокно ~1 Гц (олива), то есть в окне период/1000 с — LTP в столько
        раз слабее LTD [порядок, сверить]. Шаг — A+ средней связи пути.
        """

        climbing = next((p for p in self.projections if p.post is proj.post and p.name.startswith("io ->")), None)

        if climbing is None:
            return

        cf = spikes.index_select(0, climbing.pre[:, 0].long())[:, None]
        pf = spikes.index_select(0, proj.pre.flatten().long()).view(proj.pre.shape)
        unit = proj.cap / self.cfg.learned_cap_psp
        chance = self.cfg.plasticity_period_ms / 1000.0
        delta = self.cfg.stdp_a_plus * unit * pf * (chance - (cf > 0).float())
        before = proj.weight.clone()
        proj.weight.add_(delta).clamp_(0.0, proj.cap)
        proj.early.add_((proj.weight - before).to(proj.early.dtype))

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
        "v", "i_syn", "i_inh", "i_ngf", "ngf_mf", "i_nmda", "nmda_mf", "i_slow", "i_dend", "pool", "area_pool", "adapt", "adapt_slow", "refractory", "spikes",
        "trace_pre", "trace_pre_rise", "ltp_kernel", "trace_post", "counts", "eligibility",
        "eligibility_scale", "dropped", "bias", "interoception", "kick_current", "kick_left", "stp_x", "stp_u4", "stp_x4", "inh_gain",
        "chemistry", "pending", "pending_dend", "tick", "mf_rate",
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
        self.i_inh[cells] = 0.0
        self.i_ngf[cells] = 0.0
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
                self.net_pre, self.net_weight,
                self.ear_pre, self.ear_post, self.ear_weight,
                self.eligibility, self.early, self.birth_weight,
            )
        ) + sum(t.nbytes() for t in (self.out_all, self.out_plastic, self.in_plastic)) + (
            self.proc_rows.nbytes() if self.proc_rows is not None else 0
        )
