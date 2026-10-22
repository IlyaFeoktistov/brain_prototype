from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import gc
import math
import time

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
    Зона мозга: колонка коры по слоям (layers) или ядро — главные клетки E и
    местные тормозные I. У колонки E, I, SOM, VIP, LAMP5 — популяции L2/3
    (I — PV).
    """

    name: str
    title: str
    E: Population
    I: Population
    SOM: Population | None = None
    VIP: Population | None = None
    # Колонка по слоям: слой -> {"E", "PV", "SOM", "VIP", "LAMP5"}.
    # E, I, SOM, VIP, LAMP5 — популяции слоя 2/3.
    layers: dict | None = None
    LAMP5: Population | None = None

    @property
    def output(self) -> Population:
        """
        Выход зоны вниз (к мышцам, стволу, ядрам) — пирамидные L5 колонки
        (кортикоспинальные клетки — L5, Lemon 2008); без слоёв — E.
        """

        return self.layers["L5"]["E"] if self.layers and "L5" in self.layers else self.E

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
    ):
        self.name = name
        self.source = source
        # Номера входов — int32 (клеток меньше 2³¹), как в общем массиве
        # синапсов (_pack): int64 до упаковки — лишние ~1.3 ГБ на мозг 388 тыс.
        self.pre = pre.to(torch.int32) if pre.dtype == torch.int64 else pre
        self.post = post
        self.weight = weight
        self.plastic = plastic
        self.cap = cap
        self.innate_mean = float(weight.mean()) if plastic else 0.0
        # Вес по данным: шаг STDP — в долях средней связи пути (cap /
        # learned_cap_psp; chemical_plasticity).
        self.data_weight = False
        # На ветки дендрита (обратные пути коры, SOM → E).
        self.dendrite = False
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
    # Синапсов в куске устойчивой сортировки (_stable_order).
    CHUNK = 1 << 23

    def __init__(self, keys: torch.Tensor, n_slots: int, pad: int, dummies: int):
        device = keys.device
        counts = torch.bincount(keys, minlength=n_slots)
        ptr = torch.zeros(n_slots + 1, dtype=torch.int32, device=device)
        ptr[1:] = torch.cumsum(counts, 0).to(torch.int32)
        perm = self._stable_order(keys, ptr, n_slots)
        # Хотя бы одна ячейка: на пустую таблицу тоже смотрят индексы.
        perm = perm if len(perm) else torch.full((1,), pad, dtype=torch.int32, device=device)
        self.ptr = ptr
        self._setup(ptr[:-1], counts.to(torch.int32), perm, pad, dummies)

    @classmethod
    def _stable_order(cls, keys: torch.Tensor, ptr: torch.Tensor, n_slots: int) -> torch.Tensor:
        """
        То же, что argsort(keys, stable=True), но кусками: синапсы куска
        встают в строки своих нейронов вслед за синапсами прежних кусков.
        [замер] argsort разом по 327 млн ключей (мозг 388 тыс.) — пик ~10 ГБ
        видеопамяти при карте 6 ГБ: индексы int64 и буферы сортировки
        уходили в ОЗУ и своп. Кусками по 8 млн — ~0.5 ГБ сверх мозга (к этому
        моменту он уже занимает почти всю карту), порядок тот же.
        """

        device = keys.device
        perm = torch.empty(len(keys), dtype=torch.int32, device=device)
        fill = ptr[:-1].long().clone()

        for c0 in range(0, len(keys), cls.CHUNK):
            k = keys[c0: c0 + cls.CHUNK].long()
            order = torch.argsort(k, stable=True)
            sk = k[order]
            del k
            cnt = torch.bincount(sk, minlength=n_slots)
            # Место синапса в строке: сколько синапсов того же нейрона в
            # куске перед ним (порядок в куске устойчивый).
            first = torch.cumsum(cnt, 0) - cnt
            rank = torch.arange(len(sk), device=device) - first[sk]
            perm[fill[sk] + rank] = (order + c0).to(torch.int32)
            fill += cnt

        return perm

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
    Анатомический мозг (brain/anatomy.py): колонки коры по слоям, ядра и
    пути между ними по данным. Знания хранятся только в весах пластичных
    проекций; веса меняются сами (chemical_plasticity).
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
        self.plasticity = True
        # Наблюдатель (веб-интерфейс): on_chunk(выстрелившие, время мозга) —
        # не чаще раза в watch_seconds реального времени, одним снимком «кто
        # стрелял с прошлого показа» (накопитель seen считается в самом шаге):
        # показ отстаёт, но сеть не ждёт его на каждой пачке шагов.
        self.on_chunk = None
        self.watch_seconds = 0.05
        self._watched_at = 0.0

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
        from brain import anatomy

        cfg = self.cfg
        # Клеток зоны — по размеру мозга (anatomy.area_sizes): ядра по данным
        # и долям, остальное — колонкам коры.
        sizes = anatomy.area_sizes(cfg)
        self.areas: dict[str, Area] = {}
        start = 0

        for spec in cfg.areas:
            n = sizes[spec.name]

            if self._has_layers(spec.name):
                area, start = self._column(spec, n, start)
                self.areas[spec.name] = area
                continue

            if spec.cells > 0:
                # Ядро с заданным числом главных клеток (реле — по рецепторам).
                n_e, n_i = anatomy.nucleus_cells(spec)
            else:
                # Своя доля местных тормозных (0 — только 2 немых места: у ядра
                # их нет, но зона всегда с E и I).
                n_i = max(2, int(n * spec.local_inhibitory))
                n_e = n - n_i

            E = Population(f"{spec.name}.E", start, n_e)

            I = Population(f"{spec.name}.I", E.stop, n_i)
            start = I.stop

            self.areas[spec.name] = Area(spec.name, spec.title, E, I)

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

    def _has_layers(self, name: str) -> bool:
        """Зона новой коры — колонка по слоям (cfg.layered_areas)."""

        return name in self.cfg.layered_areas

    def _column_layers(self, name: str) -> list[tuple[str, int, int]]:
        """Слои колонки зоны (Potjans): у агранулярной — без L4."""

        cfg = self.cfg
        cells = list(cfg.column_cells)

        if name in cfg.agranular_areas:
            cells = [c for c in cells if c[0] != "L4"]
        elif name in cfg.thin_l4_areas:
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
        area = Area(spec.name, spec.title, top["E"], top["PV"], top.get("SOM"), top.get("VIP"), layers, top.get("LAMP5"))
        return area, start

    def _layer_shares(self) -> dict[str, dict[str, float]]:
        """Доли тормозных типов слоя среди построенных (cfg.column_kinds), в сумме 1."""

        kinds = self.cfg.column_kinds
        out = {}

        for layer, *v in self.cfg.layer_interneurons:
            share = {kind: s for kind, s in zip(INHIBITORY_KINDS, v) if kind in kinds}
            # Полная колонка — доли как в данных (сумма ≈ 1 с округлением).
            total = sum(share.values()) if len(share) < len(INHIBITORY_KINDS) else 1.0
            out[layer] = {kind: s / total for kind, s in share.items()}

        return out

    def _fill(self, e: float, i: float) -> torch.Tensor:
        """Параметр клетки: главным (E всех слоёв) — e, тормозным — i; фиктивным — 0."""

        out = torch.zeros(self.n_slots, device=self.device)

        for area in self.areas.values():
            for pop in area.excitatory:
                out[pop.slice] = e

            for pop in area.inhibitory:
                out[pop.slice] = i

        return out

    def _init_interneurons(self) -> None:
        """
        Клетки колонок по типам (E, PV, SOM, VIP, LAMP5) — человек, Allen
        (config column_e_tau_m, pv_tau_m …): своя мембрана и адаптация;
        кратковременная пластичность по парам типов (column_stp). Расстояние
        до порога — в _init_state (bias).
        """

        cfg = self.cfg
        # Тип клетки колонки: номер в KINDS; −1 — клетки ядер.
        self.cell_type = torch.full((self.n_slots,), -1, dtype=torch.long, device=self.device)

        for area in self.areas.values():
            if area.layers is None:
                continue

            for group in area.layers.values():
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

                for number, kind in enumerate(KINDS):
                    if kind in group:
                        self.cell_type[group[kind].slice] = number

        self.has_som = bool((self.cell_type >= 0).any())
        self.typed_pre = self.cell_type >= 0
        # Кратковременная пластичность по парам типов (cfg.column_stp): у
        # клетки-источника — U, восстановление и облегчение к каждому типу
        # получателя (столбцы KINDS).
        n = len(KINDS)
        self.stp_U4 = torch.ones(self.n_slots, n, device=self.device)
        self.stp_rec4 = torch.zeros(self.n_slots, n, device=self.device)
        self.stp_fac4 = torch.zeros(self.n_slots, n, device=self.device)

        for src, dst, U, rec, fac in cfg.column_stp:
            rows = self.cell_type == KINDS.index(src)
            col = KINDS.index(dst)
            self.stp_U4[rows, col] = U
            self.stp_rec4[rows, col] = cfg.dt_ms / rec
            self.stp_fac4[rows, col] = math.exp(-cfg.dt_ms / fac) if fac > 0 else 0.0

    def _init_neurons(self) -> None:
        cfg = self.cfg
        # Главные клетки — AdEx, тормозные — LIF.
        self.tau_m = self._fill(cfg.e_tau_m, cfg.i_tau_m)
        self.tau_syn = self._fill(cfg.e_tau_syn, cfg.i_tau_syn)
        self.delta_t = self._fill(cfg.e_delta_t, 0.0)
        self.adapt_b = self._fill(cfg.e_adapt_b, 0.0)

        for spec in cfg.areas:
            if spec.adapt_b is not None:
                self.adapt_b[self.areas[spec.name].E.slice] = spec.adapt_b
        self._init_interneurons()
        self.v_peak = self._fill(0.0, cfg.v_threshold)
        self.v_peak[self.n:] = float("inf")
        self.tau_m[self.n:] = 1.0
        self.tau_syn[self.n:] = 1.0
        self.is_adex = self.delta_t > 0
        # Ветка LTD (пре после поста) — у всех синапсов (маску читает ядро ltd).
        self.ltd_mask = self._fill(1.0, 1.0)
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

        dt = cfg.dt_ms
        self.decay_m = dt / self.tau_m
        self.decay_syn = torch.exp(-dt / self.tau_syn)
        # Медленное торможение LAMP5 (нейроглиаформные): свой ток со своим
        # спадом; скачок τm/τ — тот же интеграл PSP, что у быстрого синапса.
        self.decay_ngf = math.exp(-dt / cfg.lamp5_tau_syn)
        self.ngf_scale = self.tau_m / cfg.lamp5_tau_syn
        self.decay_dend = math.exp(-dt / cfg.dendrite_tau_ms)
        # Сколько плато ветки поднимает сому, мВ.
        self.dendrite_soma = torch.full((self.n_slots,), float(cfg.dendrite_soma_mv), device=self.device)
        # На ветки дендрита идут пути с признаком dendrite (path_flags: обратные
        # пути коры, SOM → E). Маски по клеткам ядро доставки читает, но
        # клеточного признака нет — они пустые.
        self.dendrite_pre = torch.zeros(self.n_slots, dtype=torch.bool, device=self.device)
        self.dendrite_post = torch.zeros(self.n_slots, dtype=torch.bool, device=self.device)
        # Чьё торможение растёт с частотой (inh_gain_per_spike).
        self.inh_adapts = torch.zeros(self.n_slots, device=self.device)

        for spec in cfg.areas:
            if spec.inhibition_adapts:
                self.inh_adapts[self.areas[spec.name].E.slice] = 1.0
        # Кратковременная депрессия путей ядер (cfg.nucleus_stp): ресурс — у
        # клетки-источника; у прочих клеток ресурс не тратится.
        self.stp_recover = torch.zeros(self.n_slots, device=self.device)
        self.stp_u_cell = torch.zeros(self.n_slots, device=self.device)

        for src, _, U, tau_rec in cfg.nucleus_stp:
            if src in self.areas:
                cells = self.areas[src].E.slice
                self.stp_u_cell[cells] = U
                self.stp_recover[cells] = dt / tau_rec
        self.decay_inh_gain = math.exp(-dt / cfg.inh_gain_tau_ms)
        self.decay_adapt = math.exp(-dt / cfg.e_adapt_tau)
        self.decay_plus = math.exp(-dt / cfg.stdp_tau_plus)
        # Ядро LTP со сдвигом: exp(-t/τ+) - exp(-t/τrise), нормировано на
        # пик 1; без сдвига (rise 0) — exp(-t/τ+).
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
        custom = [a for a in cfg.areas if a.stdp_tau_plus is not None]

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
        # Фон: число входных спайков за шаг ~ Poisson(λ); у ядер — общий
        # фон, у колонок — внешний вход Potjans (ниже).
        self.noise_mean = torch.full((self.n_slots,), cfg.background_rate_hz * dt / 1000.0, device=self.device)
        self.noise_gain = self._fill(cfg.background_weight, cfg.background_weight)

        # Колонки: внешний вход — остальной мозг, Пуассон K × частота с PSP
        # 0.15 мВ каждой клетке (Potjans Table 5).
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
            (not path.plastic or path.learned)
            and src is not None and dst is not None
            and src.layers is not None and dst.layers is not None and src is not dst
            and path.psp_mv is None and path.k > 0
        )

    def _external_minus_modeled(self) -> None:
        """
        Внешний вход Potjans — всё, что приходит в клетку извне колонки,
        включая таламус и другие зоны (Table 3). Явные пути модели — уже
        его часть: внешних входов остаётся K_ext − явные (так у Schmidt et al.
        2018 в мультиобластной модели). Анатомический мозг.
        """

        cfg = self.cfg
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

        direction = path.direction or next(
            (d for s, t, d in self.cfg.pathway_directions if (s, t) == (path.src, path.dst)), "",
        )
        return direction in ("thalamic", "thalamic_l1", "thalamic_l3") and dst.layers is not None

    def _thalamic_path(self, path, pre: Population, post: Population, inhibitory: bool) -> tuple[int, float]:
        """
        Таламус → колонка по данным: ~85 таламических клеток на клетку L4
        (Bruno & Sakmann 2006), единичный ВПСП у взрослых in vivo 1.0 ± 0.2 мВ
        (Schoonover et al. 2014, n 11; у молодых — ~0.6, Bruno & Sakmann).
        Синапс депрессирует (PPR 0.67 при 10–20 Гц, там же) — пока без
        кратковременной пластичности [долг: τ восстановления]; на тормозные
        L4 — ~56 входов (Potjans: 0.062 × 902), синапс на быстрые тормозные в
        2–3 раза сильнее (Cruikshank, Lewis & Connors 2007) — ×2. Моторный
        таламус — в L1 и L3 одной дозой (anatomy._motor_thalamic). Масштаб —
        как у межзонных путей: вес × √(k_data / k), недостающее — ток по частоте
        ядра in vivo.
        """

        from brain import anatomy

        cfg = self.cfg
        k_data = 56.0 if inhibitory else 85.0
        motor = ("thalamic_l1", "thalamic_l3")

        if path.direction in motor:
            # Моторный таламус: одна таламическая доза клетки (85 входов) на
            # все сходящиеся на зону ядра и на L1 и L3 — не по дозе сенсорного
            # реле на каждое ядро. [замер] по дозе на ядро L4 руки и моторной
            # коры стреляли 34–48 Гц при цели 1.4 (tools/input_trace.py).
            # Сколько синапсов моторного таламуса у клетки M1 — не мерено
            # [сверить]; на тормозные — нет (Bopp et al. 2017, L4 M1 мыши).
            if inhibitory:
                return 0, 0.0

            k_data /= sum(1 for p in cfg.pathways if p.dst == path.dst and p.direction in motor)

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
        self.nmda_on = cfg.nmda
        tau = cfg.nmda_tau_ms
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

        terms = self.mean_field
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

        # Среднее поле путей с депрессией (cfg.nucleus_stp): недостающие
        # синапсы устают, как построенные — × средний ресурс клеток источника.
        self.mf_stp = torch.zeros(Q + 1, P + 1, dtype=torch.bool, device=dev)
        self.mf_stp_on = False
        stp_pairs = {(src, dst) for src, dst, *_ in self.cfg.nucleus_stp}

        for post, pre, coeff, rate, ach, slow in terms:
            q, i = post_index[(post.start, post.size)], pre_index[(pre.start, pre.size)]
            src_area = pre.name.split(".")[0]

            if (src_area, post.name) in stp_pairs or (src_area, post.name.split(".")[0]) in stp_pairs:
                self.mf_stp[q, i] = True
                self.mf_stp_on = True
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
        self.mf_decay = math.exp(-self.cfg.dt_ms / self.cfg.mean_field_tau_ms)
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
        k_data = cfg.interareal_share * (local + outside) / max(1, self.cortical_inputs.get(post.name, 1))
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

        direction = path.direction or next(
            (d for s, t, d in self.cfg.pathway_directions if (s, t) == (path.src, path.dst)),
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
            if direction in ("thalamic_l1", "thalamic_l3"):
                # Моторный таламус (anatomy._motor_thalamic): L1 — ветки
                # дендрита L2/3, L3 — тела L2/3; тормозным — нет.
                post, inhibitory, top_down = dst.layers["L23"]["E"], None, direction == "thalamic_l1"
            elif direction in ("ff", "thalamic") and "L4" in dst.layers:
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

        for name in self.cfg.map_areas:
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

    def _sheet_place(self, spec, area, pop: Population) -> torch.Tensor:
        """
        Место клетки колонки без карты на листе коры, мм: равномерно на
        квадрате площади (клеток зоны) / (клеток колонки 1 мм², Potjans) —
        все слои на одном квадрате.
        """

        cache = self.__dict__.setdefault("sheet_pos", {})

        if pop.name not in cache:
            per_mm2 = sum(e + i for _, e, i in self._column_layers(spec.name))
            cells = sum(p.size for _, p in area.populations)
            side = math.sqrt(cells / per_mm2)
            g = torch.Generator(device=self.device).manual_seed(self.cfg.seed * 7919 + pop.start)
            cache[pop.name] = torch.rand(pop.size, 2, device=self.device, generator=g) * side

        return cache[pop.name]

    def _sheet_pre(self, spec, area, post: Population, k: int, pre_pop: Population, sigma_mm: float) -> torch.Tensor:
        """k источников без повторов, вероятность ∝ exp(−d² / 2σ²) по листу."""

        a, b = self._sheet_place(spec, area, post), self._sheet_place(spec, area, pre_pop)
        out = []

        for chunk in torch.split(torch.arange(post.size, device=self.device), 512):
            w = torch.exp(-torch.cdist(a[chunk], b).pow(2) / (2 * sigma_mm ** 2)) + 1e-12

            if post is pre_pop:
                w[torch.arange(len(chunk), device=self.device), chunk] = 0.0
            out.append(torch.multinomial(w, k, replacement=False, generator=self.generator))

        return pre_pop.start + torch.cat(out)

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

        params = {(a, b): (U, rec, fac) for a, b, U, rec, fac in self.cfg.column_stp}

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
        own = tuple(spec.column_gains)

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

                        if spec.name in cfg.map_areas:
                            if spec.name == "visual" and (src_layer, src_kind, dst_layer, dst_kind) == ("L4", "E", "L23", "E"):
                                # Сложные клетки L2/3 собирают простые L4 своей
                                # ориентации рядом (Alonso & Martinez 1998).
                                pre_index = self._complex_pre(post, k, pre, 5.0)
                            else:
                                pre_index = self._nearest_pre(post, min(k, pre.size - (pre is post)), pre)
                            k = pre_index.shape[1]
                        elif cfg.column_geometry and not (src_kind == dst_kind == "E" and cfg.sigma_ee_um <= 0):
                            # Колонка без карты — лист коры: партнёр выбирается с
                            # вероятностью по расстоянию (Гаусс, σ — Campagnola
                            # Fig. 2A: E → E 125, E → I 99, I → E 96, I → I 131
                            # мкм); число входов прежнее.
                            if src_kind == dst_kind == "E":
                                sigma = cfg.sigma_ee_um
                            else:
                                sigma = cfg.sigma_ei_um if src_kind == "E" else cfg.sigma_ie_um if dst_kind == "E" else cfg.sigma_ii_um
                            pre_index = self._sheet_pre(spec, area, post, min(k, pre.size - (pre is post)), pre, sigma / 1000.0)
                            k = pre_index.shape[1]

                        if (src_layer, src_kind, dst_layer, dst_kind) == ("L23", "E", "L23", "E"):
                            # Обучаемая связь зоны (AreaSpec.ee_plastic): веса по
                            # данным — врождённо от 0 до 2 средних PSP пары,
                            # потолок learned_cap_psp средних.
                            w = weight * math.sqrt(max(1.0, k_data / k))
                            ee_weight = (0.0, 2.0 * w) if spec.ee_plastic else (w, w)
                            proj = self._connect(
                                name, pre, post, k, ee_weight, plastic=spec.ee_plastic,
                                cap=cfg.learned_cap_psp * w, modulator=spec.ee_modulator, pre_index=pre_index,
                            )
                            if proj is not None:
                                proj.data_weight = True
                            continue

                        proj = self._connect(name, pre, post, k, (weight * root, weight * root), pre_index=pre_index)

                        if proj is not None and spec.name in cfg.vogels_areas and (src_kind, dst_kind) == ("PV", "E"):
                            # Пластичность торможения PV → E (Vogels et al. 2011;
                            # D'amour & Froemke 2015) — декорреляция признаков.
                            proj.vogels = True
                            proj.cap = cfg.vogels_cap

                        if proj is not None and src_kind == "SOM" and dst_kind == "E" and dst_layer in ("L23", "L5"):
                            # Мартинотти — на пучки дендритов в слое 1 (L2/3 и L5).
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
        retinotopic: float = 0.0,
        pre_index: torch.Tensor | None = None,
    ) -> Projection | None:
        k = min(k, pre_pop.size)

        if k <= 0:
            return None

        # Врождённый путь с постоянным весом — не храним, а вычисляем
        # (топографический хранится: хеш получателей не знает карты).
        if self.cfg.procedural_wiring and pre_index is None and not plastic and weight[0] == weight[1] != 0 and not topographic and not retinotopic:
            proj = ProceduralProjection(
                name, pre_pop, post, k, weight[0],
                seed=self.cfg.seed * 1009 + len(self.projections), device=self.device,
            )
            self.projections.append(proj)
            return proj

        if pre_index is not None:
            # Готовые источники (колонка-карта: ближайшие по месту).
            pre = pre_index
        elif retinotopic > 0:
            pre = self._retinotopic_pre(post, k, pre_pop, retinotopic)
        elif topographic > 0:
            pre = self._topographic_pre(post, k, pre_pop, topographic)
        else:
            pre = self._random_pre(post, k, pre_pop)
        initial = self._uniform((post.size, k), *weight)

        proj = Projection(
            name,
            "net",
            pre,
            post,
            initial,
            plastic=plastic,
            cap=cap,
        )
        proj.pre_pop = pre_pop
        proj.modulator = modulator
        proj.ach_suppressed = ach_suppressed
        self.projections.append(proj)

        return proj

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

            if not spec.local_psp:
                # Местных тормозных нет — нет и местного контура.
                continue

            # Местный контур по данным (anatomy.NUCLEI): входов (по данным)
            # и PSP; клеток в модели меньше — недостающие связи средним
            # током по частоте источника in vivo (как у путей, k_data):
            # главные — anatomy.RATES_HZ, тормозные — частота из таблицы.
            from brain import anatomy

            k_ei, ei, k_ie, ie, k_ii, ii, i_hz = spec.local_psp[:7]
            # Необязательно — местное возбуждение главных (входов E → E, ВПСП).
            k_ee, ee = spec.local_psp[7:9] if len(spec.local_psp) >= 9 else (0, 0.0)
            e_hz = anatomy.RATES_HZ.get(spec.name, 1.0)

            for name, pre, post, k, psp, hz in (("E->I", area.E, area.I, k_ei, ei, e_hz),
                                                 ("I->E", area.I, area.E, k_ie, ie, i_hz),
                                                 ("I->I", area.I, area.I, k_ii, ii, i_hz),
                                                 ("E->E local", area.E, area.E, k_ee, ee, e_hz)):
                if k <= 0:
                    continue

                w = psp / self._psp_peak(post)
                built = max(1, min(k, pre.size))
                root = math.sqrt(k / built)
                tau = float(self.tau_m[post.start])
                self._missing(post, pre, k * hz / 1000.0 * w * tau * (1.0 - 1.0 / root), hz / 1000.0)
                # Местные связи по месту на карте органа: LGN (поле зрения),
                # CN и IC (полоса улитки); MGN — по флагу relay_local_inhibition.
                local = spec.name == "lgn" or (
                    spec.name == "mgn" and cfg.relay_local_inhibition) or spec.name in ("cn", "ic")
                index = self._lgn_local(area, pre, post, built) if local else None

                if spec.name == "cn" and name == "I->E" and index is not None:
                    index = self._cn_d_stellate_out(area, index)
                self._connect(f"{spec.name}: {name}", pre, post, built, (w * root, w * root), pre_index=index)

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
            src = self.areas[path.src]
            dst = self.areas[path.dst]
            pre, post, inhibitory, top_down = self._laminar(path, src, dst)
            k = path.k
            # Ветка на тормозные клетки получателя — только по данным (ниже).
            k_inhibitory, inhibitory_weight = 0, 0.0

            if path.psp_mv is not None:
                # Вес из измеренного PSP; обучаемый — от 0 до 2 средних,
                # потолок learned_cap_psp средних.
                w = path.psp_mv / self._psp_peak(post)
                weight = (w, w)

                if path.k_data > 0:
                    # Входов по данным больше, чем строим (PathwaySpec.k_data).
                    from brain import anatomy

                    k = max(1, min(path.k, pre.size))
                    root = math.sqrt(max(1.0, path.k_data / k))
                    rate = anatomy.RATES_HZ.get(path.src, 1.0) / 1000.0
                    tau = float(self.tau_m[post.start])
                    self._missing(post, pre, path.k_data * rate * w * tau * (1.0 - 1.0 / root), rate,
                                  ach=path.ach_suppressed)
                    w = w * root
                    weight = (w, w)

                if path.plastic:
                    weight = (0.0, 2.0 * w)
                    path = replace(path, cap=cfg.learned_cap_psp * abs(w))

                if path.k_data_inhibitory > 0 and inhibitory is not None:
                    # Прямое торможение: путь бьёт и по местным тормозным
                    # клеткам получателя (входов и PSP — по данным).
                    from brain import anatomy

                    wi = path.psp_inhibitory_mv / self._psp_peak(inhibitory)
                    k_inhibitory = max(1, min(round(path.k_data_inhibitory), pre.size))
                    root = math.sqrt(max(1.0, path.k_data_inhibitory / k_inhibitory))
                    rate = anatomy.RATES_HZ.get(path.src, 1.0) / 1000.0
                    tau = float(self.tau_m[inhibitory.start])
                    self._missing(inhibitory, pre, path.k_data_inhibitory * rate * wi * tau * (1.0 - 1.0 / root), rate,
                                  ach=path.ach_suppressed)
                    inhibitory_weight = wi * root

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
            elif path.psp_mv is None:
                raise ValueError(f"путь {path.src} -> {path.dst} без данных: нет psp_mv, и это не путь между колонками")

            # Путь по карте реле (TRN ↔ LGN): входы — ближайшие по месту.
            index = self._lgn_local(self.areas[path.mapped], pre, post, min(k, pre.size)) if path.mapped else None
            proj = self._connect(
                path.name or f"{path.src} -> {path.dst}", pre, post, k,
                weight, plastic=path.plastic, cap=path.cap,
                modulator=path.modulator, ach_suppressed=path.ach_suppressed,
                topographic=path.topographic, retinotopic=path.retinotopic, pre_index=index,
            )

            if proj is not None:
                proj.dendrite = top_down
                proj.cap = path.cap
                # Вес по данным (средний PSP пути — cap / learned_cap_psp):
                # шаг STDP — в долях этой средней связи (chemical_plasticity).
                proj.data_weight = path.plastic and (self._data_path(path) or path.psp_mv is not None)
                sigma = cfg.interareal_weight_sigma

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
                topographic=path.topographic, retinotopic=path.retinotopic,
            )

        # ---- органы-датчики: рецепторы -> своя зона ----
        # У каждого органа свой участок общего массива рецепторов: ухо —
        # первым (как всегда), дальше остальные.
        for sensor, offset in zip(self.sensors, self.sensor_offsets):
            area = self.areas[sensor.area]
            relay = self.areas.get(sensor.relay)

            if relay is not None and sensor.brainstem:
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
                self._triad(self.projections[-1])
                # Местные тормозные клетки реле получают сетчатку так же (1–3
                # входа, триады; ~25% клеток таламуса приматов — Arcelli et al.
                # 1997).
                k = 2

                if cfg.relay_local_inhibition:
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
                elif relay is not None and sensor.wiring == "oriented" and pop is inhibitory:
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

                    if sensor.psp_mv is not None:
                        weight = torch.full((pop.size, k), sensor.psp_mv / self._psp_peak(pop), device=self.device)

                if relay is not None:
                    # Таламус → L4 по данным, как у остальных путей таламус →
                    # кора (_thalamic_path): на E —
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
                    self.projections.append(proj)
                    self.modeled_inputs[pop.name] = self.modeled_inputs.get(pop.name, 0.0) + k_data
                    continue

                self.projections.append(
                    Projection(
                        f"{sensor.name} -> {pop.name}", "sensory", pre + offset, pop, weight * gain,
                    )
                )

        self._v1_to_colliculus()
        self._external_minus_modeled()
        self._build_mean_field()
        self._pack()
        self._init_nmda()

    def _v1_to_colliculus(self) -> None:
        """
        V1 L5 → верхнее двухолмие (Lund et al. 1975; Schiller & Tehovnik 2001,
        примат): по месту в поле. Карты обе вокруг точки взгляда — у V1 окно
        ямки (столбец − fovea_col, px), у двухолмия боковое поле (место клетки
        − середина, как цель саккады в session); клетка двухолмия в пределах
        окна ямки получает cfg.v1_colliculus ближайших по горизонтали клеток
        V1 L5, вне окна — без этого входа.
        """

        cfg = self.cfg
        k, psp = cfg.v1_colliculus

        if not (k and self.eye_sensor is not None
                and "colliculus" in self.areas and self.areas.get("visual") is not None
                and self.areas["visual"].layers and "L5" in self.areas["visual"].layers):
            return

        L5 = self.areas["visual"].layers["L5"]["E"]
        SC = self.areas["colliculus"].E
        _, _, cols = self._oriented_layout(self.eye_sensor, L5.size)
        x_v1 = cols.float() - cfg.fovea_col
        field = cfg.retina_cols * cfg.periphery_fields
        x_sc = (torch.arange(SC.size, device=self.device).float() + 0.5) * field / SC.size - field / 2
        inside = (x_sc >= float(x_v1.min())) & (x_sc <= float(x_v1.max()))
        d = (x_sc[:, None] - x_v1[None, :]).abs() + torch.rand((SC.size, L5.size), device=self.device, generator=self.generator)
        pre = d.topk(min(k, L5.size), dim=1, largest=False).indices + L5.start
        w = psp / self._psp_peak(SC)
        weight = torch.where(inside[:, None], w, 0.0).expand(SC.size, pre.shape[1]).contiguous()
        proj = Projection(f"{L5.name} -> {SC.name}", "net", pre.to(torch.int32), SC, weight)
        proj.pre_pop = L5
        self.projections.append(proj)

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

        if area.name in ("mgn", "cn", "ic", "vnll"):
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
            # «ic.I» — тормозные клетки ядра (ГАМК-проекция IC → MGN).
            area_name, _, kind = name.partition(".")
            area = self.areas[area_name]
            pop = area.I if kind == "I" else area.E
            return bands(area, pop), torch.arange(pop.start, pop.stop, device=self.device), "net", (area, pop)

        for src, dst, k, spread, psp in cfg.brainstem_links:
            name, kind = dst.split(".")
            area = self.areas[name]
            post = area.E if kind == "E" else area.I
            src_band, src_index, table, src_area = source(src)
            order = torch.argsort(src_band)
            counts = torch.bincount(src_band, minlength=cfg.cochlea_bands)
            starts = torch.cumsum(counts, 0) - counts
            spread = torch.full((post.size, 1), spread, device=self.device)

            if dst == "cn.I":
                # Тормозные CN: каждая ~1/доля-я — D-звёздчатая (широкая
                # полоса), прочие — L-звёздчатые (cfg.cn_d_stellate).
                share, wide = cfg.cn_d_stellate
                every = max(1, round(1.0 / share)) if share > 0 else post.size + 1
                spread[torch.arange(post.size, device=self.device) % every == 0] = wide

            want = bands(area, post)[:, None] + (torch.rand((post.size, k), device=self.device, generator=g) * (2 * spread + 1)).long() - spread
            want = want.clamp(0, cfg.cochlea_bands - 1)
            pick = starts[want] + (torch.rand((post.size, k), device=self.device, generator=g) * counts[want]).long()
            pre = src_index[order[pick]]
            weight = torch.full((post.size, k), psp / self._psp_peak(post), device=self.device)
            proj = Projection(f"{src} -> {post.name}", table, pre if table == "sensory" else pre.to(torch.int32), post, weight)

            if src_area is not None:
                proj.pre_pop = src_area[1]

            self.projections.append(proj)

            if dst == "mgn.E" and psp > 0:
                self._triad(proj)

    def _path_delay_steps(self, p) -> int:
        """
        Задержка пути от спайка до тока в получателе (cfg.path_delays) — шагов
        сверх обычного одного. Внутри зоны — синапс и местный аксон
        (delay_local_ms); кора → кора — delay_cortical_ms; таламус → кора —
        delay_thalamocortical_ms; прочее (кора → ядра, ядра ↔ ядра) —
        delay_subcortical_ms.
        """

        cfg = self.cfg

        if not cfg.path_delays:
            return 0

        src = getattr(p, "pre_pop", None)
        src = src.name.split(".")[0] if src is not None else ""
        dst = p.post.name.split(".")[0]
        a, b = self.areas.get(src), self.areas.get(dst)
        cortex = lambda area: area is not None and area.layers is not None
        relays = {"lgn", "mgn", "md", "thalamus", "vl_cb"}

        if src == dst or not src:
            ms = cfg.delay_local_ms
        elif cortex(a) and cortex(b):
            ms = cfg.delay_cortical_ms
        elif src in relays and cortex(b):
            ms = cfg.delay_thalamocortical_ms
        else:
            ms = cfg.delay_subcortical_ms

        return max(0, round(ms / cfg.dt_ms) - 1)

    def _triad(self, driver: Projection) -> None:
        """
        Триада реле таламуса (Sherman & Guillery 2006; кошка — Montero 1991,
        Van Horn et al. 2000): окончание водителя возбуждает реле и дендрит
        интернейрона, дендрит сразу тормозит то же реле — без спайка сомы
        интернейрона. Здесь — парный тормозной синапс от того же водителя на
        то же реле: проводимость cfg.triad_ratio от возбуждающей (прямое
        торможение реле LGN мыши — IPSC/EPSC 0.53 при равных движущих силах,
        eNeuro 2024). Вес торможения нормирован к ipsp_holding, возбуждения —
        к epsp_holding: w_I = r · w_E · (V_ipsp − E_GABA) / (E_AMPA − V_epsp).
        Задержка ~1 мс (Blitz & Regehr 2005) не воспроизведена — приходит в
        тот же шаг, что ВПСП (задержки путей — только медленным путём).
        """

        cfg = self.cfg
        ratio = cfg.triad_ratio

        if ratio <= 0:
            return

        scale = (cfg.ipsp_holding_mv - cfg.gaba_reversal_mv) / (cfg.ampa_reversal_mv - cfg.epsp_holding_mv)
        proj = Projection(f"{driver.name} (триада)", driver.source, driver.pre.clone(), driver.post, -ratio * scale * driver.weight.abs())
        proj.pre_pop = getattr(driver, "pre_pop", None)
        self.projections.append(proj)

    def _cn_d_stellate_out(self, area, index: torch.Tensor) -> torch.Tensor:
        """
        Выход D-звёздчатых CN широкий (аксоны по многим изочастотным
        пластинам; Oertel et al. 1990; Smith & Rhode 1989): доля входов
        T-клетки, равная доле D среди тормозных (cfg.cn_d_stellate), берётся
        у D-клеток в пределах ± полос их входа, остальные — ближайшие L.
        """

        share, wide = self.cfg.cn_d_stellate
        every = max(1, round(1.0 / share)) if share > 0 else 0

        if not every:
            return index

        I = area.I
        d_cells = torch.arange(0, I.size, every, device=self.device)
        d_band = self._relay_place(area, I)[d_cells, 0].long()
        e_band = self._relay_place(area, area.E)[:, 0].long()
        n_d = max(1, round(index.shape[1] * share))
        g = self.generator
        # Для каждой T-клетки — случайные D из её ± wide полос.
        near = (d_band[None, :] - e_band[:, None]).abs() <= wide
        weights = near.float() + 1e-9
        pick = torch.multinomial(weights, n_d, replacement=False, generator=g)
        out = index.clone()
        out[:, -n_d:] = (I.start + d_cells[pick]).to(out.dtype)
        return out

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

        def flatten(group, dummy: bool, posts: bool = True):
            # Плоские массивы заводятся сразу, пути копируются в них по
            # одному, и прежний тензор пути тут же освобождается. [замер]
            # склейка списков (cat из int64 и копий expand) держала три
            # копии всех синапсов — пик ~10 ГБ видеопамяти на мозге 388 тыс.
            total = sum(p.weight.numel() for p in group) + (self.DUMMY_SYNAPSES if dummy else 0)
            pre = torch.empty(total, dtype=torch.int32, device=self.device)
            post = torch.empty(total, dtype=torch.int32, device=self.device) if posts else None
            weight = torch.empty(total, dtype=torch.float32, device=self.device)
            offset = 0

            for p in group:
                n = p.weight.numel()
                shape = p.weight.shape
                pre[offset: offset + n].view(shape).copy_(p.pre)
                if posts:
                    cells = torch.arange(p.post.start, p.post.stop, dtype=torch.int32, device=self.device)
                    post[offset: offset + n].view(shape).copy_(cells[:, None].expand(shape))
                weight[offset: offset + n].view(shape).copy_(p.weight)
                p.pre = pre[offset: offset + n].view(shape)
                p.weight = weight[offset: offset + n].view(shape)
                p.offset = offset
                offset += n

            if dummy:
                # Фиктивные синапсы: фиктивный нейрон -> фиктивный нейрон, вес 0.
                ids = self.n + torch.arange(self.DUMMY_SYNAPSES, device=self.device) % self.DUMMY_NEURONS
                pre[offset:] = ids
                if posts:
                    post[offset:] = ids
                weight[offset:] = 0.0

            return pre, post, weight

        # Получатель синапса на видеокарте считает ядро доставки по пути
        # (path_post, path_fanin) — массив получателей там не нужен (1.3 ГБ на
        # 327 млн синапсов); шагу на PyTorch его соберёт _posts.
        gpu = self.device.type == "cuda" and _kernels.deliver is not None
        self.net_pre, self.net_post, self.net_weight = flatten(net, dummy=True, posts=not gpu)
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
        self.ear_inhibitory = bool((self.ear_weight < 0).any())

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

        # Пути на ветки дендрита (обратные пути коры, SOM → E).
        dendritic = [p for p in net if getattr(p, "dendrite", False)]
        suppressed = [p for p in net if getattr(p, "ach_suppressed", False)]
        self.dendrite_syn = None
        self.ach_mask = None
        # Оба признака постоянны внутри пути: ядро доставки (brain/csrc)
        # находит путь синапса поиском по началам путей — масок на каждый
        # синапс не нужно (0.84 ГБ на 279 млн синапсов). Маски — только для
        # шага на PyTorch (процессор).
        paths = sorted((p for p in net if p.weight.numel()), key=lambda p: p.offset)
        # Кратковременная депрессия путей ядер (cfg.nucleus_stp): ресурс — у
        # клетки-источника (stp_x), прочие выходы этих клеток — без истощения.
        stp_sources = {src for src, *_ in self.cfg.nucleus_stp}
        stp_paths = {(src, dst) for src, dst, *_ in self.cfg.nucleus_stp}

        def no_depletion(p) -> bool:
            # Получатель в nucleus_stp — зона («gpi») или популяция («ca3.E»).
            src = getattr(p, "pre_pop", None)
            src = src.name.split(".")[0] if src is not None else ""
            return src in stp_sources and (src, p.post.name.split(".")[0]) not in stp_paths and (src, p.post.name) not in stp_paths

        flags = [int(p in dendritic) | (int(p in suppressed) << 1) | (int(no_depletion(p)) << 2) for p in paths]
        free = [p for p in paths if no_depletion(p)]
        self.stp_free = None

        if free:
            self.stp_free = torch.zeros(self.net_weight.shape, dtype=torch.bool, device=self.device)
            for p in free:
                self.stp_free[p.offset: p.offset + p.weight.numel()] = True
        self.path_start = torch.tensor([p.offset for p in paths] + [self.n_synapses], dtype=torch.long, device=self.device)
        self.path_flags = torch.tensor(flags + [0], dtype=torch.uint8, device=self.device)
        # Задержка каждого пути (шагов сверх одного) — по его типу
        # (cfg.path_delays); кольцо — в _init_state.
        delays = [self._path_delay_steps(p) for p in paths]
        self.path_delay = torch.tensor(delays + [0], dtype=torch.int32, device=self.device)
        self.ring_slots = max(delays + [0]) + 1
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

    # Серотонин — дорсальное ядро шва, ацетилхолин — ядро Мейнерта, орексин —
    # латеральный гипоталамус.
    MODULATORS = {"dopamine": "vta", "noradrenaline": "lc", "adenosine": None, "galanin": "vlpo",
                  "serotonin": "raphe", "acetylcholine": "nbm", "orexin": "orexin"}

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
        names = list(modulators)
        self.chemistry_names = names
        # float64: аденозин копится часами — прибавки за шаг ~1e-10.
        self.chemistry = torch.zeros(len(names), device=self.device, dtype=torch.float64)
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

        # Ацетилхолин из ядра Мейнерта: уровень, который читают пути, —
        # занятость его рецепторов, а не число из кода.
        self.ach_chemistry = names.index("acetylcholine")

        # Часы (SCN): ток клеткам часов по фазе суток — молекулярный цикл
        # клетки, выставляется set_clock (copy_ — для CUDA Graph).
        self.clock_mv = torch.zeros(self.n_slots, device=self.device)
        # Спит мозг или нет, решает переключатель (выброс VLPO — галанин —
        # больше норадреналина LC, Saper 2001), а не код: по нему в каждом
        # шаге выбираются вымывание аденозина и прочее.
        f64 = lambda xs: torch.tensor(xs, device=self.device, dtype=torch.float64)
        rates = [self._chemistry_rates(s) for s in (False, True)]
        self.chemistry_by_state = [
            (torch.exp(-self.cfg.dt_ms / f64(tau)), f64(release), f64(basal)) for tau, release, basal in rates
        ]
        self.galanin_i, self.noradrenaline_i = names.index("galanin"), names.index("noradrenaline")
        # Мозг стартует бодрствующим, а не «без химии»: вещество ядра — в
        # устойчивом уровне наяву, c = частота ядра × τ / 1000 (спайков на
        # клетку за τ; anatomy.RATES_HZ). Без этого первые сотни мс
        # ацетилхолина нет — коллатерали CA3 не подавлены, и стартовый
        # всплеск разжигал судорогу, которая держалась и после подъёма
        # ацетилхолина (tools/hippo_onset.py). Аденозин — давление сна,
        # утром после сна низкое: с нуля.
        from brain import anatomy

        for i, name in enumerate(names):
            hz = anatomy.RATES_HZ.get(modulators[name] or "", 0.0)

            if name != "adenosine" and hz > 0:
                self.chemistry[i] = hz * getattr(self.cfg, f"{name}_tau_ms") / 1000.0

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
        branches = max(1, cfg.dendrite_branches)
        self.tick = torch.zeros(1, dtype=torch.long, device=self.device)
        # Кольцо задержек путей (cfg.path_delays): [шаг, вход (возб, торм,
        # LAMP5, NMDA), клетка] и для веток дендрита; tick — номер шага.
        self.ring = torch.zeros(self.ring_slots, 4, self.n_slots, device=self.device)
        self.dend_ring = torch.zeros(self.ring_slots, self.n_slots * branches, device=self.device)
        self.i_syn = z()
        # Тормозной ток (проводимость): отдельно от возбуждающего.
        self.i_inh = z()
        # Медленное торможение от LAMP5 (см. decay_ngf).
        self.i_ngf = z()
        # Ток веток дендрита (пути с признаком dendrite).
        self.i_dend = torch.zeros(self.n_slots * branches, device=self.device)
        self.adapt = z()
        self.refractory = z()
        self.spikes = z()
        # STDP-следы: пре (LTP) и пост (LTD).
        self.trace_pre = z()
        self.trace_pre_rise = z()
        self.ltp_kernel = z()
        self.trace_post = z()
        # Счётчик спайков для окна наблюдения.
        self.counts = z()
        # Кто стрелял с прошлого показа (наблюдатель, on_chunk).
        self.seen = z()
        # Ток покоя (мВ): расстояние до порога по типу клетки и фон ядер.
        self.bias = z()
        gap = cfg.v_threshold - cfg.v_rest

        for area in self.areas.values():
            if area.layers is None:
                continue

            for group in area.layers.values():
                # До порога — как у клетки этого типа (реобаза × сопротивление).
                for kind, data_gap in (("E", cfg.column_e_gap_mv), ("PV", cfg.pv_gap_mv), ("SOM", cfg.som_gap_mv),
                                       ("VIP", cfg.vip_gap_mv), ("LAMP5", cfg.lamp5_gap_mv)):
                    if kind in group:
                        self.bias[group[kind].slice] = gap - data_gap

                # Разброс между клетками (Allen; config column_gap_sd_mv),
                # расстояние до порога не меньше 3 мВ.
                for kind, sd in zip(KINDS, cfg.column_gap_sd_mv):
                    if kind not in group:
                        continue
                    pop = group[kind]
                    spread = torch.randn(pop.size, device=self.device, generator=self.generator) * sd
                    self.bias[pop.slice] = (self.bias[pop.slice] + spread).clamp(max=gap - 3.0)

        for spec in cfg.areas:
            hz = spec.autonomous_hz

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
        for name, mv in cfg.nucleus_bias_mv:
            if name in self.areas:
                self.bias[self.areas[name].E.slice] += mv

        # Разброс возбудимости между клетками ядра — как у главных клеток
        # колонок (Allen, column_gap_sd_mv E): без него все клетки ядра
        # одинаковы и включаются разом — ядро «всё или ничего» (узел сна:
        # проигравшая сторона молчала целиком, а в записях стреляет реже).
        sd = cfg.column_gap_sd_mv[0]

        for area in self.areas.values():
            if area.layers is None:
                E = area.E
                self.bias[E.slice] += torch.randn(E.size, device=self.device, generator=self.generator) * sd

        # Ацетилхолин — занятость его рецепторов (ядро Мейнерта); тензор — его
        # читает записанный шаг CUDA Graph.
        self.acetylcholine = torch.zeros(1, device=self.device)
        # T-ток реле таламуса (cfg.thalamic_t_current, модель IFB — Smith et
        # al. 2000): ворота h — восстановление ниже V_h, закрытие выше.
        self.t_mask = torch.zeros(self.n_slots, dtype=torch.bool, device=self.device)

        if cfg.thalamic_t_current:
            for name in ("lgn", "mgn", "md", "thalamus", "vl_cb"):
                if name in self.areas:
                    self.t_mask[self.areas[name].E.slice] = True
        self.t_h = z()
        # Толчок (kick, стенды): ток на несколько шагов, гаснет сам.
        self.kick_current = z()
        self.kick_left = torch.zeros(1, device=self.device)
        # Истинная eligibility = eligibility * eligibility_scale:
        # затухание всех следов — одно умножение скаляра.
        self.eligibility_scale = torch.ones(1, device=self.device)
        # Спайки, не влезшие в таблицу шага (должно быть 0).
        self.dropped = torch.zeros(1, device=self.device)
        self._init_chemistry()
        # Ресурс пресинапса путей ядер с депрессией (nucleus_stp; 1 — полный)
        # и усиление торможения, приходящего к клетке (1 — норма).
        self.stp_x = torch.ones(self.n_slots, device=self.device)
        # Кратковременная пластичность клеток колонок: u (до первого спайка
        # 0) и ресурс x к каждому типу получателя.
        self.stp_u4 = torch.zeros(self.n_slots, len(KINDS), device=self.device)
        self.stp_x4 = torch.ones(self.n_slots, len(KINDS), device=self.device)
        # Десенситизация рецепторов ГАМК-А от LAMP5 (нейроглиаформных):
        # второе, медленное состояние их синапсов (Karayannis et al. 2010).
        self.is_lamp5 = self.cell_type == KINDS.index("LAMP5")
        # Старт — устоявшаяся сила при частоте покоя LAMP5 (1/(1 + U·f·τ)):
        # восстановление идёт минуты, мозг не «рождается» с отдохнувшими
        # синапсами — иначе любой замер покоя был бы переходным процессом.
        lamp5_hz = cfg.column_target_hz[0][1 + KINDS.index("LAMP5")]
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
            hz = torch.as_tensor(sensor.spontaneous_hz, dtype=torch.float32, device=self.device)
            spontaneous[offset: offset + sensor.size] = hz * cfg.dt_ms / 1000.0

        self.spontaneous = spontaneous if bool((spontaneous > 0).any()) else None

        self.ear_input = torch.zeros(
            self.CHUNK, self.n_receptors, device=self.device
        )

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
        glutamate = None

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
                self.path_delay, self.tick, self.ring, self.dend_ring,
            )

            if self.ring_slots > 1:
                # Пришедшее сейчас из кольца задержек — во входы шага.
                now = self.tick % self.ring_slots
                arrived = self.ring.index_select(0, now).squeeze(0)
                drive.add_(arrived[0])
                inhibition.add_(arrived[1])
                ngf.add_(arrived[2])

                if glutamate is not None:
                    glutamate.add_(arrived[3])
                self.i_dend.add_(self.dend_ring.index_select(0, now).squeeze(0).view_as(self.i_dend))
                self.ring.index_fill_(0, now, 0.0)
                self.dend_ring.index_fill_(0, now, 0.0)
                self.tick.add_(1)
        else:
            # Шаг на PyTorch (процессор, нет ядер): без задержек путей.
            synapses, pre_of = self._events(self.out_all, fired_before)
            post = self._posts().index_select(0, synapses)
            weight = self.net_weight.index_select(0, synapses)

            if self.ach_mask is not None:
                # Ацетилхолин подавляет передачу по отмеченным путям.
                weight = weight * (1.0 - self.acetylcholine * self.ach_mask.index_select(0, synapses))

            # Пути с признаком dendrite — на ветки дендрита, не в сому.
            to_dendrite = (
                self.dendrite_pre.index_select(0, pre_of)
                & self.dendrite_post.index_select(0, post)
            )

            if self.dendrite_syn is not None:
                to_dendrite = to_dendrite | self.dendrite_syn.index_select(0, synapses)
            branches = max(1, cfg.dendrite_branches)
            slot = post.long() * branches + self.net_branch.index_select(0, synapses).long()
            dendrite = torch.zeros_like(self.i_dend).index_add_(0, slot, weight * to_dendrite)
            self.i_dend.mul_(self.decay_dend).add_(dendrite)
            weight = weight * ~to_dendrite

            # Синапсы клеток колонок: свои U, ресурс и облегчение к каждому
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

            # Истощение путей ядер (nucleus_stp): синапс отдаёт долю
            # оставшегося ресурса пресинапса (у клеток колонок — своё, выше).
            depletion = self.stp_x.index_select(0, pre_of)

            if self.stp_free is not None:
                depletion = torch.where(self.stp_free.index_select(0, synapses), 1.0, depletion)

            depletion = torch.where(typed, 1.0, depletion)
            weight = weight * depletion
            # Торможение к клетке × её усиление (растёт, если клетка
            # стреляет выше нормы).
            weight = torch.where(weight < 0, weight * self.inh_gain.index_select(0, post), weight)

            # Торможение — проводимость: копится отдельно, к соме приходит ×
            # движущая сила (см. ниже).
            inh = weight.clamp(max=0.0)
            # От LAMP5 — в медленный ток (lamp5_tau_syn), остальное — в быстрый.
            slow = self.cell_type.index_select(0, pre_of) == KINDS.index("LAMP5")
            inhibition = torch.zeros_like(drive).index_add_(0, post, torch.where(slow, 0.0, inh))
            ngf = torch.zeros_like(drive).index_add_(0, post, torch.where(slow, inh, 0.0))
            weight = weight.clamp(min=0.0)

            if self.nmda_on:
                # NMDA — от тех же возбуждающих синапсов.
                glutamate = torch.zeros_like(drive).index_add_(0, post, weight)

            drive.index_add_(0, post, weight)

        if self.proc_rows is not None:
            # Врождённые пути с постоянным весом: получатели — из хеша.
            proc_post, proc_weight = self.proc_rows.rows(fired_before)
            drive.index_add_(0, proc_post.flatten(), proc_weight.flatten())

        if self.spontaneous is not None:
            # Рецепторы стреляют и в тишине (SensorSpec.spontaneous_hz).
            heard_now = (heard_now + (torch.rand_like(heard_now) < self.spontaneous).float()).clamp(max=1.0)

        heard = heard_now.index_select(0, self.ear_pre)

        if self.ear_inhibitory:
            # Тормозные синапсы рецепторов (триады сетчатка → LGN) — в
            # проводимость торможения, как у синапсов мозга.
            heard = heard * self.ear_weight
            drive.index_add_(0, self.ear_post, heard.clamp(min=0.0))
            inhibition.index_add_(0, self.ear_post, heard.clamp(max=0.0))
        else:
            drive.index_add_(0, self.ear_post, heard * self.ear_weight)

        if plastic:
            # LTD: пре пришёл после поста.
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
        self.i_inh.mul_(self.decay_syn).add_(inhibition * self.syn_scale)
        self.i_ngf.mul_(self.decay_ngf).add_(ngf * self.ngf_scale)

        if self.nmda_on:
            self.i_nmda.mul_(self.decay_nmda).add_(glutamate * self.nmda_scale)

        # ---- мембрана (AdEx / LIF) ----
        exp_term = torch.where(
            self.is_adex,
            self.delta_t
            * torch.exp(
                ((self.v - cfg.v_threshold) / self.safe_delta_t).clamp(max=10.0)
            ),
            0.0,
        )

        synaptic = self.i_syn
        # Движущая сила ГАМК-А: ток ∝ (v − E_ГАМК), нормирован к −55 мВ, где
        # мерили IPSP (Campagnola et al. 2022); у реверсии торможение только
        # шунтирует, ниже неё не тянет (Gulledge & Stuart 2003).
        force = ((self.v - cfg.gaba_reversal_mv) / (cfg.ipsp_holding_mv - cfg.gaba_reversal_mv)).clamp(min=0.0)
        # Возбуждение — тоже проводимость: ток ∝ (E_AMPA − v), нормирован к
        # −70 мВ, где мерили EPSP. Сильный фон открывает каналы и сам
        # укорачивает мембрану — высокопроводящее состояние коры in vivo
        # (Destexhe & Paré 1999; Destexhe, Rudolph & Paré 2003).
        excite = ((cfg.ampa_reversal_mv - self.v) / (cfg.ampa_reversal_mv - cfg.epsp_holding_mv)).clamp(min=0.0)
        dc_e, dc_i = self.dc_mv, self.dc_inh_mv

        if self.mf_on:
            # Недостающие соседи стреляют, как их построенная популяция
            # (частота сглажена, mean_field_tau_ms): ток — матрица популяций ×
            # частоты.
            self.mf_counts.zero_().index_add_(0, self.mf_pre_id, self.spikes)
            inst = self.mf_counts / self.mf_size / cfg.dt_ms
            self.mf_rate.mul_(self.mf_decay).add_(inst * (1.0 - self.mf_decay))
            exc = self.mf_exc - self.acetylcholine * self.mf_ach
            inh = self.mf_inh
            # Кратковременная пластичность и у недостающих соседей: их
            # синапсы устают так же, как у построенных клеток той же
            # популяции (средняя сила к типу получателя). Без этого L2/3 на
            # 330 Гц кормила себя без депрессии и держалась «горящей» при PV
            # на 400 Гц. Сила следующего спайка: u⁺ = u + U(1 − u), выброс
            # u⁺·x, к силе покоя U.
            nxt = self.stp_u4 + self.stp_U4 * (1.0 - self.stp_u4)
            gain = nxt * self.stp_x4 / self.stp_U4
            gain = torch.where(self.is_lamp5[:, None], gain * self.ngf_d[:, None], gain)
            gain = torch.where(self.typed_pre[:, None], gain, 1.0)
            n = len(KINDS)
            self.mf_gain[:, :n].zero_().index_add_(0, self.mf_pre_id, gain)
            self.mf_gain[:, :n].div_(self.mf_size[:, None])
            g = self.mf_gain.index_select(1, self.mf_post_type).t()
            exc, inh = exc * g, inh * g

            if self.mf_stp_on:
                resource = torch.zeros_like(self.mf_rate).index_add_(0, self.mf_pre_id, self.stp_x) / self.mf_size
                exc = torch.where(self.mf_stp, exc * resource[None, :], exc)

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

        # Плато на дендрите — на самой горячей ветке, если её ток выше порога:
        # усиливает возбуждающий вход сомы и сам поднимает сому
        # (dendrite_soma_mv).
        branch = self.i_dend.view(self.n_slots, max(1, cfg.dendrite_branches))
        plateau = torch.sigmoid(
            (branch - cfg.dendrite_threshold_mv) / cfg.dendrite_slope_mv
        ).amax(1)
        synaptic = synaptic + cfg.dendrite_gain * plateau * synaptic.clamp(min=0.0) + self.dendrite_soma * plateau

        # Вещества через рецепторы: занятость c / (c + K) × ток клетки.
        occupancy = (self.chemistry / (self.chemistry + self.chemistry_half)).to(torch.float32)
        modulated = occupancy @ self.receptor_mv

        # T-ток (IFB): ток в мВ покоя = g_T / g_L · h · (E_T − V) при V ≥ V_h.
        t_open = self.t_mask & (self.v >= cfg.t_vh_mv)
        t_current = torch.where(t_open, cfg.t_gt_ratio * self.t_h * (cfg.t_et_mv - self.v), 0.0)
        dv = (
            cfg.v_rest - self.v + exp_term - self.adapt
            + synaptic + self.bias + modulated + self.clock_mv
            + self.kick_current * (self.kick_left > 0) + t_current
        ) * self.decay_m
        # Ворота h: ниже V_h восстанавливаются (τ+) до равновесия h∞(V), выше —
        # закрываются (τ−).
        h_inf = torch.sigmoid((cfg.t_h_half_mv - self.v) / cfg.t_h_slope_mv)
        self.t_h.copy_(torch.where(
            self.t_mask,
            torch.where(self.v < cfg.t_vh_mv, self.t_h + (h_inf - self.t_h) * (cfg.dt_ms / cfg.t_tau_rec_ms),
                        self.t_h - self.t_h * (cfg.dt_ms / cfg.t_tau_inact_ms)),
            0.0,
        ))
        v = torch.where(self.refractory <= 0, self.v + dv, self.v)
        self.refractory.sub_(cfg.dt_ms)

        fired = v >= self.v_peak
        self.spikes.copy_(fired)
        # Выброс веществ ядрами: доля выстреливших клеток каждого ядра.
        released = (self.chemistry_source @ self.spikes).to(torch.float64)
        # Спит мозг или нет, решает переключатель: выброс VLPO (галанин)
        # больше норадреналина LC (Saper 2001).
        (dw, rw, bw), (ds, rs, bs) = self.chemistry_by_state
        asleep = self.chemistry[self.galanin_i] > self.chemistry[self.noradrenaline_i]
        self.chemistry.mul_(torch.where(asleep, ds, dw)).add_(released * torch.where(asleep, rs, rw) + torch.where(asleep, bs, bw))
        self.acetylcholine.copy_(occupancy[self.ach_chemistry : self.ach_chemistry + 1])

        self.v.copy_(torch.where(fired, cfg.v_reset, v))
        self.refractory.copy_(torch.where(fired, cfg.refractory_ms, self.refractory))
        self.adapt.mul_(self.decay_adapt).add_(self.adapt_b * self.spikes)
        # Ресурс путей ядер с депрессией восстанавливается к 1, спайк тратит
        # долю stp_u_cell (nucleus_stp).
        self.stp_x.add_((1.0 - self.stp_x) * self.stp_recover)
        self.stp_x.sub_(self.stp_u_cell * self.stp_x * self.spikes)
        # Усиление торможения: каждый спайк E поднимает, медленно
        # возвращается к 1 (в равновесии 1 + шаг · τ · частота).
        self.inh_gain.sub_(1.0).mul_(self.decay_inh_gain).add_(1.0)
        self.inh_gain.add_(cfg.inh_gain_per_spike * self.spikes * self.inh_adapts)
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
        self.counts.add_(self.spikes)
        self.seen.add_(self.spikes)
        self.kick_left.sub_(1.0).clamp_(min=0.0)

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
            self.compact and _kernels.deliver is not None
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

            if use_graph:
                self._graphs[key].replay()
            else:
                self._chunk(self.plasticity)


            if self.on_chunk is not None and time.perf_counter() - self._watched_at >= self.watch_seconds:
                self._watched_at = time.perf_counter()
                fired = (self.seen[: self.n] > 0).nonzero().squeeze(1).cpu()
                self.seen.zero_()
                self.on_chunk(fired, self.time_ms + (c + 1) * self.CHUNK * self.cfg.dt_ms)

        self.time_ms += chunks * self.CHUNK * self.cfg.dt_ms

        # Свернуть масштаб eligibility обратно в сами следы.
        self.eligibility.mul_(self.eligibility_scale)
        self.eligibility_scale.fill_(1.0)

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

            if self.cfg.event_overflow == "error":
                raise RuntimeError(message + " — cfg.event_overflow='warn' для диагностики")

            print("ВНИМАНИЕ:", message, flush=True)

    def observe(
        self,
        duration_ms: float,
        sensory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Прожить окно и вернуть частоты (Hz) всех нейронов."""

        self.counts.zero_()
        self.run(duration_ms, sensory)

        steps = max(1, round(duration_ms / self.cfg.dt_ms / self.CHUNK)) * self.CHUNK

        return self.counts[: self.n] * (1000.0 / (steps * self.cfg.dt_ms))

    # ==============================================================
    # ПЛАСТИЧНОСТЬ
    # ==============================================================

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
                simple = proj.modulator != "climbing"
                unit = proj.cap / cfg.learned_cap_psp if proj.data_weight else 1.0
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
            if fused and proj.modulator != "climbing":
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

            unit = proj.cap / cfg.learned_cap_psp if proj.data_weight else 1.0
            before = proj.weight.clone()
            proj.weight.add_(proj.eligibility.float() * (unit * factor))
            proj.weight.clamp_(0.0, proj.cap)

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

    # ==============================================================
    # READ-ONLY РЕЖИМ
    # ==============================================================

    # Динамическое состояние: буферы, которые CUDA Graph
    # читает и пишет по фиксированным адресам. Поэтому их
    # всегда восстанавливают через copy_, а не присваиванием.
    _DYNAMIC = (
        "v", "i_syn", "i_inh", "i_ngf", "ngf_mf", "i_nmda", "nmda_mf", "i_dend", "adapt", "refractory", "spikes",
        "trace_pre", "trace_pre_rise", "ltp_kernel", "trace_post", "counts", "eligibility",
        "eligibility_scale", "dropped", "bias", "t_h", "kick_current", "kick_left", "stp_x", "stp_u4", "stp_x4", "inh_gain",
        "chemistry", "tick", "mf_rate",
    )

    def _snapshot(self) -> dict:
        return {name: getattr(self, name).clone() for name in self._DYNAMIC}

    def _restore(self, saved: dict) -> None:
        for name, value in saved.items():
            target = getattr(self, name)

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
        self.adapt[cells] = 0.0
        self.v[cells] = self.cfg.v_rest

    # ==============================================================
    # СОХРАНЕНИЕ
    # ==============================================================

    def state_dict(self) -> dict:
        return {
            "dynamic": self._snapshot(),
            "time_ms": self.time_ms,
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

        synapses = state["synapses"]

        # copy_ в те же тензоры: view проекций остаются валидными.
        for name in ("net_pre", "net_weight", "ear_pre"):
            getattr(self, name).copy_(synapses[name])

        self.birth_weight.copy_(synapses["birth_weight"])
        self.net_branch.copy_(synapses["net_branch"])
        self.early.copy_(synapses["early"])

        for proj in self.plastic:
            proj.innate_mean = synapses["innate_mean"][proj.name]

        # Прежние таблицы (по проводке при рождении) — освободить до
        # перестройки: иначе обе живут разом, +1.9 ГБ видеопамяти сверх мозга.
        self.out_all = self.out_plastic = self.in_plastic = None
        self._graphs = {}

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

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
