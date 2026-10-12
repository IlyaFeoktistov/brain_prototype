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
    """Зона мозга: свои E и I нейроны, своё торможение."""

    name: str
    title: str
    kind: str
    E: Population
    I: Population


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
            cfg.sensory_size, device=self.device
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

            E = Population(f"{spec.name}.E", start, n_e)
            I = Population(f"{spec.name}.I", E.stop, n_i)
            start = I.stop

            self.areas[spec.name] = Area(spec.name, spec.title, spec.kind, E, I)

        self.n = start
        # Фиктивные нейроны [n, n_slots): никогда не стреляют, на них
        # указывают пустые ячейки таблиц синапсов. Их много, чтобы
        # пустые ячейки не дрались за один адрес (atomic-конфликты).
        self.n_slots = self.n + self.DUMMY_NEURONS

    def _fill(self, values: dict[str, float]) -> torch.Tensor:
        """Параметр нейрона по типу популяции: cortex_E, memory_E, I."""

        out = torch.zeros(self.n_slots, device=self.device)

        for area in self.areas.values():
            out[area.E.slice] = values[f"{area.kind}_E"]
            out[area.I.slice] = values["I"]

        return out

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
        self.decay_eligibility = math.exp(-dt / cfg.eligibility_tau)
        # Число фоновых входных спайков за шаг ~ Poisson(λ).
        self.noise_mean = cfg.background_rate_hz * dt / 1000.0
        self.noise_std = math.sqrt(self.noise_mean)
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
    ) -> Projection | None:
        k = min(k, pre_pop.size)

        if k <= 0:
            return None

        # Врождённый путь с постоянным весом — не храним, а вычисляем
        # (топографический хранится: хеш получателей не знает карты).
        if self.cfg.procedural_wiring and not plastic and weight[0] == weight[1] != 0 and not topographic:
            proj = ProceduralProjection(
                name, pre_pop, post, k, weight[0],
                seed=self.cfg.seed * 1009 + len(self.projections), device=self.device,
            )
            proj.source_kind = next(
                a.kind for a in self.areas.values() if pre_pop in (a.E, a.I)
            )
            self.projections.append(proj)
            return proj

        pre = (
            self._topographic_pre(post, k, pre_pop, topographic) if topographic > 0
            else self._random_pre(post, k, pre_pop)
        )
        proj = Projection(
            name,
            "net",
            pre,
            post,
            self._uniform((post.size, k), *weight),
            plastic=plastic,
            cap=cap,
            innate_max=weight[1],
        )
        proj.pre_pop = pre_pop
        proj.innate_range = weight
        proj.modulator = modulator
        proj.ach_suppressed = ach_suppressed
        proj.source_kind = next(
            a.kind for a in self.areas.values() if pre_pop in (a.E, a.I)
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

        # ---- локальные микросхемы зон ----
        for spec in cfg.areas:
            area = self.areas[spec.name]

            self._connect(
                f"{spec.name}: E->E", area.E, area.E, spec.k_ee,
                spec.ee_weight, plastic=spec.ee_plastic, cap=spec.ee_cap,
                modulator=spec.ee_modulator, ach_suppressed=spec.ee_ach_suppressed,
            )
            if spec.columns > 1:
                self._wire_columns(spec, area)

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

            proj = self._connect(
                f"{path.src} -> {path.dst}", src.E, dst.E, path.k,
                path.weight, plastic=path.plastic, cap=path.cap,
                modulator=path.modulator, ach_suppressed=path.ach_suppressed,
                topographic=path.topographic,
            )

            if proj is not None:
                proj.predictive = path.predictive
            self._connect(
                f"{path.src} -> {path.dst}.I", src.E, dst.I, path.k_inhibitory,
                (path.inhibitory_weight, path.inhibitory_weight),
            )

        # ---- ухо: sensory -> слуховая кора ----
        ear = self.areas["auditory"]
        k = min(cfg.k_sensory, cfg.sensory_size)

        for pop, gain in ((ear.E, 1.0), (ear.I, cfg.sensory_inhibitory_gain)):
            w = cfg.sensory_weight * gain
            self.projections.append(
                Projection(
                    f"ear -> {pop.name}",
                    "sensory",
                    torch.randint(
                        0, cfg.sensory_size, (pop.size, k),
                        device=self.device, generator=self.generator,
                    ),
                    pop,
                    torch.full((pop.size, k), w, device=self.device),
                )
            )

        self._pack()

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

        # Пути гиппокампа учатся от ацетилхолина (encode), а не от дофамина.
        # Пути в стриатум D2 («стой») — от провала дофамина: знак −1
        # (D2-рецептор: LTP при паузе дофамина, Frank 2005; Shen 2008).
        self.encoding = [p for p in self.plastic if p.modulator == "acetylcholine"]
        inverted = [p for p in self.plastic if p.modulator == "dopamine_d2"]
        # Пути из зон, стреляющих непрерывно: след STDP там ни о чём не
        # говорит (LTP ≈ LTD), а во сне раздувал общее слово — учит их
        # совпадение (language._learn_context).
        coincidence = [p for p in self.plastic if p.modulator == "coincidence"]
        self.dopamine_mask = None

        if self.encoding or inverted or coincidence:
            self.dopamine_mask = torch.ones(self.n_plastic, dtype=torch.float16, device=self.device)

            for p in self.encoding + coincidence:
                self.dopamine_mask[p.offset: p.offset + p.weight.numel()] = 0.0

            for p in inverted:
                self.dopamine_mask[p.offset: p.offset + p.weight.numel()] = -1.0

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
        self.i_syn = z()
        self.i_slow = z()
        # Дендритный ток (контекст из рабочей памяти, см. dendritic_context).
        self.i_dend = torch.zeros(self.n_slots * max(1, cfg.dendrite_branches), device=self.device)
        # Пул торможения смысловой коры: отфильтрованное число её спайков.
        self.pool = torch.zeros(1, device=self.device)
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
        # Ток покоя (мВ): гомеостаз возбудимости зоны.
        self.bias = z()

        for area in self.areas.values():
            self.bias[area.I.slice] = cfg.i_excitability

        # Развитие до рождения (калибровка): гомеостаз быстрый везде.
        self.developing = False
        # Ацетилхолин (тензор — его читает записанный шаг CUDA Graph).
        self.acetylcholine = torch.zeros(1, device=self.device)
        # Сигнал «изнутри тела» (мВ): чувство возбуждает свой набор
        # нейронов лимбической зоны — как ухо возбуждает слуховую кору.
        self.interoception = z()
        # Истинная eligibility = eligibility * eligibility_scale:
        # затухание всех следов — одно умножение скаляра.
        self.eligibility_scale = torch.ones(1, device=self.device)
        # Спайки, не влезшие в таблицу шага (должно быть 0).
        self.dropped = torch.zeros(1, device=self.device)
        self._init_chemistry()
        # Базовый контур (base_circuit): ресурс пресинапса (истощение,
        # 1 — полный) и усиление торможения, приходящего к клетке (1 — норма).
        self.stp_x = torch.ones(self.n_slots, device=self.device)
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


        # Вход уха на CHUNK шагов вперёд.
        self.ear_input = torch.zeros(
            self.CHUNK, cfg.sensory_size, device=self.device
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
            branches = max(1, cfg.dendrite_branches)
            slot = post.long() * branches + self.net_branch.index_select(0, synapses).long()
            dendrite = torch.zeros_like(self.i_dend).index_add_(0, slot, weight * to_dendrite)
            self.i_dend.mul_(self.decay_dend).add_(dendrite)
            weight = weight * ~to_dendrite

        if cfg.base_circuit:
            # Истощение: синапс отдаёт долю оставшегося ресурса пресинапса.
            weight = weight * self.stp_x.index_select(0, pre_of)
            # Торможение к клетке × её усиление (растёт, если клетка
            # стреляет выше нормы).
            weight = torch.where(weight < 0, weight * self.inh_gain.index_select(0, post), weight)

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

        if cfg.dendritic_context:
            # Плато на дендрите усиливает возбуждающий вход сомы:
            # контекст выбирает среди того, что предлагает текущее слово.
            if cfg.dendrite_branches > 1:
                # Плато — на самой горячей ветке, если её ток выше порога.
                branch = self.i_dend.view(self.n_slots, cfg.dendrite_branches)
                plateau = torch.sigmoid(
                    (branch - cfg.dendrite_threshold_mv) / cfg.dendrite_slope_mv
                ).amax(1)
                # Плато ветки деполяризует и сому (см. dendrite_soma_mv).
                somatic = cfg.dendrite_soma_mv * plateau
            else:
                plateau = self.i_dend / (self.i_dend + cfg.dendrite_half_mv)
                somatic = 0.0
            synaptic = synaptic + cfg.dendrite_gain * plateau * synaptic.clamp(min=0.0) + somatic

        if cfg.pool_inhibition_mv > 0:
            synaptic = synaptic - cfg.pool_inhibition_mv * self.pool * self.pool_mask

        # Вещества через рецепторы: занятость c / (c + K) × ток клетки.
        occupancy = (self.chemistry / (self.chemistry + self.chemistry_half)).to(torch.float32)
        modulated = occupancy @ self.receptor_mv

        dv = (
            cfg.v_rest - self.v + exp_term - self.adapt - self.adapt_slow
            + synaptic + self.bias + self.interoception + modulated + self.clock_mv
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
        self.adapt_slow.mul_(self.decay_slow_ahp).add_(self.slow_ahp_b * self.spikes)

        if cfg.base_circuit:
            # Ресурс восстанавливается к 1, спайк тратит долю stp_u.
            self.stp_x.add_((1.0 - self.stp_x) * self.stp_recover)
            self.stp_x.sub_(cfg.stp_u * self.stp_x * self.spikes * self.cortex_e)
            # Усиление торможения: каждый спайк E коры поднимает, медленно
            # возвращается к 1 (в равновесии 1 + шаг · τ · частота).
            self.inh_gain.sub_(1.0).mul_(self.decay_inh_gain).add_(1.0)
            self.inh_gain.add_(cfg.inh_gain_per_spike * self.spikes * self.inh_adapts)

        if cfg.pool_inhibition_mv > 0:
            self.pool.mul_(self.decay_pool).add_((self.spikes * self.pool_mask).sum())
        self.counts.add_(self.spikes)
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
                part = sensory[begin: begin + self.CHUNK]
                self.ear_input[: len(part)].copy_(part)
                self.ear_input[len(part):].zero_()
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
            proj.weight.clamp_(0.0, proj.cap)

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

            E = self.areas[spec.name].E
            active = (rates[E.slice] > 0).float().mean()
            error = (spec.target_activity - active) / spec.target_activity
            # До рождения (калибровка) — критический период: подстройка
            # быстрая во всех зонах; потом у зоны своя скорость.
            speed = 1.0 if self.developing else spec.homeostasis
            step = (self.cfg.homeostasis_rate * speed * error).clamp(-1.0, 1.0)
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
            total = proj.weight.sum(1)
            budget = (
                proj.weight.shape[1]
                * proj.innate_mean
                * self.cfg.synaptic_scaling_budget
            )
            factor = (budget / total.clamp(min=1e-9)).clamp(max=1.0)
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
        "v", "i_syn", "i_slow", "i_dend", "pool", "adapt", "adapt_slow", "refractory", "spikes",
        "trace_pre", "trace_pre_rise", "ltp_kernel", "trace_post", "counts", "eligibility",
        "eligibility_scale", "dropped", "bias", "interoception", "stp_x", "inh_gain",
        "chemistry",
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

    def silence(self, area: str) -> None:
        """
        Погасить устойчивую активность зоны (и её ток CAN): медленная волна
        сна — кора целиком замолкает (down-state).
        """

        if area not in self.areas:
            return

        for pop in (self.areas[area].E, self.areas[area].I):
            self.i_syn[pop.slice] = 0.0
            self.i_slow[pop.slice] = 0.0
            self.adapt[pop.slice] = 0.0
            self.adapt_slow[pop.slice] = 0.0
            self.v[pop.slice] = self.cfg.v_rest

    def quiet(self, area: str) -> None:
        """Погасить активность зоны (только внутри frozen())."""

        for pop in (self.areas[area].E, self.areas[area].I):
            self.i_syn[pop.slice] = 0.0
            self.i_slow[pop.slice] = 0.0
            self.v[pop.slice] = self.cfg.v_rest

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
