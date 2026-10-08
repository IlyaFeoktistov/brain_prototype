from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math

import torch


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
        self.eligibility = (
            torch.zeros_like(weight) if plastic else None
        )


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

        for spec in cfg.areas:
            if spec.adapt_b is not None:
                self.adapt_b[self.areas[spec.name].E.slice] = spec.adapt_b
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
        self.safe_delta_t = self.delta_t.clamp(min=1e-3)

        # Скачок синаптического тока при весе J даёт
        # интеграл PSP ≈ J мВ·τm независимо от τsyn.
        self.syn_scale = self.tau_m / self.tau_syn
        # Быстрый ток рабочей памяти ослаблен: она должна держать
        # прошлые слова, а не дублировать текущее.
        for area in self.areas.values():
            if area.kind == "memory":
                self.syn_scale[area.E.slice] *= cfg.memory_ampa_ratio

        dt = cfg.dt_ms
        self.decay_m = dt / self.tau_m
        self.decay_syn = torch.exp(-dt / self.tau_syn)
        self.decay_slow = math.exp(-dt / cfg.memory_tau_syn)
        self.slow_scale = self.slow_gain * self.tau_m / cfg.memory_tau_syn
        # Затухание адаптации: у ядер может быть своё (медленное) время.
        self.decay_adapt = torch.full(
            (self.n_slots,), math.exp(-dt / cfg.e_adapt_tau), device=self.device
        )

        for spec in cfg.areas:
            if spec.adapt_tau is not None:
                E = self.areas[spec.name].E
                self.decay_adapt[E.slice] = math.exp(-dt / spec.adapt_tau)
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
        return low + (high - low) * torch.rand(
            shape, device=self.device, generator=self.generator
        )

    def _connect(
        self,
        name: str,
        pre_pop: Population,
        post: Population,
        k: int,
        weight: tuple[float, float],
        plastic: bool = False,
        cap: float = 0.0,
    ) -> Projection | None:
        k = min(k, pre_pop.size)

        if k <= 0:
            return None

        proj = Projection(
            name,
            "net",
            self._random_pre(post, k, pre_pop),
            post,
            self._uniform((post.size, k), *weight),
            plastic=plastic,
            cap=cap,
            innate_max=weight[1],
        )
        proj.pre_pop = pre_pop
        proj.innate_range = weight
        proj.source_kind = next(
            a.kind for a in self.areas.values() if pre_pop in (a.E, a.I)
        )
        self.projections.append(proj)

        return proj

    def _init_projections(self) -> None:
        cfg = self.cfg
        self.projections: list[Projection] = []

        # ---- локальные микросхемы зон ----
        for spec in cfg.areas:
            area = self.areas[spec.name]

            self._connect(
                f"{spec.name}: E->E", area.E, area.E, spec.k_ee,
                spec.ee_weight, plastic=spec.ee_plastic, cap=spec.ee_cap,
            )
            self._connect(
                f"{spec.name}: E->I", area.E, area.I, cfg.k_ei,
                (cfg.ei_weight, cfg.ei_weight),
            )
            self._connect(
                f"{spec.name}: I->E", area.I, area.E, cfg.k_ie,
                (cfg.ie_weight, cfg.ie_weight),
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

        net = sorted(
            (p for p in self.projections if p.source == "net"),
            key=lambda p: not p.plastic,
        )
        sensory = [p for p in self.projections if p.source == "sensory"]
        self.projections = net + sensory
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
        self.ear_pre, self.ear_post, self.ear_weight = flatten(sensory, dummy=False)

        self.n_synapses = self.net_weight.numel() - self.DUMMY_SYNAPSES
        self.n_plastic = sum(p.weight.numel() for p in self.plastic)

        # Хвост — фиктивные ячейки для пустых мест таблиц.
        self.eligibility = torch.zeros(
            self.n_plastic + self.DUMMY_SYNAPSES, device=self.device
        )

        # Вес каждого пластичного синапса при рождении: выученное —
        # это отклонение от него, а не от общего врождённого максимума.
        self.birth_weight = self.net_weight[: self.n_plastic].clone()

        # Свежая (ранняя, early-LTP/LTD) часть выученного: входит в вес,
        # но затухает к закреплённому весу, пока сон её не закрепит.
        self.early = torch.zeros(self.n_plastic, device=self.device)

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

        self._build_tables()

    def _build_tables(self) -> None:
        P, S = self.n_plastic, self.n_synapses
        self.out_all = self._table(self.net_pre[:S], S)
        self.out_plastic = self._table(self.net_pre[:P], P)
        self.in_plastic = self._table(self.net_post[:P], P)
        # Таблицы — новые тензоры: старые CUDA Graph на них не смотрят.
        self._graphs = {}

    def _table(self, keys: torch.Tensor, pad: int) -> torch.Tensor:
        """
        Синапсы, сгруппированные по нейрону: [n_slots, max_degree].
        Пустые ячейки -> pad + (разные фиктивные синапсы).
        """

        keys = keys.long()
        ids = torch.arange(len(keys), device=self.device)
        order = torch.argsort(keys, stable=True)
        keys, ids = keys[order], ids[order]

        counts = torch.bincount(keys, minlength=self.n_slots)
        starts = torch.cumsum(counts, 0) - counts
        position = torch.arange(len(keys), device=self.device) - starts[keys]

        width = max(1, int(counts.max())) if len(keys) else 1
        table = (
            pad
            + torch.arange(self.n_slots * width, device=self.device)
            .view(self.n_slots, width) % self.DUMMY_SYNAPSES
        ).to(torch.int32)
        table[keys, position] = ids.to(torch.int32)

        return table

    def _init_state(self) -> None:
        cfg = self.cfg
        z = lambda: torch.zeros(self.n_slots, device=self.device)

        self.v = cfg.v_rest + self._uniform((self.n_slots,), 0.0, 10.0)
        self.i_syn = z()
        self.i_slow = z()
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
        # Ток покоя (мВ): гомеостаз возбудимости зоны.
        self.bias = z()

        for area in self.areas.values():
            self.bias[area.I.slice] = cfg.i_excitability

        # Сигнал «изнутри тела» (мВ): чувство возбуждает свой набор
        # нейронов лимбической зоны — как ухо возбуждает слуховую кору.
        self.interoception = z()
        # Истинная eligibility = eligibility * eligibility_scale:
        # затухание всех следов — одно умножение скаляра.
        self.eligibility_scale = torch.ones(1, device=self.device)
        # Спайки, не влезшие в таблицу шага (должно быть 0).
        self.dropped = torch.zeros(1, device=self.device)
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

        synapses = self.out_all.index_select(0, fired_before).flatten()
        drive.index_add_(
            0,
            self.net_post.index_select(0, synapses),
            self.net_weight.index_select(0, synapses),
        )

        heard = heard_now.index_select(0, self.ear_pre)
        drive.index_add_(0, self.ear_post, heard * self.ear_weight)

        if plastic:
            # LTD: пре пришёл после поста. Нейроны рабочей памяти
            # (ковариационное правило) этой ветки не имеют.
            rows = self.out_plastic.index_select(0, fired_before)
            synapses = rows.flatten()
            post = self.net_post.index_select(0, synapses).view_as(rows)
            ltd = self.trace_post.index_select(0, post.flatten()).view_as(rows)
            ltd = ltd * self.ltd_mask.index_select(0, fired_before)[:, None]
            self.eligibility.index_add_(
                0,
                synapses,
                ltd.flatten() * (-cfg.stdp_a_minus / self.eligibility_scale),
            )

        self.i_syn.mul_(self.decay_syn).add_(drive * self.syn_scale)
        self.i_slow.mul_(self.decay_slow).add_(drive.clamp(min=0.0) * self.slow_scale)

        # ---- мембрана (AdEx / LIF) ----
        exp_term = torch.where(
            self.is_adex,
            self.delta_t
            * torch.exp(
                ((self.v - cfg.v_threshold) / self.safe_delta_t).clamp(max=10.0)
            ),
            0.0,
        )

        dv = (
            cfg.v_rest - self.v + exp_term - self.adapt
            + self.i_syn + self.i_slow + self.bias + self.interoception
        ) * self.decay_m
        v = torch.where(self.refractory <= 0, self.v + dv, self.v)
        self.refractory.sub_(cfg.dt_ms)

        fired = v >= self.v_peak
        self.spikes.copy_(fired)

        self.v.copy_(torch.where(fired, cfg.v_reset, v))
        self.refractory.copy_(torch.where(fired, cfg.refractory_ms, self.refractory))
        self.adapt.mul_(self.decay_adapt).add_(self.adapt_b * self.spikes)
        self.counts.add_(self.spikes)
        self.dropped.add_((self.spikes.sum() - self.max_spikes).clamp(min=0.0))

        # ---- STDP -> eligibility ----
        if plastic:
            # LTP: пост выстрелил после пре.
            fired_now = self._fired()
            synapses = self.in_plastic.index_select(0, fired_now).flatten()
            pre = self.net_pre.index_select(0, synapses)
            self.eligibility.index_add_(
                0,
                synapses,
                self.ltp_kernel.index_select(0, pre)
                * (cfg.stdp_a_plus / self.eligibility_scale),
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
        for t in range(self.CHUNK):
            self._step(self.ear_input[t], plastic)

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

        with torch.cuda.graph(graph):
            self._chunk(plastic)

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
            tag = eligibility.abs()
            tag = tag / tag.max().clamp(min=1e-12)
            self.early.mul_(1.0 - capture * tag)

    def forget(self) -> None:
        """
        Свежая память затухает: ранняя часть изменений уходит к
        закреплённому весу с полураспадом early_ltp_half_life фраз.
        Закреплённое сном (capture) не затухает.
        """

        half_life = self.cfg.early_ltp_half_life

        if not self.plasticity or not self.cfg.forgetting_enabled or half_life <= 0:
            return

        fade = 1.0 - 0.5 ** (1.0 / half_life)
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
            proj.weight[weak] = low + (high - low) * torch.rand(
                count, device=self.device, generator=self.generator
            )
            proj.birth[weak] = proj.weight[weak]
            proj.early[weak] = 0.0
            proj.eligibility[weak] = 0.0
            grown += count

        if grown:
            self._build_tables()
            self.synapses_grown += grown

        return grown

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
            step = (self.cfg.homeostasis_rate * error).clamp(-1.0, 1.0)
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

        for name, pre, post in links:
            proj = next((p for p in self.projections if p.name == name), None)

            if proj is None or not len(pre) or not len(post):
                continue

            low = proj.innate_range[0] if proj.plastic else 0.0
            start = proj.pre_pop.start
            pick = torch.randint(0, len(post), (count,), device=self.device, generator=self.generator)
            rows = post[pick]
            which = torch.randint(0, len(pre), (count,), device=self.device, generator=self.generator)
            new_pre = pre[which] + start
            # Уже есть такая связь — не дублировать.
            fresh = ~(proj.pre[rows] == new_pre[:, None].to(proj.pre.dtype)).any(1)
            rows, new_pre = rows[fresh], new_pre[fresh]

            if not len(rows):
                continue

            cols = proj.weight[rows].argmin(1)
            proj.pre[rows, cols] = new_pre.to(proj.pre.dtype)
            proj.weight[rows, cols] = low

            if proj.plastic:
                proj.birth[rows, cols] = low
                proj.early[rows, cols] = 0.0
                proj.eligibility[rows, cols] = 0.0

            grown += len(rows)

        if grown:
            self._build_tables()
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
        "v", "i_syn", "i_slow", "adapt", "refractory", "spikes",
        "trace_pre", "trace_pre_rise", "ltp_kernel", "trace_post", "counts", "eligibility",
        "eligibility_scale", "dropped", "bias", "interoception",
    )

    def _snapshot(self) -> dict:
        return {name: getattr(self, name).clone() for name in self._DYNAMIC}

    def _restore(self, saved: dict) -> None:
        for name, value in saved.items():
            getattr(self, name).copy_(value)

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
            "synapses_grown": self.synapses_grown,
            "synapses": {
                "net_pre": self.net_pre,
                "net_weight": self.net_weight,
                "ear_pre": self.ear_pre,
                "birth_weight": self.birth_weight,
                "early": self.early,
                "innate_mean": {p.name: p.innate_mean for p in self.plastic},
            },
            "generator": self.generator.get_state(),
        }

    def load_state_dict(self, state: dict) -> None:
        self._restore(state["dynamic"])
        self.time_ms = state["time_ms"]
        self.synapses_grown = state.get("synapses_grown", 0)

        synapses = state["synapses"]

        # copy_ в те же тензоры: view проекций остаются валидными.
        for name in ("net_pre", "net_weight", "ear_pre"):
            getattr(self, name).copy_(synapses[name])

        if "birth_weight" in synapses:
            self.birth_weight.copy_(synapses["birth_weight"])

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
        return self.n_synapses + self.ear_weight.numel()

    def memory_bytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for t in (
                self.net_pre, self.net_post, self.net_weight,
                self.ear_pre, self.ear_post, self.ear_weight,
                self.eligibility, self.early, self.out_all, self.out_plastic, self.in_plastic,
            )
        )
