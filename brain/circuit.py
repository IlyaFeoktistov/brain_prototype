"""
Зона с типами клеток по готовой модели: CA3 Kopsick et al. 2024 (J Comput
Neurosci, https://doi.org/10.1007/s10827-024-00881-3; сеть — код авторов,
data/anatomy/hippocampus/ca3_kopsick.json, разбор — tools/ca3_kopsick_extract.py,
docs/CA3_TRANSFER.md). 8 типов клеток (пирамидные + 7 тормозных), клетки
Izhikevich 2007 (9 параметров), 51 связь со своими проводимостью, временем
спада и Tsodyks–Markram, фон по типу. Без специфичности типов сеть CA3
неустойчива (Kopsick et al. 2023, https://doi.org/10.1007/s12559-021-09954-2)
— у нас прежний CA3 «E + 10% I» был двустабилен (CHAT_PLAN шаг 3, циклы 3–4).

Зубчатая — та же сеть зоны: 4 типа Dyhrfjeld-Johnsen et al. 2007, без фона
авторов, HIPP — AdEx (data/anatomy/hippocampus/dg_dyhrfjeld.json, tools/dg_extract.py).

Счёт — как в эталоне tools/ca3_kopsick.py (ядро CARLsim6 feat/ca3net),
перенесённый на шаг мозга dt (0.5 мс):
  - спад STP по связи: u ×(1 − dt/τ_u), x += (1 − x)·dt/τ_x;
  - спад проводимости ×(1 − dt/τ_d) — тот же заряд, что ×(1 − 1/τ_d) за 1 мс у
    CARLsim (сумма ряда × шаг = τ_d);
  - задержка 1 мс (delay_ms связей) — round(1/dt) шагов мозга;
  - доставка: u⁺ = u + U(1 − u⁻), x⁺ = x − u⁺x⁻, Δg = вес × g × u⁺x⁻/U;
  - фон lognormal(μ, σ типа) × U(0, 1) пА, потолок, новый каждую 1 мс;
  - мембрана: RK4 подшагами ≤ 0.2 мс, ток = фон − g_AMPA(v − 0) − g_GABAa(v + 70)
    + вход мозга; спайк при v > vpeak → v = c, u += d; рефрактерность 1 мс
    после спайка на последнем подшаге шага (как в ядре CARLsim).

Связи: вероятность авторов × (клеток получателя у авторов / у нас) — у
клетки столько же входов, сколько у авторов (проверено эталоном: 10 000
клеток с числом входов авторов держат тот же покой, 6c25b0b).

Вход мозга (пути DG → CA3, EC → CA3 и их среднее поле, вещества) приходит в
мВ движка — «ток в мВ покоя» LIF-клетки, у которой вес пути = измеренный
ВПСП / пик (engine._psp_peak). Перевод в пА — множителем типа, при котором
пик ВПСП клетки Izhikevich в покое тот же (set_input_gain): вес пути по-прежнему
означает измеренный ВПСП.

Шаг без синхронизаций с процессором (пишется в CUDA Graph): события —
плотным списком постоянной ёмкости (EventTable движка).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

DATA = Path(__file__).resolve().parent.parent / "data" / "anatomy"

# Реверсия AMPA / ГАМК-А у авторов (CARLsim6: 0 и −70 мВ).
E_AMPA, E_GABAA = 0.0, -70.0
# Подшаг RK4 у авторов — 0.2 мс (5 на 1 мс); у нас — не длиннее.
RK4_MAX_MS = 0.2
# Нижняя граница v у CARLsim (−90 мВ).
V_FLOOR = -90.0
# Сглаживание частоты источника недостающих входов (как у недостающих
# таламических, cfg.tc_missing_tau_ms) [допущение].
MISSING_TAU_MS = 50.0


def largest_remainder(total: int, weights: list[float]) -> list[int]:
    """Целые доли total пропорционально weights (сумма ровно total)."""

    s = sum(weights)
    raw = [total * w / s for w in weights]
    out = [int(math.floor(r)) for r in raw]
    for i in sorted(range(len(raw)), key=lambda i: -(raw[i] - out[i]))[: total - sum(out)]:
        out[i] += 1
    return out


def load(name: str) -> dict:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def principal(net: dict) -> str:
    """
    Главный тип сети (главные клетки зоны, area.E): поле «principal» или
    единственный возбуждающий тип (CA3 Kopsick).
    """

    if "principal" in net:
        return net["principal"]

    exc = [g for g, x in net["groups"].items() if x["excitatory"]]

    if len(exc) != 1:
        raise ValueError("сеть зоны: возбуждающих типов несколько — укажите principal")
    return exc[0]


def inhibitory_share(net: dict) -> float:
    """
    Доля клеток сети не главного типа — для AreaSpec.local_inhibitory: они
    стоят на местах area.I (у CA3 — тормозные, у зубчатой ещё и мшистые).
    """

    groups = net["groups"]
    total = sum(g["cells"] for g in groups.values())
    return 1.0 - groups[principal(net)]["cells"] / total


class TypedCircuit:
    """
    Сеть зоны по данным авторов на клетках зоны мозга: главные (area.E) —
    главный тип сети (principal), места area.I — остальные типы по долям
    авторов (тормозные и, у зубчатой, мшистые). Состояние клеток (v, u) —
    своё; мозг видит их спайки и v.
    """

    def __init__(self, net: dict, area, dt: float, device: torch.device, generator: torch.Generator,
                 compact: bool, capacity: float):
        self.area = area
        self.device = device
        self.dt = dt
        groups = net["groups"]
        main = principal(net)
        rest = [g for g in groups if g != main]

        # Клеток по типам: главные — area.E, остальные — area.I по долям авторов.
        sizes = {main: area.E.size}
        sizes.update(zip(rest, largest_remainder(area.I.size, [groups[g]["cells"] for g in rest])))
        self.names = [main] + rest
        self.size = sizes
        offsets = [0]
        for g in self.names[:-1]:
            offsets.append(offsets[-1] + sizes[g])
        self.start = dict(zip(self.names, offsets))
        self.n = n = area.E.size + area.I.size
        self.first = area.E.start

        if area.I.start != area.E.stop:
            raise ValueError("сеть зоны: E и I зоны должны идти подряд")

        def per_cell(key):
            return torch.cat([torch.full((sizes[g],), float(groups[g][key])) for g in self.names]).to(device)

        # Тип с моделью AdEx (Brette & Gerstner 2005; поле model: "adex"):
        # C dv/dt = −gL(v − EL) + gL·ΔT·exp((v − VT)/ΔT) − w + I, τw dw/dt =
        # a(v − EL) − w — то же, что u Izhikevich при a = 1/τw, b = a_AdEx,
        # vr = EL; k = 0 (квадратичного члена нет). У Izhikevich gL = 0.
        for g in self.names:
            x = groups[g]
            if x.get("model") == "adex":
                x.setdefault("k", 0.0)
                x.setdefault("vt", x["VT"])
            else:
                x.setdefault("gL", 0.0)
                x.setdefault("EL", 0.0)
                x.setdefault("VT", 0.0)
                x.setdefault("deltaT", 1.0)
        self.gL, self.EL, self.VT, self.dT = per_cell("gL"), per_cell("EL"), per_cell("VT"), per_cell("deltaT")
        self.C, self.k, self.vr, self.vt = per_cell("C"), per_cell("k"), per_cell("vr"), per_cell("vt")
        self.a, self.b, self.vpeak = per_cell("a"), per_cell("b"), per_cell("vpeak")
        self.c, self.d = per_cell("c"), per_cell("d")
        # Фон авторов (lognormal по типу) — у сети, подобранной на покой без
        # входных зон (CA3 Kopsick); у сети без фона (зубчатая Hippocampome)
        # весь вход — пути мозга.
        self.has_background = all("background_mu" in groups[g] for g in self.names)
        if self.has_background:
            self.mu, self.sigma = per_cell("background_mu"), per_cell("background_sigma")
        self.cap = float(net["background_cap_pa"])
        # Разброс фона вокруг его среднего (диагностика стыка, tools/ca3_margin.py
        # bgvar=): 1 — фон авторов; среднее (E[lognormal × U(0,1)] =
        # e^(μ+σ²/2)/2) не меняется.
        self.bg_scale = 1.0
        if self.has_background:
            self.bg_mean = torch.exp(self.mu + self.sigma ** 2 / 2) / 2
        self.type_of = torch.cat([torch.full((sizes[g],), i, dtype=torch.long) for i, g in enumerate(self.names)]).to(device)
        # Шаги мозга на 1 мс (фон и задержка авторов — по 1 мс).
        self.per_ms = max(1, round(1.0 / dt))
        refractory = per_cell("refractory_ms")
        self.ref_steps = torch.round(refractory / dt).clamp(min=1).to(torch.int32)
        self.substeps = max(1, math.ceil(dt / RK4_MAX_MS - 1e-9))

        # Слоты STP (связь × пресинаптическая клетка) и проводимости (связь ×
        # получатель).
        links = net["links"]
        delays = {int(link["delay_ms"]) for link in links}
        if len(delays) != 1:
            raise ValueError("сеть зоны: задержки связей разные — нужна очередь по задержкам")
        self.delay_steps = max(1, round(delays.pop() / dt))
        stp_pre, stp_U, stp_iu, stp_ix, stp_scale = [], [], [], [], []
        g_cell, g_decay, g_exc = [], [], []
        # Топография (поле bins у сети, axon_profile у типов — зубчатая,
        # Morgan & Soltesz 2008): построенные синапсы — с вероятностью пары у
        # авторов на том же расстоянии по длине зоны (доля общих источников
        # — их), недостающие до числа входов по данным — независимыми
        # событиями своей клетки (_missing_step).
        self.topo = "bins" in net and all("axon_profile" in groups[g] for g in self.names)
        stp_link, g_link = [], []
        stp_off, g_off = [], []
        s_total = g_total = 0

        for link in links:
            pre, post = link["pre"], link["post"]
            npre, npost = sizes[pre], sizes[post]
            stp_off.append(s_total)
            g_off.append(g_total)
            stp_pre.append(torch.arange(npre) + self.start[pre])
            stp_U.append(torch.full((npre,), link["U"]))
            stp_iu.append(torch.full((npre,), dt / link["tau_u_ms"]))
            stp_ix.append(torch.full((npre,), dt / link["tau_x_ms"]))
            # Входов у клетки по данным больше, чем клеток-источников у нас:
            # связь со всеми, сила × недостающая доля (средний вход тот же)
            # [своё: при малой популяции-источнике].
            factor = 1.0 if self.topo else max(1.0, self._p(link, groups, sizes))
            stp_scale.append(torch.full((npre,), link["weight"] * link["g"] / link["U"] * factor))
            stp_link.append(torch.full((npre,), len(stp_link), dtype=torch.long))
            g_link.append(torch.full((npost,), len(g_link), dtype=torch.long))
            g_cell.append(torch.arange(npost) + self.start[post])
            g_decay.append(torch.full((npost,), 1.0 - dt / link["tau_d_ms"]))
            g_exc.append(torch.full((npost,), bool(groups[pre]["excitatory"])))
            s_total += npre
            g_total += npost

        # Фиктивные ячейки в конце: на них указывают пустые места списка событий.
        self.dummies = 1024

        def cat(xs, dtype=torch.float32, pad=0.0):
            return torch.cat(xs + [torch.full((self.dummies,), pad)]).to(device=device, dtype=dtype)

        self.stp_pre = cat(stp_pre, torch.long, n)
        self.stp_U = cat(stp_U, pad=1.0)
        self.stp_iu = cat(stp_iu)
        self.stp_ix = cat(stp_ix)
        self.stp_scale = cat(stp_scale)
        self.n_stp = s_total
        self.g_cell = cat(g_cell, torch.long, n)
        self.g_decay = cat(g_decay)
        self.g_exc = cat(g_exc, torch.bool, False)
        self.n_g = g_total
        self.stp_link = cat(stp_link, torch.long, len(links))
        self.g_link = torch.cat(g_link).to(device)
        self.link_pre = torch.tensor([self.names.index(link["pre"]) for link in links], device=device)
        self.link_npre = torch.tensor([float(sizes[link["pre"]]) for link in links], device=device)
        self.link_scale = torch.tensor([link["weight"] * link["g"] / link["U"] for link in links], device=device)
        self.n_links = len(links)
        self.type_sizes = torch.tensor([float(sizes[g]) for g in self.names], device=device)
        # Недостающих входов на слот проводимости (связь × получатель).
        missing = torch.zeros(g_total, device=device)

        if self.topo:
            bins = int(net["bins"])
            bin_of = {g: (torch.arange(sizes[g], device=device) * bins // sizes[g]) for g in self.names}
            profile = {g: torch.tensor(groups[g]["axon_profile"], device=device) for g in self.names}
            centre = (len(profile[self.names[0]]) - 1) // 2
            offsets = torch.arange(bins, device=device)
            # S[b] — сумма профиля по всем местам источника для получателя в b.
            reach = {g: profile[g][(offsets[None, :] - offsets[:, None]) + centre].sum(1) for g in self.names}
            self.link_A = []
            for i, link in enumerate(links):
                pre = link["pre"]
                conv = link["p"] * groups[pre]["cells"]
                # Вероятность пары у авторов: A × профиль(смещение); A — чтобы у
                # клетки было conv входов (их клеток источника на место × S).
                self.link_A.append(conv / (groups[pre]["cells"] / bins * float(reach[pre].mean())))
            self._bins, self._bin_of, self._profile, self._centre, self._reach = bins, bin_of, profile, centre, reach

        # Синапсы: случайно с вероятностью авторов × (их клеток получателя /
        # наших) — у клетки число входов авторов; по пресинаптической клетке.
        keys, syn_g, syn_s = [], [], []
        by_pre: dict[str, list[int]] = {}
        for i, link in enumerate(links):
            by_pre.setdefault(link["pre"], []).append(i)

        for g in self.names:
            npre = sizes[g]
            mine = by_pre.get(g, [])
            width = sum(sizes[links[i]["post"]] for i in mine) or 1
            chunk = max(1, 40_000_000 // width)

            for lo in range(0, npre, chunk):
                hi = min(npre, lo + chunk)

                for i in mine:
                    link = links[i]
                    npost = sizes[link["post"]]

                    if self.topo:
                        shift = self._bin_of[g][lo:hi, None] - self._bin_of[link["post"]][None, :] + self._centre
                        p = (self.link_A[i] * self._profile[g][shift]).clamp(max=1.0)
                    else:
                        p = min(1.0, self._p(link, groups, sizes))
                    mask = torch.rand((hi - lo, npost), generator=generator, device=device) < p
                    pre_l, post_l = mask.nonzero(as_tuple=True)
                    del mask
                    pre_l = pre_l + lo
                    keys.append(pre_l + self.start[g])
                    if self.topo:
                        missing.index_add_(0, post_l + g_off[i], torch.full_like(post_l, -1, dtype=torch.float32))
                    syn_g.append((post_l + g_off[i]).int())
                    syn_s.append((pre_l + stp_off[i]).int())

        if self.topo:
            # Недостающие: входов у клетки по данным (у края зоны — меньше, как у
            # авторов: conv × S[место] / среднее S) минус построенные.
            for i, link in enumerate(links):
                pre, post = link["pre"], link["post"]
                conv = link["p"] * groups[pre]["cells"]
                target = conv * self._reach[pre][self._bin_of[post]] / self._reach[pre].mean()
                missing[g_off[i]: g_off[i] + sizes[post]] += target
            self.g_missing = missing.clamp(min=0.0)
        else:
            self.g_missing = missing
        # Частота типа (спайков на клетку за шаг), сглажена, — для недостающих.
        self.rate_decay = 1.0 - dt / MISSING_TAU_MS

        keys = torch.cat(keys)
        self.synapses = len(keys)
        pad_g = torch.arange(self.dummies, device=device, dtype=torch.int32) + g_total
        pad_s = torch.arange(self.dummies, device=device, dtype=torch.int32) + s_total
        self.syn_g = torch.cat(syn_g + [pad_g]).long()
        self.syn_s = torch.cat(syn_s + [pad_s]).long()

        from brain.engine import EventTable

        # Строки таблицы — клетки зоны (n — фиктивная, пустое место списка).
        self.table = EventTable(keys, n + 1, self.synapses, self.dummies)
        del keys
        self.compact = compact

        if compact:
            # Ёмкость списка событий шага — доля синапсов, как у мозга
            # (cfg.event_capacity).
            self.table.prepare_compact(max(32768, int(self.synapses * capacity)), n, 1)
        # Выстреливших за шаг — не больше (остальные в dropped); при покое
        # CA3 ~10 спайков на шаг.
        self.max_fired = min(n, 4096)
        self.reset_state()
        # мВ входа мозга → пА (set_input_gain; до вызова — 1) и вход мозга
        # при частотах покоя источников, мВ (set_rest_input; до вызова — 0).
        self.gain = torch.ones(n, device=device)
        self.rest_mv = torch.zeros(n, device=device)

    @staticmethod
    def _p(link: dict, groups: dict, sizes: dict) -> float:
        """
        Вероятность связи у нас: авторов × (их клеток получателя / наших) —
        у клетки столько же входов, сколько у авторов (их вероятность × их
        клеток источника) при тех же долях типов. Больше 1 — входов по
        данным больше, чем у нас клеток источника.
        """

        return link["p"] * groups[link["post"]]["cells"] / sizes[link["post"]]

    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        n, dev = self.n, self.device
        self.v = self.vr.clone()
        self.u = torch.zeros(n, device=dev)
        self.ref = torch.zeros(n, dtype=torch.int32, device=dev)
        self.su = torch.zeros(self.n_stp + self.dummies, device=dev)
        self.sx = torch.ones(self.n_stp + self.dummies, device=dev)
        self.G = torch.zeros(self.n_g + self.dummies, device=dev)
        # Очередь спайков на задержку: спайк шага s доставляется на шаге
        # s + delay_steps; движок отдаёт спайки прошлого шага — в очереди
        # delay_steps − 1 шагов (у CARLsim спайк 1-й мс доставляется во 2-й).
        self.queue = torch.zeros(max(1, self.delay_steps - 1), n, dtype=torch.bool, device=dev)
        self.ext = torch.zeros(n, device=dev)
        self.tick = torch.zeros(1, dtype=torch.long, device=dev)
        self.dropped = torch.zeros(1, device=dev)
        self.type_rate = torch.zeros(len(self.names), device=dev)

    STATE = ("v", "u", "ref", "su", "sx", "G", "queue", "ext", "tick", "type_rate")

    def state_dict(self) -> dict:
        return {k: getattr(self, k).clone() for k in self.STATE}

    def load_state_dict(self, state: dict) -> None:
        for k in self.STATE:
            getattr(self, k).copy_(state[k])

    # ------------------------------------------------------------------

    def _missing_step(self, arrived: torch.Tensor, u: torch.Tensor, x: torch.Tensor) -> None:
        """
        Недостающие входы (топография): у каждого слота — свой пуассон с
        частотой популяции-источника (сглажена MISSING_TAU_MS), прирост g —
        средняя по источникам связи сила спайка сейчас (Tsodyks–Markram:
        (u + U(1 − u))·x). Независимые события — без лишнего общего входа
        (так же недостающие таламические, van Albada, Helias & Diesmann 2015,
        https://doi.org/10.1371/journal.pcbi.1004490).
        """

        dev = self.device
        counts = torch.zeros(len(self.names), device=dev).index_add_(0, self.type_of, arrived.float())
        self.type_rate.mul_(self.rate_decay).add_(counts / self.type_sizes * (1.0 - self.rate_decay))
        now = (u + self.stp_U * (1.0 - u)) * x
        eff = torch.zeros(self.n_links + 1, device=dev).index_add_(0, self.stp_link, now)[: self.n_links]
        eff = eff / self.link_npre * self.link_scale
        lam = self.g_missing * self.type_rate.index_select(0, self.link_pre).index_select(0, self.g_link)
        self.G[: self.n_g] += torch.poisson(lam) * eff.index_select(0, self.g_link)

    def _membrane(self, v, u, current, h):
        """Один подшаг RK4 Izhikevich 2007 (ток — по v начала подшага, как у CARLsim)."""

        inv_c = h / self.C

        def fv(vv, ww):
            adex = self.gL * (self.EL - vv + self.dT * torch.exp(((vv - self.VT) / self.dT).clamp(max=10.0)))
            return (self.k * (vv - self.vr) * (vv - self.vt) + adex - ww + current) * inv_c

        def fu(vv, ww):
            return self.a * (self.b * (vv - self.vr) - ww) * h

        k1, l1 = fv(v, u), fu(v, u)
        k2, l2 = fv(v + k1 / 2, u + l1 / 2), fu(v + k1 / 2, u + l1 / 2)
        k3, l3 = fv(v + k2 / 2, u + l2 / 2), fu(v + k2 / 2, u + l2 / 2)
        k4, l4 = fv(v + k3, u + l3), fu(v + k3, u + l3)
        return v + (k1 + 2 * k2 + 2 * k3 + k4) / 6, u + (l1 + 2 * l2 + 2 * l3 + l4) / 6

    def set_input_gain(self, tau_m: torch.Tensor, tau_syn: torch.Tensor) -> None:
        """
        пА на 1 мВ входа мозга по клеткам: пик ВПСП клетки Izhikevich в покое
        от тока c·S(t) равен пику LIF-клетки мозга от S(t) (S — ток единичного
        веса в мВ: скачок τm/τsyn, спад τsyn — engine._psp_peak). Малым
        сигналом от покоя (v = vr, u = 0) тем же RK4. Вызывает движок, когда
        знает τm и τsyn клеток зоны.
        """

        # float64: отклик малого сигнала (~10⁻⁴ мВ) у v ≈ −70 мВ в float32
        # теряется в округлении (у клеток с большой ёмкостью — до нуля).
        dt = self.dt
        sl = slice(self.first, self.first + self.n)
        tau_m, tau_syn = tau_m[sl].double(), tau_syn[sl].double()
        steps = int(10 * float(torch.maximum(tau_m, tau_syn).max()) / dt) + 1
        # LIF мозга: пик от S.
        S = tau_m / tau_syn
        zeros = torch.zeros(self.n, device=self.device, dtype=torch.float64)
        v_lif, peak_lif = zeros.clone(), zeros.clone()
        decay = torch.exp(-dt / tau_syn)
        # Izhikevich: малый ток eps·S пА, отклик линеен.
        eps = 1e-2
        params = {k: getattr(self, k) for k in ("C", "k", "vr", "vt", "a", "b", "gL", "EL", "VT", "dT")}

        for k, x in params.items():
            setattr(self, k, x.double())
        v, u = self.vr.clone(), zeros.clone()
        peak_iz = zeros.clone()
        h = dt / self.substeps

        for _ in range(steps):
            v_lif = v_lif + (S - v_lif) * dt / tau_m
            peak_lif = torch.maximum(peak_lif, v_lif)
            current = eps * S

            for _ in range(self.substeps):
                v, u = self._membrane(v, u, current, h)
            peak_iz = torch.maximum(peak_iz, (v - self.vr) / eps)
            S = S * decay

        for k, x in params.items():
            setattr(self, k, x)

        if bool((peak_iz <= 0).any()):
            raise ValueError("сеть зоны: нет отклика клетки на малый ток — множитель входа не посчитать")
        self.gain = (peak_lif / peak_iz).float()

    def set_rest_input(self, rest_mv: torch.Tensor, scale: str = "psp") -> None:
        """
        Вход мозга клеткам зоны при частотах покоя источников (мВ движка).
        Фон авторов (lognormal по типу) подобран под покой CA3 без EC и DG —
        он и есть весь внешний вход клетки в покое (их сеть не имеет входных
        зон). Явные пути мозга поверх него дали бы вход дважды (PC +22 мВ ≈
        +380 пА — сеть горела, все типы 30–390 Гц); поэтому в клетку идёт вход
        мозга минус rest_mv — так же внешний вход колонок Potjans в мозге —
        «всё внешнее минус явные пути» (Schmidt et al. 2018; engine.
        _external_minus_modeled). scale — масштаб отклонения: "psp" —
        множитель по пику ВПСП (set_input_gain), "authors" — в долях фона
        авторов (BrainConfig.circuit_input_scale).
        """

        if self.has_background:
            # Без фона авторов вход мозга идёт целиком (rest_mv остаётся 0).
            self.rest_mv.copy_(rest_mv[self.first: self.first + self.n])
            if scale == "authors":
                # Отклонение входа мозга от покоя — в долях фона авторов:
                # фон × (вход / вход покоя − 1). Фон авторов — весь вход EC и
                # DG в покое (Kopsick et al., препринт
                # https://doi.org/10.1101/2024.03.27.586909, Methods), а явный
                # путь, собранный из чисел крысы и другой мишени (EC → CA3: 3600
                # × 0.1 мВ), в покое давал PC 358 пА при всём фоне 84 пА — часть
                # больше целого (CHAT_PLAN шаг 3, циклы 31–32). Доли путей — по
                # их вкладу в покой (сумма 1), у каждой клетки — свой фон
                # (тормозные — свой тип); при частотах покоя прибавка 0,
                # колебания и спайки пути сохраняются.
                live = self.rest_mv > 1e-6
                self.gain = torch.where(live, self.bg_mean / self.rest_mv.clamp(min=1e-6), self.gain)

    # ------------------------------------------------------------------

    def step(self, ext_mv: torch.Tensor, fired_prev: torch.Tensor,
             brain_spikes: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Шаг dt. ext_mv — вход мозга клеткам зоны (мВ движка), fired_prev —
        спайки клеток зоны прошлого шага (brain_spikes — всех клеток мозга, здесь не нужны). Возвращает (v, спайки шага).
        """

        dev = self.device
        # 1. Спад STP и проводимостей.
        u = self.su * (1.0 - self.stp_iu)
        x = self.sx + (1.0 - self.sx) * self.stp_ix
        self.G.mul_(self.g_decay)

        # 2. Задержка: спайки delay_steps шагов назад.
        if self.delay_steps > 1:
            slot = self.tick % self.queue.shape[0]
            arrived = self.queue.index_select(0, slot).squeeze(0)
            self.queue.index_copy_(0, slot, fired_prev[None])
        else:
            arrived = fired_prev

        # 3. Доставка: STP по слотам выстреливших пре (u⁻, x⁻ — до спада
        # шага, как в ядре CARLsim), прирост g по синапсам.
        spiked = torch.cat([arrived, torch.zeros(1, dtype=torch.bool, device=dev)]).index_select(0, self.stp_pre)
        u_new = torch.where(spiked, u + self.stp_U * (1.0 - self.su), u)
        x_new = torch.where(spiked, x - u_new * self.sx, x)
        eff = u_new * self.sx * self.stp_scale
        self.su.copy_(u_new)
        self.sx.copy_(x_new)
        fired = torch.nonzero_static(arrived, size=self.max_fired, fill_value=self.n).squeeze(1)
        self.dropped.add_((arrived.sum() - self.max_fired).clamp(min=0))

        if self.compact:
            syn, _ = self.table.compact(fired, self.dropped)
        else:
            syn, _ = self.table.events(fired)
        self.G.index_add_(0, self.syn_g.index_select(0, syn), eff.index_select(0, self.syn_s.index_select(0, syn)))

        if self.topo:
            self._missing_step(arrived, u, x)
        live = self.G[: self.n_g]
        cells = self.g_cell[: self.n_g]
        exc = self.g_exc[: self.n_g]
        g_e = torch.zeros(self.n, device=dev).index_add_(0, cells, torch.where(exc, live, 0.0))
        g_i = torch.zeros(self.n, device=dev).index_add_(0, cells, torch.where(exc, 0.0, live))

        # 4. Фон: lognormal × U(0, 1), потолок; новый каждую 1 мс.
        if self.has_background:
            fresh = torch.exp(self.mu + self.sigma * torch.randn(self.n, device=dev))
            fresh = (fresh * torch.rand(self.n, device=dev)).clamp(max=self.cap)
            if self.bg_scale != 1.0:
                fresh = self.bg_mean + (fresh - self.bg_mean) * self.bg_scale
            self.ext.copy_(torch.where(self.tick % self.per_ms == 0, fresh, self.ext))
        # Вход мозга — сверх его величины при частотах покоя источников: фон
        # авторов — уже весь внешний вход клетки в покое.
        drive = self.ext + (ext_mv - self.rest_mv) * self.gain

        # 5. Мембрана: подшаги RK4; рефрактерность — с прошлого шага.
        h = self.dt / self.substeps
        spikes = torch.zeros(self.n, dtype=torch.bool, device=dev)
        refr = self.ref > 0
        v, uu = self.v, self.u

        for j in range(self.substeps):
            # Выстрелившая в этом шаге клетка до конца шага стоит в сбросе (v
            # = c, u не меняется): один спайк — одно u += d. У CARLsim спайк
            # возможен в каждом подшаге 1 мс, d прибавляется за каждый, а
            # спайк считается один — с отрицательным d (HIPP Hippocampome,
            # −58 пА) клетка от этого залипала на 700–1000 Гц (CHAT_PLAN
            # шаг 3, цикл 6); при покое CA3 двойных спайков за шаг нет.
            hold = refr | spikes
            current = drive - g_e * (v - E_AMPA) - g_i * (v - E_GABAA)
            v_next, u_next = self._membrane(v, uu, current, h)
            spike = (v_next > self.vpeak) & ~hold
            v_next = torch.where(spike, self.c, v_next).clamp(min=V_FLOOR)
            u_next = u_next + torch.where(spike, self.d, 0.0)

            if j == self.substeps - 1:
                v_next = torch.where(refr, self.c, v_next)
            else:
                v_next = torch.where(refr, v, v_next)
            v_next = torch.where(spikes, self.c, v_next)
            uu = torch.where(hold, uu, u_next)
            v = v_next
            spikes |= spike

        # Рефрактерность 1 мс — после спайка в любом подшаге (у CARLsim — на
        # последнем подшаге 1 мс; у нас шаг 0.5 мс, спайк в начале шага иначе
        # давал второй через 0.5 мс).
        self.ref.copy_(torch.where(refr, self.ref - 1, torch.where(spikes, self.ref_steps, self.ref)))

        self.v.copy_(v)
        self.u.copy_(uu)
        self.tick.add_(1)
        return self.v, spikes
