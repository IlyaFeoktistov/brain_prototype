"""
Клетка реле LGN одна, вне мозга: AdEx движка (утечка, экспонента, T-ток
IFB, AMPA с движущей силой, покой −63) — контроль, и тот же AdEx с
динамическим порогом θ(t) вместо постоянного V_T. Сеть и движок не меняются:
это проверка гипотезы, не механизм мозга.

Порог от инактивации натриевых каналов (Platkiewicz & Brette 2011,
https://doi.org/10.1371/journal.pcbi.1001129, теория): θ = θ₀ − k_a·ln h,
h∞(V) = 1 / (1 + e^((V − V_i)/k_i)) →

    θ∞(V) = θ₀ + k_a·ln(1 + e^((V − V_i)/k_i)),   dθ/dt = (θ∞ − θ) / τ_θ.

k_a — наклон активации Na; в AdEx это ΔT (у модели 2 мВ), у каналов —
единицы мВ. При k_a ≥ k_i порог на медленной рампе растёт не медленнее V
и спайка нет вовсе (клетка реле на рампу стреляет) — ряд k_a < k_i. V_i, k_i, τ_θ для таламуса не измерены — ряд значений.
θ₀ у каждого варианта — такой, чтобы при медленной рампе порог был как в
срезе (контроль, −40.3 мВ по dV/dt > 10 при 0.1 мВ/мс; данные среза
−39 ± 4). Предсказания, которые сравниваются с опытом:
    1) порог у быстрого ВПСП (критерий dV/dt > 20) — ниже, чем у рампы?
    2) порог против наклона V за 10 мс (рампы 0.02–5 мВ/мс): TRN кошки in vivo
       −51.3 − 2.09 мс × наклон (https://doi.org/10.1371/journal.pone.0030154);
    3) восстановление: после 100 мс на −55 мВ порог ВПСП выше и
       возвращается к покою за время ~τ инактивации.
Ещё: передача одиночного ВПСП 6.75 мВ (как у модели) с −63 и −59 мВ.

    PYTHONPATH=. python tools/relay_dynamic_threshold.py
"""

from __future__ import annotations

import math

import torch

from brain.config import BrainConfig

cfg = BrainConfig()
DT = 0.02
TAU_M, DELTA_T, TAU_AMPA = 20.0, cfg.e_delta_t, cfg.relay_ampa_tau_ms


class Cell:
    """Пачка клеток (по одной на вариант θ) — уравнение мембраны как в engine.step."""

    def __init__(self, theta0, k_a, v_i, k_i, tau_th, dynamic):
        self.theta0, self.k_a, self.v_i, self.k_i, self.tau_th = theta0, k_a, v_i, k_i, tau_th
        self.dynamic = dynamic
        n = theta0.numel()
        self.v = torch.full((n,), -63.0, dtype=torch.float64)
        self.th = self.theta_inf(self.v)
        self.i = torch.zeros(n, dtype=torch.float64)
        self.t_h = torch.sigmoid((cfg.t_h_half_mv - self.v) / cfg.t_h_slope_mv)
        self.bias = torch.full((n,), -63.0 - cfg.v_rest, dtype=torch.float64)

    def theta_inf(self, v):
        if not self.dynamic:
            return self.theta0.clone()
        return self.theta0 + self.k_a * torch.nn.functional.softplus((v - self.v_i) / self.k_i)

    def settle(self, v, ms=300.0):
        self.bias.fill_(v - cfg.v_rest)
        for _ in range(int(ms / DT)):
            self.step()

    def step(self, kick=None):
        v = self.v
        if kick is not None:
            self.i += kick
        self.i *= math.exp(-DT / TAU_AMPA)
        excite = ((cfg.ampa_reversal_mv - v) / (cfg.ampa_reversal_mv - cfg.epsp_holding_mv)).clamp(min=0.0)
        exp_term = DELTA_T * torch.exp(((v - self.th) / DELTA_T).clamp(max=10.0))
        t_open = v >= cfg.t_vh_mv
        t_cur = torch.where(t_open, cfg.t_gt_ratio * self.t_h * (cfg.t_et_mv - v), torch.zeros_like(v))
        dv = (cfg.v_rest - v + exp_term + self.i * excite + self.bias + t_cur) * DT / TAU_M
        h_inf = torch.sigmoid((cfg.t_h_half_mv - v) / cfg.t_h_slope_mv)
        self.t_h = torch.where(v < cfg.t_vh_mv, self.t_h + (h_inf - self.t_h) * DT / cfg.t_tau_rec_ms,
                               self.t_h - self.t_h * DT / cfg.t_tau_inact_ms)
        self.th = self.th + (self.theta_inf(v) - self.th) * DT / self.tau_th
        self.v = v + dv
        return dv / DT


def onset(cell, steps, drive, crit):
    """Начало первого спайка (V на шаге до dV/dt > crit) и наклоны V за 10 и 1 мс до него."""

    hist = [cell.v.clone()]
    n = cell.v.numel()
    on, s10, s1 = (torch.full((n,), math.nan, dtype=torch.float64) for _ in range(3))
    n10, n1 = round(10 / DT), round(1 / DT)

    for k in range(steps):
        drive(k)
        rate = cell.step(cell.kick if k == 0 and getattr(cell, "kick", None) is not None else None)
        hit = (rate > crit) & on.isnan()
        if hit.any():
            prev = hist[-1]
            on[hit] = prev[hit]
            if len(hist) > n10:
                s10[hit] = ((prev - hist[-1 - n10]) / 10)[hit]
            if len(hist) > n1:
                s1[hit] = ((prev - hist[-1 - n1]) / 1)[hit]
        hist.append(cell.v.clone())
        if not on.isnan().any():
            break

    cell.kick = None
    return on, s10, s1


def ramp(cell, rate, crit):
    cell.settle(-63.0)
    def drive(k):
        cell.bias += rate * DT
    return onset(cell, int((40.0 / rate + 50.0) / DT), drive, crit)


def epsp_kick(amp_mv):
    """Ток AMPA, дающий ВПСП amp_mv с −63 мВ у клетки без экспоненты (подбор по пику)."""

    probe = Cell(torch.tensor([0.0]), 0.0, 0.0, 1.0, 1.0, False)
    probe.theta0 = torch.tensor([200.0], dtype=torch.float64)
    probe.th = probe.theta0.clone()
    lo, hi = 0.0, 500.0
    for _ in range(40):
        mid = (lo + hi) / 2
        probe.v.fill_(-63.0); probe.i.zero_(); probe.bias.fill_(-63.0 - cfg.v_rest)
        probe.t_h = torch.sigmoid((cfg.t_h_half_mv - probe.v) / cfg.t_h_slope_mv)
        peak = -63.0
        for k in range(int(30 / DT)):
            probe.step(torch.tensor([mid], dtype=torch.float64) if k == 0 else None)
            peak = max(peak, float(probe.v))
        lo, hi = (mid, hi) if peak - (-63.0) < amp_mv else (lo, mid)
    return (lo + hi) / 2


def epsp(cell, v0, kick, crit=20.0, pre=None):
    if pre is None:
        cell.settle(v0)
    else:
        cell.settle(pre[0], pre[1])
        cell.bias.fill_(v0 - cfg.v_rest)
        for _ in range(int(pre[2] / DT)):
            cell.step()
    cell.kick = torch.full_like(cell.v, kick)
    on, _, s1 = onset(cell, int(25 / DT), lambda k: None, crit)
    return on, s1


def main() -> None:
    grid = [(k_a, v_i, k_i, tau) for k_a in (2.0, 4.0) for v_i in (-63.0, -55.0) for k_i in (6.0,)
            for tau in (1.0, 3.0, 10.0)]
    k_a, v_i, k_i, tau = (torch.tensor(c, dtype=torch.float64) for c in zip(*grid))
    n = len(grid)
    target = -40.3

    # θ₀: медленная рампа (0.1 мВ/мс, dV/dt > 10) — как у контроля.
    lo, hi = torch.full((n,), -90.0, dtype=torch.float64), torch.full((n,), -30.0, dtype=torch.float64)
    for _ in range(18):
        mid = (lo + hi) / 2
        on, _, _ = ramp(Cell(mid, k_a, v_i, k_i, tau, True), 0.1, 10.0)
        up = on < target
        lo, hi = torch.where(up, mid, lo), torch.where(up, hi, mid)
    theta0 = (lo + hi) / 2

    ctl = lambda: Cell(torch.full((1,), cfg.v_threshold, dtype=torch.float64), 0.0, 0.0, 1.0, 1.0, False)
    dyn = lambda: Cell(theta0.clone(), k_a, v_i, k_i, tau, True)
    names = ["контроль AdEx"] + [f"k_a {a:.0f} V_i {b:.0f} τθ {d:>4.0f}" for a, b, c, d in grid]

    def both(fn):
        a, b = fn(ctl()), fn(dyn())
        return [torch.cat([x, y]) for x, y in zip(a, b)]

    print(f"θ₀ подобран под рампу −40.3: " + " ".join(f"{float(t):.1f}" for t in theta0))
    rates = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
    rows = [both(lambda c, r=r: ramp(c, r, 20.0)) for r in rates]
    on = torch.stack([r[0] for r in rows]); s10 = torch.stack([r[1] for r in rows])
    ok = ~(on.isnan() | s10.isnan())
    cnt = ok.sum(0)
    mx = torch.where(ok, s10, 0.0).sum(0) / cnt; my = torch.where(ok, on, 0.0).sum(0) / cnt
    x = torch.where(ok, s10 - mx, 0.0); y = torch.where(ok, on - my, 0.0)
    slope = (x * y).sum(0) / (x * x).sum(0)
    icpt = my - slope * mx

    kick = epsp_kick(6.75)
    big = epsp_kick(13.0)
    e_on, e_s1 = both(lambda c: epsp(c, -59.0, big))
    rec = [both(lambda c, t=t: epsp(c, -59.0, big, pre=(-55.0, 100.0, t))) for t in (2.0, 10.0, 30.0, 100.0)]
    pass63 = both(lambda c: epsp(c, -63.0, kick))[0]
    pass59 = both(lambda c: epsp(c, -59.0, kick))[0]

    print("\nкритерий dV/dt > 20 мВ/мс; ВПСП 13 мВ с −59 (рампы — с −63)")
    print(f"{'вариант':26s} {'рампа 0.1':>9s} {'рампа 5':>8s} {'линия (наклон за 10 мс)':>24s} {'ВПСП':>6s} "
          f"{'после −55: 2 / 10 / 30 / 100 мс':>32s}  ВПСП 6.75 спайк с −63 / −59")
    for j, name in enumerate(names):
        r = "  ".join(f"{float(x[0][j]):6.1f}" for x in rec)
        p = lambda t: "да" if not math.isnan(float(t[j])) else "нет"
        print(f"{name:26s} {float(on[2, j]):9.1f} {float(on[-1, j]):8.1f} {float(icpt[j]):10.1f} {float(slope[j]):+6.2f}·наклон"
              f" {float(e_on[j]):6.1f}   {r}   {p(pass63)} / {p(pass59)}")
    print("TRN кошки in vivo: линия −51.3 −2.09·наклон; ВПСП −44.7 −0.62·наклон за 1 мс")


if __name__ == "__main__":
    main()
