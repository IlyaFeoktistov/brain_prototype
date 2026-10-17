"""
Изолированный тест правила пластичности (DESIGN «Учится ли цепь фактам»):
одна клетка (LIF, τm 20 мс, до порога 18 мВ, синапс 5 мс), n входов по
rate Гц; corr из них — общий источник с дрожью (коррелированы), остальные
независимы; средняя частота одинакова. Окно STDP — Froemke & Dan 2002
(τ+ 13.5, τ− 34.5 мс, A−/A+ = 51/103), шаг A+ — доля среднего веса за пару
(Bi & Poo 1998: ~1%). Правила (вес w в долях потолка cap = 10 средних):

- ours — как в мозге: LTP аддитивное (A+ · w̄), LTD ∝ весу (A− · w);
- gütig μ — Gütig et al. 2003: LTP A+ (1 − w)^μ, LTD A− w^μ;
- +sum — после каждых 50 мс сумма весов возвращается к исходной
  (гетеросинаптическое сохранение, engine._conserve).

Печатает: отношение среднего веса коррелированных к независимым (отражает
ли вес корреляцию), σ/среднее, доли у 0 и у потолка, среднее к исходному,
частоту клетки.

    PYTHONPATH=. .venv/bin/python tools/stdp_rule_test.py [n=500] [corr=100] [rate=3] [jitter=5] [seconds=200] [a_plus=0.01] [seed=0]
"""

import math
import sys

import torch

opts = {"n": "500", "corr": "100", "rate": "3", "jitter": "5", "seconds": "200", "a_plus": "0.01", "seed": "0", "mean_mv": ""}
opts.update(dict(kv.split("=", 1) for kv in sys.argv[1:]))
dev = "cuda" if torch.cuda.is_available() else "cpu"
n, nc = int(opts["n"]), int(opts["corr"])
rate, dt = float(opts["rate"]) / 1000.0, 0.5
steps = int(float(opts["seconds"]) * 1000 / dt)
tau_m, tau_s, gap, refr = 20.0, 5.0, 18.0, 2.0
tau_p, tau_m_stdp = 13.5, 34.5
a_plus = float(opts["a_plus"])
a_minus = a_plus * 51.0 / 103.0
# Средний вес (пик ВПСП, мВ) — чтобы клетка стреляла от флуктуаций около
# порога: средний вход ≈ порог (n · частота · пик · τ_эфф).
w_mean = float(opts["mean_mv"]) if opts["mean_mv"] else gap / (n * rate * tau_m) * 0.95
cap = 10.0 * w_mean
RULES = [("ours", None, False), ("ours", None, True)] + [("gütig", mu, s) for s in (False, True) for mu in (0.0, 0.25, 0.5, 0.75, 1.0)]
R = len(RULES)
g = torch.Generator(device=dev).manual_seed(int(opts["seed"]))
w0 = torch.rand(n, generator=g, device=dev) * 2 * w_mean
w = w0.repeat(R, 1)
total0 = w0.sum()
v = torch.zeros(R, device=dev)
syn = torch.zeros(R, device=dev)
x_pre = torch.zeros(R, n, device=dev)
x_post = torch.zeros(R, device=dev)
ref = torch.zeros(R, device=dev)
spikes_out = torch.zeros(R, device=dev)
jit = int(float(opts["jitter"]) / dt)
# Общий источник коррелированных: спайк источника → каждый вход с дрожью.
source = torch.rand(steps + jit, generator=g, device=dev) < rate
pending = torch.zeros(jit + 1, nc, device=dev)
decay_m, decay_s = math.exp(-dt / tau_m), math.exp(-dt / tau_s)
decay_p, decay_mm = math.exp(-dt / tau_p), math.exp(-dt / tau_m_stdp)
is_ours = torch.tensor([r[0] == "ours" for r in RULES], device=dev)[:, None]
mu = torch.tensor([r[1] if r[1] is not None else 0.0 for r in RULES], device=dev)[:, None]
conserve = torch.tensor([r[2] for r in RULES], device=dev)[:, None]
# Нормировка ВПСП: пик тока синапса 5 мс на мембране 20 мс.
peak = (tau_s / (tau_m - tau_s)) * ((tau_s / tau_m) ** (tau_s / (tau_m - tau_s)) - (tau_s / tau_m) ** (tau_m / (tau_m - tau_s)))
kick = 1.0 / peak * (1 - decay_m)

for t in range(steps):
    pre = (torch.rand(n, generator=g, device=dev) < rate).float()

    if nc:
        # Коррелированные: копии общего источника, каждая — со своей дрожью.
        if bool(source[t]):
            delay = torch.randint(0, jit + 1, (nc,), generator=g, device=dev)
            pending[delay, torch.arange(nc, device=dev)] = 1.0
        pre[:nc] = pending[0]
        pending = torch.roll(pending, -1, 0)
        pending[-1] = 0.0

    # Мембрана и синапс.
    syn = syn * decay_s + (w * pre).sum(1)
    v = torch.where(ref > 0, torch.zeros_like(v), v * decay_m + syn * kick)
    fired = (v >= gap).float()
    v = torch.where(fired > 0, torch.zeros_like(v), v)
    ref = torch.where(fired > 0, torch.full_like(ref, refr), (ref - dt).clamp(min=0))
    spikes_out += fired
    # Следы и STDP.
    x_pre = x_pre * decay_p + pre
    x_post = x_post * decay_mm + fired
    u = w / cap
    ltp = torch.where(is_ours, torch.full_like(w, a_plus * w_mean), a_plus * cap * (1 - u).clamp(min=0) ** mu)
    ltd = torch.where(is_ours, a_minus * w, a_minus * cap * u.clamp(min=0) ** mu)
    w = w + ltp * x_pre * fired[:, None] - ltd * x_post[:, None] * pre
    w = w.clamp(0.0, cap)

    if t % int(50 / dt) == 0:
        scaled = w * (total0 / w.sum(1, keepdim=True).clamp(min=1e-9))
        w = torch.where(conserve, scaled.clamp(max=cap), w)

secs = steps * dt / 1000.0
print(f"вход: {n} × {opts['rate']} Гц, коррелированных {nc} (дрожь {opts['jitter']} мс), {secs:.0f} с; "
      f"средний вес {w_mean:.3f} мВ, потолок {cap:.2f}, A+ {a_plus}")
print(f"{'правило':18s} {'корр/незав':>10s} {'σ/ср':>6s} {'у 0':>6s} {'у потолка':>9s} {'ср/исх':>7s} {'Гц':>6s}")

for k, (name, m, s) in enumerate(RULES):
    wk = w[k]
    ratio = float(wk[:nc].mean() / wk[nc:].mean().clamp(min=1e-9)) if nc else float("nan")
    tag = f"{name}{'' if m is None else f' μ={m}'}{' +sum' if s else ''}"
    print(f"{tag:18s} {ratio:10.2f} {float(wk.std() / wk.mean().clamp(min=1e-9)):6.2f} {float((wk < 1e-3 * cap).float().mean()) * 100:5.1f}% "
          f"{float((wk > 0.99 * cap).float().mean()) * 100:8.1f}% {float(wk.mean() / w0.mean()):7.2f} {float(spikes_out[k]) / secs:6.1f}")
