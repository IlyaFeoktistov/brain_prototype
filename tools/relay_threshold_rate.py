"""
Порог клетки реле LGN (вне сети, как в relay_threshold) против скорости
деполяризации перед спайком. В TRN кошки in vivo порог ниже при быстрой
деполяризации: −51.3 − 2.09 мс × наклон за 10 мс до спайка, у ВПСП
−44.7 − 0.62 мс × наклон за 1 мс (одна клетка; критерий dV/dt > 20 мВ/мс;
https://doi.org/10.1371/journal.pone.0030154, in vivo).

    а) рампа тока от −63 мВ с разной скоростью;
    б) ВПСП сетчатки ×k с удерживаемого V. При dV/dt > 10 сам подъём
       сильного ВПСП (×3 и выше) быстрее критерия — «порог» там равен
       начальному V, это не начало спайка.

    PYTHONPATH=. python tools/relay_threshold_rate.py [dt=0.1]
"""

import sys
import torch
from brain.config import BrainConfig
from brain.mind import Mind

cfg = BrainConfig(); cfg.size = "small"; cfg.autosave = False; cfg.dt_ms = float(dict(a.split('=') for a in sys.argv[1:]).get('dt', 0.1))
b = Mind(cfg).brain; b.plasticity = False
lgn = b.areas["lgn"].E
eye = [x.name for x in b.sensors].index("eye")
off, size = b.sensor_offsets[eye], b.sensors[eye].size
b.spontaneous[off: off + size] = 0.0
for p in b.projections:
    if p.post is lgn and not p.name.startswith("eye"):
        p.weight.zero_()
b.noise_gain[lgn.slice] = 0.0
drive = [p for p in b.projections if p.post is lgn and p.name.startswith("eye")]
w0 = [p.weight.clone() for p in drive]
spread = b.bias[lgn.slice] - b.bias[lgn.slice].mean()
b.CHUNK, b._graphs = 1, {}
dt = cfg.dt_ms
quiet = torch.zeros(1, b.n_receptors, device=b.device)
n10, n1 = int(round(10 / dt)), int(round(1 / dt))
print(f"dt {dt} мс, V_T {cfg.v_threshold}, ΔT {float(b.delta_t[lgn.start]):.1f}, τm {float(b.tau_m[lgn.start]):.0f}")

def hold(v, ms=150.0):
    b.bias[lgn.slice] = (v - cfg.v_rest) + spread
    for _ in range(int(ms / dt)):
        b.run(dt, quiet)

def onset_run(steps, step_fn, crit):
    hist = [b.v[lgn.slice].clone()]
    on = torch.full((lgn.size,), float("nan"), device=b.device)
    s10 = on.clone(); s1 = on.clone()
    for k in range(steps):
        step_fn(k)
        v = b.v[lgn.slice].clone(); prev = hist[-1]
        hit = ((v - prev) / dt > crit) & on.isnan()
        if hit.any():
            on[hit] = prev[hit]
            if len(hist) > n10: s10[hit] = (prev - hist[-1 - n10])[hit] / 10
            if len(hist) > n1: s1[hit] = (prev - hist[-1 - n1])[hit] / 1
        hist.append(v)
    ok = ~on.isnan()
    return on[ok], s10[ok], s1[ok]

for crit in (10.0, 20.0):
    print(f"\nкритерий dV/dt > {crit:.0f} мВ/мс")
    print("  а) рампа тока от −63: скорость фона мВ/мс → порог (медиана), наклон V за 10 мс")
    xs, ys = [], []
    for rate in (0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0):
        hold(-63.0)
        def f(k):
            b.bias[lgn.slice] += rate * dt
            b.run(dt, quiet)
        on, s10, _ = onset_run(int(min(600, 60 / rate + 50) / dt), f, crit)
        if len(on):
            print(f"     {rate:5.2f}: порог {float(on.median()):6.1f}  наклон {float(s10.nanmedian()):5.2f} мВ/мс  ({len(on)} клеток)")
            xs.append(s10); ys.append(on)
    x, y = torch.cat(xs), torch.cat(ys); m = ~x.isnan()
    x, y = x[m], y[m]
    k = float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum())
    print(f"     линия: порог = {float(y.mean() - k * x.mean()):.1f} {k:+.2f}·наклон10   [TRN in vivo: −51.3 −2.09]")
    print("  б) ВПСП сетчатки ×k с удерживаемого V: наклон V за 1 мс → порог")
    xs, ys = [], []
    for scale in (2.0, 3.0, 4.0, 6.0):
        for p, w in zip(drive, w0): p.weight.copy_(w * scale)
        for v0 in (-60.0, -56.0, -52.0):
            hold(v0)
            kick = quiet.clone(); kick[0, off: off + size] = 1.0
            on, _, s1 = onset_run(int(20 / dt), lambda k: b.run(dt, kick if k == 0 else quiet), crit)
            if len(on) > 5:
                print(f"     ×{scale:.0f} с {v0:.0f}: порог {float(on.median()):6.1f}  наклон1 {float(s1.nanmedian()):5.2f}  ({len(on)} клеток)")
                xs.append(s1); ys.append(on)
    for p, w in zip(drive, w0): p.weight.copy_(w)
    if xs:
        x, y = torch.cat(xs), torch.cat(ys); m = ~x.isnan(); x, y = x[m], y[m]
        k = float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum())
        print(f"     линия: порог = {float(y.mean() - k * x.mean()):.1f} {k:+.2f}·наклон1   [TRN in vivo, ВПСП: −44.7 −0.62]")
