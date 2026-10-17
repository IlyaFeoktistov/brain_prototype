"""
Вход клетки L2/3 PV по путям (частота источника × вес) и постоянные токи —
кто гонит популяцию.

    PYTHONPATH=. .venv/bin/python tools/anatomy_inputs.py
"""

import torch
from brain.config import BrainConfig
from brain.mind import Mind
torch.manual_seed(0)
cfg = BrainConfig(); cfg.save_path="/nonexistent/b.pt"; cfg.autosave=False
cfg.anatomy=True; cfg.hand=True; cfg.size="small"; cfg.dt_ms=0.5
b = Mind(cfg).brain; b.plasticity=False; b.run(2500.0)
r = b.observe(1000.0)
tau = cfg.e_tau_syn
for zone in ("auditory", "prefrontal"):
    pv = b.areas[zone].layers["L23"]["PV"]
    print(f"\n{zone} L2/3 PV: {float(r[pv.slice].mean()):.1f} Гц; пост. ток возб {float(b.dc_mv[pv.slice].mean()):.2f}, торм {float(b.dc_inh_mv[pv.slice].mean()):.2f}, ток покоя {float(b.bias[pv.slice].mean()):.2f}, фон шума ×{float(b.noise_gain[pv.slice].mean()):.2f}")
    rows = []
    for p in b.projections:
        if p.post is not pv and p.post.name != pv.name: continue
        if p.source != "net": continue
        pre = r.index_select(0, p.pre.flatten().long()).view(p.pre.shape)
        drive = float((pre * p.weight.float()).sum(1).mean()) * tau / 1000.0
        rows.append((drive, p.name, p.pre.shape[1], float(pre.mean())))
    for d, n, k, hz in sorted(rows, key=lambda x: -abs(x[0]))[:12]:
        print(f"   {n:40s} k={k:4d} частота пре {hz:6.1f}  вклад {d:7.2f}")
