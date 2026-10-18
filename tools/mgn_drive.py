"""
Вход реле и тормозных MGN по путям: Σ вес × частота источника (Гц) — тишина,
тон 1 кГц (своя полоса ± 2) и речь. Разделяет источники торможения реле:
местные интернейроны (mgn: I->E), TRN (trn_mgn -> mgn); возбуждение — IC и
кора. local=1 — вход и выход интернейронов по карте
(cfg.relay_local_inhibition). Режим — диагностический (IC без местного торможения, фон fit=).

    PYTHONPATH=. .venv/bin/python tools/mgn_drive.py [fit=ic-off] [local=0] [scale=out*0.5] [seed=0]
"""

import math
import sys

import torch
from brain.body import Cochlea
from brain.config import BrainConfig
from brain.mind import Mind
from brain.world import Voice
from brain import anatomy

opts = {"fit": "ic-off", "local": "0", "scale": "", "seed": "0"}
opts.update(dict(kv.split("=", 1) for kv in sys.argv[1:]))
cfg = BrainConfig(); cfg.save_path = "/nonexistent/brain.pt"; cfg.autosave = False
cfg.anatomy = True; cfg.hand = True; cfg.dt_ms = 0.5; cfg.seed = int(opts["seed"]); cfg.ic_local_inhibition = False
cfg.relay_local_inhibition = opts["local"] == "1"
if opts["fit"]:
    anatomy.use_fit(cfg, opts["fit"])
torch.manual_seed(cfg.seed)
b = Mind(cfg).brain; b.plasticity = False
i = [s.name for s in b.sensors].index("ear")
ear = Cochlea(cfg, b.sensor_offsets[i], b.sensors[i].size, b.device)
m = b.areas["mgn"]
# scale=out*0.5,in*0.25 — диагностика: вес пути в этом мозге × множитель
# (out — интернейроны → реле, in — IC → интернейроны); модель не меняется.
SCALE = {"out": "mgn: I->E", "in": "ic -> mgn.I"}
for item in filter(None, opts.get("scale", "").split(",")):
    key, factor = item.split("*")
    for p in b.projections:
        if p.name == SCALE[key]:
            p.weight.mul_(float(factor))
paths = [p for p in b.projections if (p.post is m.E or p.post is m.I) and float(p.weight.abs().sum()) > 0]
centre = int((ear.centres - 1000.0).abs().argmin())
dist = {tag: (b._relay_place(m, pop)[:, 0].long() - centre).abs() for tag, pop in (("E", m.E), ("I", m.I))}
BANDS = {"всё": (0, 999), "своя": (0, 2), "соседи": (3, 10)}


def drive(r, band):
    lo, hi = BANDS[band]
    out = {}
    for p in paths:
        tag = "E" if p.post is m.E else "I"
        d = (p.weight.float() * r[p.pre.long()]).sum(1)
        out[f"{p.name} → {tag}"] = float(d[(dist[tag] >= lo) & (dist[tag] <= hi)].mean())
    return out


b.observe(2500)
q = b.observe(1000)
rates = {"тишина": q}
tone = torch.sin(2 * math.pi * 1000.0 * torch.arange(8000) / 16000)
for name, wave, sr in (("тон", tone, 16000), ("речь", *Voice().say("Кот живёт в доме."))):
    raster = ear.hear(wave, sr)
    chunk = torch.zeros(raster.shape[0], b.n_receptors, device=b.device)
    chunk[:, : raster.shape[1]] = raster
    b.observe(50, chunk[:100])
    rates[name] = b.observe((raster.shape[0] - 100) * cfg.dt_ms, chunk[100:])
    b.observe(1000)

for name, band in (("тишина", "всё"), ("речь", "всё"), ("тишина", "своя"), ("тон", "своя"), ("тишина", "соседи"), ("тон", "соседи")):
    r = rates[name]
    lo, hi = BANDS[band]
    sel = {tag: (dist[tag] >= lo) & (dist[tag] <= hi) for tag in dist}
    print(f"{name}, {band}: реле {float(r[m.E.slice][sel['E']].mean()):.1f} Гц, интернейроны {float(r[m.I.slice][sel['I']].mean()):.1f} Гц, TRN {float(r[b.areas['trn_mgn'].E.slice].mean()):.1f} Гц")
    for k, v in drive(r, band).items():
        print(f"    {k:42s} {v:+9.1f}")
