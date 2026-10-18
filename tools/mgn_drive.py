"""
Вход реле и тормозных MGN по путям: Σ вес × частота источника (Гц) — тишина,
тон 1 кГц (своя полоса ± 2) и речь. Разделяет источники торможения реле:
местные интернейроны (mgn: I->E), TRN (trn_mgn -> mgn); возбуждение — IC и
кора. Режим — диагностический (IC без местного торможения, фон fit=).

    PYTHONPATH=. .venv/bin/python tools/mgn_drive.py [fit=ic-off] [seed=0]
"""

import math
import sys

import torch
from brain.body import Cochlea
from brain.config import BrainConfig
from brain.mind import Mind
from brain.world import Voice
from brain import anatomy

opts = {"fit": "ic-off", "seed": "0"}
opts.update(dict(kv.split("=", 1) for kv in sys.argv[1:]))
cfg = BrainConfig(); cfg.save_path = "/nonexistent/brain.pt"; cfg.autosave = False
cfg.anatomy = True; cfg.hand = True; cfg.dt_ms = 0.5; cfg.seed = int(opts["seed"]); cfg.ic_local_inhibition = False
if opts["fit"]:
    anatomy.use_fit(cfg, opts["fit"])
torch.manual_seed(cfg.seed)
b = Mind(cfg).brain; b.plasticity = False
i = [s.name for s in b.sensors].index("ear")
ear = Cochlea(cfg, b.sensor_offsets[i], b.sensors[i].size, b.device)
m = b.areas["mgn"]
paths = [p for p in b.projections if (p.post is m.E or p.post is m.I) and float(p.weight.abs().sum()) > 0]
centre = int((ear.centres - 1000.0).abs().argmin())
own = {tag: (b._relay_place(m, pop)[:, 0].long() - centre).abs() <= 2 for tag, pop in (("E", m.E), ("I", m.I))}
def drive(r, sel):
    out = {}
    for p in paths:
        tag = "E" if p.post is m.E else "I"
        d = (p.weight.float() * r[p.pre.long()]).sum(1)
        d = d[own[tag]] if sel else d
        out[f"{p.name} → {tag}"] = float(d.mean())
    return out
b.observe(2500)
q = b.observe(1000)
rows = {"тишина": (drive(q, False), q)}
tone = torch.sin(2 * math.pi * 1000.0 * torch.arange(8000) / 16000)
for name, wave, sr, sel in (("тон, своя полоса", tone, 16000, True), ("речь", *Voice().say("Кот живёт в доме."), False)):
    raster = ear.hear(wave, sr)
    chunk = torch.zeros(raster.shape[0], b.n_receptors, device=b.device); chunk[:, : raster.shape[1]] = raster
    b.observe(50, chunk[:100])
    r = b.observe((raster.shape[0] - 100) * cfg.dt_ms, chunk[100:])
    rows[name] = (drive(r, sel), r)
    b.observe(1000)
rows["тишина, своя полоса"] = (drive(q, True), q)
for name in ("тишина", "речь", "тишина, своя полоса", "тон, своя полоса"):
    d, r = rows[name]
    sel = "своя" in name
    rE = r[m.E.slice][own["E"]] if sel else r[m.E.slice]
    rI = r[m.I.slice][own["I"]] if sel else r[m.I.slice]
    print(f"{name}: реле {float(rE.mean()):.1f} Гц, тормозные {float(rI.mean()):.1f} Гц, TRN {float(r[b.areas['trn_mgn'].E.slice].mean()):.1f} Гц")
    for k, v in d.items():
        print(f"    {k:42s} {v:+9.1f}")
