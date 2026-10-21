"""
Тонотопия слухового пути (DESIGN «Слух на точном движке»): свежий мозг
small, тоны tones= Гц по 500 мс на 65 дБ, одно зерно и фон. Для каждого
тона и звена (CN, IC, MGN — клетка j: полоса волокна j; A1 L4 — клетка i:
полоса (i + 0.5) · полос / клеток) — прибавка частоты по полосам
(устойчивый участок минус тишина), полоса наибольшей прибавки против
полосы тона и ширина отклика (полос с прибавкой больше половины
наибольшей). У IC полосы тона — вход: от CN, местное торможение (вес ×
частота источника; ток × τ/1000) и доля местного в тормозном.

inh_off=ic — местное торможение главных клеток IC выключено (контроль).

    PYTHONPATH=. .venv/bin/python tools/tonotopy_check.py [tones=500,1000,2000,4000] [inh_off=] [seed=0]
"""

import math
import sys

import torch

from brain.body import Cochlea
from brain.config import BrainConfig
from brain.mind import Mind

opts = {"tones": "500,1000,2000,4000", "inh_off": "", "seed": "0"}
opts.update(dict(kv.split("=", 1) for kv in sys.argv[1:]))
cfg = BrainConfig()
cfg.save_path = "/nonexistent/brain.pt"
cfg.autosave = False
cfg.dt_ms = 0.5
cfg.seed = int(opts["seed"])
torch.manual_seed(cfg.seed)
b = Mind(cfg).brain
b.plasticity = False
i = [s.name for s in b.sensors].index("ear")
ear = Cochlea(cfg, b.sensor_offsets[i], b.sensors[i].size, b.device)
bands = cfg.cochlea_bands
per = ear.size // bands

for name in filter(None, opts["inh_off"].split(",")):
    nucleus = b.areas[name]
    for p in b.projections:
        if p.name == f"{name}: I->E":
            p.weight.zero_()
    b.mf_inh[int(b.mf_post_id[nucleus.E.start]), int(b.mf_pre_id[nucleus.I.start])] = 0.0

L4 = b.areas["auditory"].layers["L4"]["E"]
stages = {
    "CN": (b.areas["cn"].E, (torch.arange(b.areas["cn"].E.size) % ear.size) // per),
    "IC": (b.areas["ic"].E, (torch.arange(b.areas["ic"].E.size) % ear.size) // per),
    "MGN": (b.areas["mgn"].E, (torch.arange(b.areas["mgn"].E.size) % ear.size) // per),
    "A1 L4": (L4, ((torch.arange(L4.size).float() + 0.5) * bands / L4.size).long()),
}
IC = b.areas["ic"]
local = [p for p in b.projections if p.name == "ic: I->E"]
up = [p for p in b.projections if p.name == "cn -> ic.E"]
tau = float(b.tau_m[IC.E.start])


def ic_inputs(r: torch.Tensor, rows: torch.Tensor) -> tuple[float, float]:
    def total(paths):
        return sum(float((p.weight[rows].float() * r[p.pre[rows].long()]).sum(1).mean()) for p in paths) * tau / 1000.0

    return total(up), total(local)


b.run(1500.0)
rate = 16000
print(f"inh_off={opts['inh_off'] or '—'}; полоса = {bands} полос на {cfg.cochlea_low_hz:.0f}–{cfg.cochlea_high_hz:.0f} Гц")

for hz in (float(x) for x in opts["tones"].split(",")):
    want = int((ear.centres - hz).abs().argmin())
    tone = torch.sin(2 * math.pi * hz * torch.arange(int(0.5 * rate)) / rate)
    raster = ear.hear(tone, rate)
    quiet = b.observe(400.0)
    sensory = torch.zeros(raster.shape[0], b.n_receptors, device=b.device)
    sensory[:, : raster.shape[1]] = raster
    b.observe(100.0, sensory[: int(100 / cfg.dt_ms)])
    loud = b.observe(400.0, sensory[int(100 / cfg.dt_ms): int(500 / cfg.dt_ms)])
    line = []

    for name, (pop, band) in stages.items():
        cells = torch.arange(pop.start, pop.stop)
        gain = (loud[cells] - quiet[cells]).cpu()
        by_band = torch.zeros(bands).index_add_(0, band, gain) / torch.bincount(band, minlength=bands).clamp(min=1)
        top = int(by_band.argmax())
        width = int((by_band > 0.5 * by_band.max()).sum()) if float(by_band.max()) > 0 else 0
        line.append(f"{name}: полоса {top} ({by_band[top]:+.1f} Гц), ширина {width}")

    rows = torch.arange(IC.E.size, device=b.device)
    rows = rows[((rows % ear.size) // per - want).abs() <= 2]
    q_up, q_loc = ic_inputs(quiet, rows)
    l_up, l_loc = ic_inputs(loud, rows)
    print(f"тон {hz:.0f} Гц (полоса {want}): " + "; ".join(line))
    print(f"    IC своей полосы: от CN {q_up:+.1f} → {l_up:+.1f}, местное торможение {q_loc:+.1f} → {l_loc:+.1f}", flush=True)
    b.observe(600.0)
