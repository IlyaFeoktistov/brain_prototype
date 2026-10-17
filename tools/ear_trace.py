"""
Где исчезает звук на пути слуховой нерв → MGN → слуховая кора: свежий мозг
small, звук на 65 дБ (улитка приводит к уровню речи) — чистый тон 1 кГц
500 мс и фраза голосом. По звеньям: частота до звука (500 мс тишины), в
первые 50 мс звука и на устойчивом участке; у клеток реле MGN — ещё
возбуждающий (i_syn) и тормозной (i_inh) токи, средние по 10-мс окнам.

У тона — ещё реле своей полосы (± 2 полосы от 1 кГц) и прочих отдельно.
local=1 — местное торможение реле (cfg.relay_local_inhibition).

    PYTHONPATH=. .venv/bin/python tools/ear_trace.py [local=0] [text=Кот живёт в доме.] [seed=0]
"""

import math
import sys

import torch

from brain.body import Cochlea
from brain.config import BrainConfig
from brain.mind import Mind
from brain.world import Voice

opts = {"local": "0", "text": "Кот живёт в доме.", "seed": "0"}
opts.update(dict(kv.split("=", 1) for kv in sys.argv[1:]))
cfg = BrainConfig()
cfg.save_path = "/nonexistent/brain.pt"
cfg.autosave = False
cfg.anatomy = True
cfg.hand = True
cfg.dt_ms = 0.5
cfg.seed = int(opts["seed"])
cfg.relay_local_inhibition = opts["local"] == "1"
torch.manual_seed(cfg.seed)
b = Mind(cfg).brain
b.plasticity = False
i = [s.name for s in b.sensors].index("ear")
ear = Cochlea(cfg, b.sensor_offsets[i], b.sensors[i].size, b.device)
A = b.areas["auditory"].layers
stages = [(f"{n.upper()} {k}", getattr(b.areas[n], k)) for n in ("cn", "ic") if n in b.areas for k in ("E", "I")] + [("MGN E", b.areas["mgn"].E), ("MGN I", b.areas["mgn"].I), ("TRN", b.areas["trn_mgn"].E),
          ("слух L4 E", A["L4"]["E"]), ("слух L4 PV", A["L4"]["PV"]), ("слух L2/3 E", A["L23"]["E"]), ("слух L2/3 PV", A["L23"]["PV"])]
WIN = 10.0
rate = 16000
tone = torch.sin(2 * math.pi * 1000.0 * torch.arange(int(0.5 * rate)) / rate)
sounds = [("тон 1 кГц", tone, rate), ("речь", *Voice().say(opts["text"]))]
b.observe(2500)

for name, wave, sr in sounds:
    raster = ear.hear(wave, sr)
    steps = int(WIN / cfg.dt_ms)
    nerve = raster[:, ear.offset: ear.offset + ear.size].sum(1).mean() / ear.size * 1000.0 / cfg.dt_ms
    quiet = b.observe(500)
    windows, ex, inh = [], [], []
    relay = b.areas["mgn"].E.slice

    for start in range(0, raster.shape[0] - steps + 1, steps):
        chunk = torch.zeros(steps, b.n_receptors, device=b.device)
        chunk[:, : raster.shape[1]] = raster[start: start + steps]
        windows.append(b.observe(WIN, chunk))
        ex.append(float(b.i_syn[relay].float().mean()))
        inh.append(float(b.i_inh[relay].float().mean()))

    onset = torch.stack(windows[:5]).mean(0)
    steady = torch.stack(windows[5:]).mean(0)
    print(f"{name} ({raster.shape[0] * cfg.dt_ms:.0f} мс): слуховой нерв {float(nerve):.0f} Гц на волокно (со спонтанными)")

    if name.startswith("тон"):
        # Реле своей полосы (тонотопия: клетка j — волокно j) и прочие.
        E = b.areas["mgn"].E
        fiber = torch.arange(E.size, device=b.device) % ear.size
        band = fiber // (ear.size // cfg.cochlea_bands)
        own = (band - int((ear.centres - 1000.0).abs().argmin())).abs() <= 2
        idx = torch.arange(E.start, E.stop, device=b.device)
        for tag, m in (("MGN E своей полосы", own), ("MGN E прочих", ~own)):
            c = idx[m]
            print(f"  {tag:19s} тишина {float(quiet[c].mean()):6.1f} → начало {float(onset[c].mean()):6.1f} → дальше {float(steady[c].mean()):6.1f} Гц")

    for label, pop in stages:
        print(f"  {label:13s} тишина {float(quiet[pop.slice].mean()):6.1f} → начало {float(onset[pop.slice].mean()):6.1f} → дальше {float(steady[pop.slice].mean()):6.1f} Гц")

    print(f"  токи MGN E: возб {sum(ex[:5]) / 5:+.2f} → {sum(ex[5:]) / max(1, len(ex) - 5):+.2f}, торм {sum(inh[:5]) / 5:+.2f} → {sum(inh[5:]) / max(1, len(inh) - 5):+.2f}", flush=True)
    b.observe(1000)
