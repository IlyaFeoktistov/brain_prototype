"""
Где исчезает звук на пути слуховой нерв → MGN → слуховая кора: свежий мозг
small, звук на 65 дБ (улитка приводит к уровню речи) — чистый тон 1 кГц
500 мс и фраза голосом. По звеньям: частота до звука (500 мс тишины), в
первые 50 мс звука и на устойчивом участке; у клеток реле MGN — ещё
возбуждающий (i_syn) и тормозной (i_inh) токи, средние по 10-мс окнам.

У тона — ещё реле своей полосы (± 2 полосы от 1 кГц) и прочих отдельно.
local=1 — местное торможение реле (cfg.relay_local_inhibition).
inh_off=cn,ic — диагностика: местное торможение главных клеток этих ядер
выключено (синапсы тормозные → E и их среднее поле), остальное как есть.
У тона, своя полоса — ещё задержка первого спайка после начала (медиана
по клеткам, ответившим за 50 мс).

    PYTHONPATH=. .venv/bin/python tools/ear_trace.py [local=0] [text=Кот живёт в доме.] [seed=0]
"""

import math
import sys

import torch

from brain.body import Cochlea
from brain.config import BrainConfig
from brain.mind import Mind
from brain.world import Voice

opts = {"inh_off": "", "local": "0", "text": "Кот живёт в доме.", "seed": "0"}
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

for name in filter(None, opts["inh_off"].split(",")):
    nucleus = b.areas[name]
    for p in b.projections:
        if p.name == f"{name}: I->E":
            p.weight.zero_()
    b.mf_inh[int(b.mf_post_id[nucleus.E.start]), int(b.mf_pre_id[nucleus.I.start])] = 0.0
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

    first = None

    for start in range(0, raster.shape[0] - steps + 1, steps):
        chunk = torch.zeros(steps, b.n_receptors, device=b.device)
        chunk[:, : raster.shape[1]] = raster[start: start + steps]
        windows.append(b.observe(WIN, chunk))

        if first is None:
            first = torch.full((b.n,), float("inf"), device=b.device)
        hit = (windows[-1][: b.n] > 0) & torch.isinf(first)
        first[hit] = start * cfg.dt_ms + WIN
        ex.append(float(b.i_syn[relay].float().mean()))
        inh.append(float(b.i_inh[relay].float().mean()))

    onset = torch.stack(windows[:5]).mean(0)
    steady = torch.stack(windows[5:]).mean(0)
    print(f"{name} ({raster.shape[0] * cfg.dt_ms:.0f} мс): слуховой нерв {float(nerve):.0f} Гц на волокно (со спонтанными)")

    if name.startswith("тон"):
        # Своя полоса (тонотопия: клетка j — волокно j, ± 2 полосы от 1 кГц) и прочие.
        for nucleus in ("cn", "ic", "mgn"):
            E = b.areas[nucleus].E
            fiber = torch.arange(E.size, device=b.device) % ear.size
            band = fiber // (ear.size // cfg.cochlea_bands)
            own = (band - int((ear.centres - 1000.0).abs().argmin())).abs() <= 2
            idx = torch.arange(E.start, E.stop, device=b.device)
            for tag, m in ((f"{nucleus.upper()} E своей полосы", own), (f"{nucleus.upper()} E прочих", ~own)):
                c = idx[m]
                lat = first[c]
                lat = lat[lat <= 50]
                when = f", первый спайк ~{float(lat.median()):.0f} мс у {len(lat)}/{len(c)}" if m is own and len(lat) else ""
                print(f"  {tag:19s} тишина {float(quiet[c].mean()):6.1f} → начало {float(onset[c].mean()):6.1f} → дальше {float(steady[c].mean()):6.1f} Гц{when}")

        # Слуховая L4 — тонотопия: клетка i — полоса (i + 0.5) · полос / клеток.
        L4 = A["L4"]["E"]
        band = ((torch.arange(L4.size, device=b.device).float() + 0.5) * cfg.cochlea_bands / L4.size).long()
        own = (band - int((ear.centres - 1000.0).abs().argmin())).abs() <= 2
        idx = torch.arange(L4.start, L4.stop, device=b.device)
        for tag, m in (("A1 L4 E своей полосы", own), ("A1 L4 E прочих", ~own)):
            c = idx[m]
            print(f"  {tag:19s} тишина {float(quiet[c].mean()):6.1f} → начало {float(onset[c].mean()):6.1f} → дальше {float(steady[c].mean()):6.1f} Гц")

    for label, pop in stages:
        print(f"  {label:13s} тишина {float(quiet[pop.slice].mean()):6.1f} → начало {float(onset[pop.slice].mean()):6.1f} → дальше {float(steady[pop.slice].mean()):6.1f} Гц")

    print(f"  токи MGN E: возб {sum(ex[:5]) / 5:+.2f} → {sum(ex[5:]) / max(1, len(ex) - 5):+.2f}, торм {sum(inh[:5]) / 5:+.2f} → {sum(inh[5:]) / max(1, len(inh) - 5):+.2f}", flush=True)
    b.observe(1000)
