"""
Синхронность популяций анатомического мозга по шагам (DESIGN «Аудит»):
свежий мозг small, покой. Диагностика строит мозг с чанком в 1 шаг
(SpikingBrain.CHUNK = 1; обычно 10 шагов = 5 мс за запуск, и observe
короче чанка всё равно живёт чанк), чтобы спайки были по шагам 0.5 мс.

У популяций pops= (по умолчанию — гиппокамп, базальные ганглии, мозжечок,
таламус, кора): средние спайки на клетку за шаг, доля шагов без спайков,
доля шагов, где стреляет > 10 / 25 / 50 / 90% клеток, период ритма
(пик автокорреляции 2–100 мс) и с какими популяциями пики совпадают.
Переполнение таблицы событий — печать (event_overflow warn).

    PYTHONPATH=. .venv/bin/python tools/burst_trace.py [ms=1000] [pops=…] [seed=0] [параметр=значение …]
"""

import sys

import torch

from brain.config import BrainConfig
from brain.engine import SpikingBrain
from brain.mind import Mind

POPS = ("entorhinal.E,entorhinal.I,dentate.E,ca3.E,ca3.I,ca1.E,ca1.I,gpe.E,gpi.E,stn.E,purkinje.E,thalamus.E,"
        "association.E,association.L4.E,association.L5.E,prefrontal.L5.E,category.L4.E,auditory.L4.E")
opts = {"ms": "1000", "pops": POPS, "seed": "0"}
cfg = BrainConfig()
cfg.save_path = "/nonexistent/brain.pt"
cfg.autosave = False
cfg.anatomy = True
cfg.hand = True
cfg.dt_ms = 0.5
cfg.event_overflow = "warn"

for kv in sys.argv[1:]:
    k, v = kv.split("=", 1)

    if k in opts:
        opts[k] = v
    else:
        old = getattr(cfg, k)
        setattr(cfg, k, (v == "1") if isinstance(old, bool) else eval(v) if isinstance(old, tuple) else type(old)(v))

SpikingBrain.CHUNK = 1
cfg.seed = int(opts["seed"])
torch.manual_seed(cfg.seed)
b = Mind(cfg).brain
b.plasticity = False
b.run(1000.0)
pops = []

for name in opts["pops"].split(","):
    area, label = name.split(".", 1)
    pops.append((name, dict(b.areas[area].populations)[label]))

steps = int(float(opts["ms"]) / cfg.dt_ms)
share = torch.zeros(steps, len(pops))

for t in range(steps):
    r = b.observe(cfg.dt_ms)

    for j, (_, p) in enumerate(pops):
        share[t, j] = (r[p.slice] > 0).float().mean()

print(f"{steps} шагов по {cfg.dt_ms} мс")
print(f"{'популяция':20s} {'клеток':>6s} {'Гц':>6s} {'нет':>5s} {'>10%':>6s} {'>25%':>6s} {'>50%':>6s} {'>90%':>6s}  ритм")
peaks = {}

for j, (name, p) in enumerate(pops):
    x = share[:, j]
    hz = float(x.mean()) * 1000.0 / cfg.dt_ms
    frac = [float((x > q).float().mean()) for q in (0.1, 0.25, 0.5, 0.9)]
    y = x - x.mean()
    ac = torch.stack([(y[:-k] * y[k:]).mean() for k in range(1, int(100 / cfg.dt_ms))]) / y.pow(2).mean().clamp(min=1e-12)
    lag = int(ac[int(2 / cfg.dt_ms):].argmax()) + int(2 / cfg.dt_ms) + 1
    rhythm = f"{lag * cfg.dt_ms:.1f} мс (r {float(ac[lag - 1]):.2f})"
    print(f"{name:20s} {p.size:6d} {hz:6.1f} {float((x == 0).float().mean()):5.2f} " + " ".join(f"{v:6.3f}" for v in frac) + "  " + rhythm)
    peaks[name] = set((x > max(0.25, 10 * float(x.mean()))).nonzero().flatten().tolist())

names = [n for n in peaks if peaks[n]]
print("совпадение пиков (± 1 шаг) — доля пиков первой, рядом с которыми пик второй:")

for a in names:
    near = []

    for c in names:
        if c == a:
            continue
        other = peaks[c] | {t + 1 for t in peaks[c]} | {t - 1 for t in peaks[c]}
        share_ab = len(peaks[a] & other) / len(peaks[a])

        if share_ab >= 0.3:
            near.append(f"{c} {share_ab:.2f}")

    print(f"  {a} ({len(peaks[a])} пиков): " + (", ".join(near) or "—"))
