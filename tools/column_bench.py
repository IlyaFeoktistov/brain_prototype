"""
Колонка коры в покое (ARCHITECTURE 10, «Слои колонки»): новый мозг с
cfg.layers живёт без входа, частоты по слоям и типам клеток сравниваются с
измерениями у бодрствующих животных и с моделью Potjans & Diesmann 2014
(Table 6: L2/3 E < 1 Гц, L4 E ~1–4, L5 E — больше всех, L6 E ~0.5–1).
Плюс нерегулярность: CV межспайковых интервалов (~0.7–1.2 в покое у коры).

    PYTHONPATH=. .venv/bin/python tools/column_bench.py [size=small] [ms=2000] [key=value ...]
"""

import sys

import torch

from brain.config import BrainConfig
from brain.mind import Mind

opts = {"size": "small", "ms": "2000"}
cfg = BrainConfig()
cfg.save_path = "/nonexistent/brain.pt"
cfg.autosave = False
cfg.layers = True

for kv in sys.argv[1:]:
    k, v = kv.split("=", 1)

    if k in opts:
        opts[k] = v
    else:
        old = getattr(cfg, k)
        setattr(cfg, k, (v == "1") if isinstance(old, bool) else type(old)(v))

cfg.size = opts["size"]
mind = Mind(cfg)
brain = mind.brain
print(f"{brain.n:,} нейронов, {brain.synapse_count():,} синапсов", flush=True)

# Прогрев — без учёта; потом окно покоя.
brain.plasticity = False
brain.run(300.0)
rates = brain.observe(float(opts["ms"]))
reference = {"L23": 0.86, "L4": 4.45, "L5": 7.59, "L6": 1.09}

for name, area in brain.areas.items():
    if area.layers is None:
        continue

    line = []

    for layer, group in area.layers.items():
        parts = []

        for kind, pop in group.items():
            r = rates[pop.slice]
            parts.append(f"{kind} {float(r.mean()):.1f}")

        parts[0] += f" (Potjans {reference[layer]})"
        line.append(f"{layer}: " + ", ".join(parts))

    print(f"{name:12s} " + " | ".join(line), flush=True)

dc = brain.dc_mv[: brain.n]
print(f"ток недостающих соседей: от {float(dc.min()):.1f} до {float(dc.max()):.1f} мВ")
