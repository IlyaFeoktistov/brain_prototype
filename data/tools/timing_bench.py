"""
Часы последовательности (зона «timing»): толчок группе 0 — бегут ли группы
по очереди и с какой скоростью (мс на группу), не гаснет ли и не
разгорается ли цепочка.

    PYTHONPATH=. .venv/bin/python data/tools/timing_bench.py [timing_link=6] [key=value ...]
"""

import sys

import torch

from brain.config import BrainConfig
from brain.mind import Mind

cfg = BrainConfig()
cfg.size = "small"
cfg.save_path = "/nonexistent/brain.pt"
cfg.mouth_letters = True
cfg.letter_ms = 25.0

for kv in sys.argv[1:]:
    k, v = kv.split("=", 1)
    old = getattr(cfg, k)
    setattr(cfg, k, type(old)(v))

mind = Mind(cfg)
mind.calibrate()
brain = mind.brain
T = brain.areas["timing"].E
g = cfg.timing_cells
first: dict[int, float] = {}
seen = []


def chunk(fired, t):
    cells = fired[(fired >= T.start) & (fired < T.stop)] - T.start
    groups = sorted(set((cells // g).tolist()))
    seen.append((t, len(cells), groups[:3], groups[-3:] if groups else []))

    for grp in groups:
        first.setdefault(grp, t)


with brain.frozen():
    brain.run(200.0)
    brain.on_chunk = chunk
    start = brain.time_ms
    brain.kick("timing", torch.arange(g, device=brain.device), cfg.timing_kick_mv, cfg.timing_kick_ms)
    brain.run(400.0)
    brain.on_chunk = None

for t, n, lo, hi in seen[:12]:
    print(f"  {t - start:5.0f} мс: клеток {n:3d}, группы {lo} … {hi}")

groups = T.size // g
reached = sorted(first)
print(f"групп {groups}; дошла до {reached[-1] if reached else -1}; "
      f"скорость ~{(first[reached[-1]] - start) / max(1, reached[-1]):.1f} мс на группу" if reached else "цепочка молчит")
