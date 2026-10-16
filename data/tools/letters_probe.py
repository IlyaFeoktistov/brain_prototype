"""
Различает ли зона букв буквы: буква на экране (взгляд на ней), ответ
клеток L2/3 E зоны букв — вектор частот; сходство ответов на разные буквы
(корреляция векторов) до и после обучения. Учится — сходство падает:
на разные буквы отвечают разные клетки (как избирательность зрительной зоны
словоформы к буквам; Dehaene & Cohen 2011).

    PYTHONPATH=. .venv/bin/python data/tools/letters_probe.py [load=МОЗГ] [letters=аомсткл] [ms=300] [zone=letters]
"""

import sys

import torch

from brain.body import Eye
from brain.config import BrainConfig
from brain.mind import Mind
from brain.world import Screen

opts = {"load": "", "letters": "аомсткл", "ms": "300", "zone": "letters", "rest": "300", "warm": "2500"}

for kv in sys.argv[1:]:
    k, v = kv.split("=", 1)
    opts[k] = v

if opts["load"]:
    mind = Mind.load(opts["load"])
else:
    cfg = BrainConfig()
    cfg.save_path = "/nonexistent/brain.pt"
    cfg.anatomy = True
    cfg.hand = True
    cfg.dt_ms = 0.5
    mind = Mind(cfg)

mind.cfg.autosave = False
cfg, brain = mind.cfg, mind.brain
brain.plasticity = False
i = [s.name for s in brain.sensors].index("eye")
eye = Eye(cfg, brain.sensor_offsets[i], brain.sensors[i].size, brain.device)
screen = Screen(cfg.retina_rows, cfg.retina_cols)
blank = screen.picture("")
brain.observe(float(opts["warm"]), eye.look(blank, float(opts["warm"])))
zone = brain.areas[opts["zone"]]
cells = zone.layers["L23"]["E"].slice if zone.layers else zone.E.slice
responses = []

for letter in opts["letters"]:
    brain.observe(float(opts["rest"]), eye.look(blank, float(opts["rest"])))
    rest = brain.observe(float(opts["rest"]), eye.look(blank, float(opts["rest"])))[cells]
    seen = brain.observe(float(opts["ms"]), eye.look(screen.picture(letter), float(opts["ms"])))[cells]
    responses.append(seen - rest)
    active = int((seen - rest > 5.0).sum())
    print(f"«{letter}»: покой {float(rest.mean()):.2f} Гц, на букву {float(seen.mean()):.2f} Гц, "
          f"клеток с приростом > 5 Гц: {active}", flush=True)

R = torch.stack(responses)
R = R - R.mean(1, keepdim=True)
R = R / R.norm(dim=1, keepdim=True).clamp(min=1e-6)
C = R @ R.t()
n = len(opts["letters"])
off = float((C.sum() - C.trace()) / (n * (n - 1)))
print(f"сходство ответов на разные буквы (средняя корреляция): {off:.3f}  (1 — одинаково, 0 — независимо)")
