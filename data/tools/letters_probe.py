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

opts = {"load": "", "letters": "аомсткл", "ms": "300", "zone": "letters", "layer": "L23", "rest": "300", "warm": "2500"}

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
# Слои по ходу сигнала: V1 L4 → V1 L2/3 → буквы L4 → буквы L2/3 (или zone/layer).
stages = [("visual", "L4"), ("visual", "L23"), ("letters", "L4"), ("letters", "L23")]

if opts.get("only"):
    stages = [(opts["zone"], opts["layer"])]

cells = {f"{z}.{l}": brain.areas[z].layers[l]["E"].slice for z, l in stages}
T = float(opts["ms"]) / 1000.0
repeats = int(opts.get("repeats", "3"))
shown = []
z_resp = {name: [] for name in cells}

for rep in range(repeats):
    for letter in opts["letters"]:
        brain.observe(float(opts["rest"]), eye.look(blank, float(opts["rest"])))
        rest = brain.observe(float(opts["rest"]), eye.look(blank, float(opts["rest"])))
        seen = brain.observe(float(opts["ms"]), eye.look(screen.picture(letter), float(opts["ms"])))
        shown.append(letter)

        for name, sl in cells.items():
            # z по Пуассону: разброс счёта за окно — √(частота покоя · T).
            z_resp[name].append((seen[sl] - rest[sl]) * T / ((rest[sl] + 0.5) * T).sqrt())

    print(f"показ {rep + 1}/{repeats}", flush=True)

# Информация о букве: сходство ответов на одну букву (разные показы) против
# сходства на разные буквы. Равны — зона о букве не знает.
same_mask = torch.tensor([[a == b for b in shown] for a in shown])
eye_mask = torch.eye(len(shown), dtype=torch.bool)

for name, zs in z_resp.items():
    R = torch.stack(zs).float().cpu()
    R = R - R.mean(1, keepdim=True)
    R = R / R.norm(dim=1, keepdim=True).clamp(min=1e-6)
    C = R @ R.t()
    same = float(C[same_mask & ~eye_mask].mean())
    other = float(C[~same_mask].mean())
    print(f"{name}: одна буква {same:.3f}, разные {other:.3f}, разница {same - other:+.3f}")
