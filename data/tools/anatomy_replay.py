"""
Собрать анатомический мозг с числами шага N подбора (anatomy_fit пишет их в
trace) и смотреть зоны по окнам: что загорается первым, когда шаг сломан.

    PYTHONPATH=. .venv/bin/python data/tools/anatomy_replay.py step=3 [trace=data/measure/logs/fit-steps.jsonl] [win=50] [ms=3500] [zones=hand,thalamus,gpi]

zones — ядра (вся зона) и колонки (каждый слой E и PV).
"""

import json
import sys

import torch

from brain.config import BrainConfig
from brain.mind import Mind

opts = {"step": "0", "trace": "data/measure/logs/fit-steps.jsonl", "win": "50", "ms": "3500", "zones": "",
        "dt": "0.5", "size": "small"}

for kv in sys.argv[1:]:
    k, v = kv.split("=", 1)
    opts[k] = v

rows = [json.loads(line) for line in open(opts["trace"], encoding="utf-8")]
row = next(r for r in rows if r["step"] == int(opts["step"]))
torch.manual_seed(0)
cfg = BrainConfig()
cfg.save_path = "/nonexistent/brain.pt"
cfg.autosave = False
cfg.anatomy = True
cfg.hand = True
cfg.size = opts["size"]
cfg.dt_ms = float(opts["dt"])
cfg.nucleus_bias_mv = tuple(row["bias"].items())
cfg.column_gain_external_pop = tuple(tuple(p) for p in row["pop"])
if row.get("from_e"):
    cfg.column_gain_from_e = tuple(row["from_e"])
brain = Mind(cfg).brain
brain.plasticity = False
cols = []

for name in opts["zones"].split(","):
    area = brain.areas[name]

    if area.layers is None:
        cols.append((name, area.E))
        continue

    for layer, group in area.layers.items():
        cols += [(f"{name[:5]}{layer}E", group["E"]), (f"{name[:5]}{layer}PV", group["PV"])]

print(f"шаг {row['step']}, ошибка {row['error']:.3f}")
print("  мс   " + " ".join(f"{c[:9]:>9s}" for c, _ in cols))
win = float(opts["win"])

for w in range(int(float(opts["ms"]) // win)):
    rates = brain.observe(win)
    print(f"{(w + 1) * win:5.0f} " + " ".join(f"{float(rates[p.slice].mean()):9.1f}" for _, p in cols), flush=True)
