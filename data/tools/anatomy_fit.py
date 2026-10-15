"""
Согласование анатомического мозга с частотами in vivo в контексте всего
мозга (ARCHITECTURE 11): фон каждого ядра (BrainConfig.nucleus_bias_mv) и
фон колонок по слою и типу (column_gain_external_pop) подстраиваются шагами
«стреляет больше нормы — меньше фона» (как веса фона по типам у Billeh et
al. 2020). Цели — anatomy.RATES_HZ и column_target_hz.

    PYTHONPATH=. .venv/bin/python data/tools/anatomy_fit.py [steps=20] [ms=500] [dt=0.5] [fresh=0] [write=1]

Стартует с чисел brain/anatomy.py (fresh=1 — фон колонок с нуля), в конце
печатает найденное и записывает его туда же (NUCLEUS_BIAS_MV,
COLUMN_EXTERNAL_POP); write=0 — только напечатать.
"""

import math
import sys

import torch

from brain import anatomy
from brain.config import BrainConfig
from brain.mind import Mind

opts = {"steps": "20", "ms": "500", "dt": "0.5", "size": "small", "fresh": "0", "write": "1"}

for kv in sys.argv[1:]:
    k, v = kv.split("=", 1)
    opts[k] = v

KINDS = ("E", "PV", "SOM", "VIP")


def measure(bias: dict, pop: dict) -> tuple[dict, dict, dict]:
    torch.manual_seed(0)
    cfg = BrainConfig()
    cfg.save_path = "/nonexistent/brain.pt"
    cfg.autosave = False
    cfg.anatomy = True
    cfg.hand = True
    cfg.size = opts["size"]
    cfg.dt_ms = float(opts["dt"])
    cfg.nucleus_bias_mv = tuple(bias.items())
    cfg.column_gain_external_pop = tuple((*key, g) for key, g in pop.items())
    mind = Mind(cfg)
    brain = mind.brain
    brain.plasticity = False
    brain.run(200.0)
    rates = brain.observe(float(opts["ms"]))
    nuclei = {name: float(rates[area.E.slice].mean()) for name, area in brain.areas.items()
              if area.layers is None and name in anatomy.RATES_HZ}
    columns: dict = {}

    for area in brain.areas.values():
        if area.layers is None:
            continue

        for layer, group in area.layers.items():
            for kind, cells in group.items():
                columns[(area.name, layer, kind)] = float(rates[cells.slice].mean())

    return nuclei, columns, cfg


base = BrainConfig()
targets = {(layer, kind): hz for layer, *v in base.column_target_hz for kind, hz in zip(KINDS, v)}
# Старт — уже найденное (anatomy.py), продолжить подбор.
bias: dict = dict(anatomy.NUCLEUS_BIAS_MV)
# Фон колонок — свой у каждой зоны (зона, слой, тип); старт — общий по
# (слой, тип) из anatomy.py.
common = {(e[0], e[1]): e[-1] for e in anatomy.COLUMN_EXTERNAL_POP if len(e) == 3}
own = {tuple(e[:3]): e[-1] for e in anatomy.COLUMN_EXTERNAL_POP if len(e) == 4}
pop: dict = {}

if opts["fresh"] == "1":
    # Фон колонок — с нуля (после изменения устройства входов).
    common, own = {}, {}

for step in range(int(opts["steps"])):
    nuclei, columns, cfg = measure(bias, pop)

    if not pop:
        pop = {key: own.get(key, common.get(key[1:], 1.0)) for key in columns}
        nuclei, columns, cfg = measure(bias, pop)
    errors = [math.log((r + 0.5) / (anatomy.RATES_HZ[n] + 0.5)) ** 2 for n, r in nuclei.items()]
    errors += [math.log((r + 0.5) / (targets[k[1:]] + 0.5)) ** 2 for k, r in columns.items()]
    print(f"шаг {step:2d} ошибка {sum(errors) / len(errors):.3f}  "
          + " ".join(f"{n}={r:.1f}" for n, r in nuclei.items()), flush=True)
    worst = sorted(columns.items(), key=lambda kv: -abs(math.log((kv[1] + 0.5) / (targets[kv[0][1:]] + 0.5))))[:8]
    print("        хуже всех: " + " ".join(f"{a}.{l}{k}={r:.1f}/{targets[(l, k)]}" for (a, l, k), r in worst), flush=True)

    for name, r in nuclei.items():
        # Ток: шаг по логарифму отношения частот (1 мВ ≈ заметная доля пути
        # до порога), ограничен ±2 мВ за шаг.
        step_mv = 2.0 * math.log((anatomy.RATES_HZ[name] + 0.5) / (r + 0.5))
        bias[name] = bias.get(name, 0.0) + max(-2.0, min(2.0, step_mv))

    for key, r in columns.items():
        pop[key] *= min(1.3, max(0.77, ((targets[key[1:]] + 0.5) / (r + 0.5)) ** 0.25))

print("nucleus_bias_mv =", tuple((n, round(v, 2)) for n, v in bias.items()))
print("column_gain_external_pop =", tuple((*key, round(g, 3)) for key, g in pop.items()))

if opts["write"] == "1":
    # Записать найденное в brain/anatomy.py вместо прежних чисел.
    from pathlib import Path

    path = Path(anatomy.__file__)
    text = path.read_text(encoding="utf-8")
    start = text.index("NUCLEUS_BIAS_MV = ")
    end = text.index("\n", text.index("COLUMN_EXTERNAL_POP = ")) + 1
    found = (
        f"NUCLEUS_BIAS_MV = {tuple((n, round(v, 2)) for n, v in bias.items())}\n"
        f"COLUMN_EXTERNAL_POP = {tuple((*key, round(g, 3)) for key, g in pop.items())}\n"
    )
    path.write_text(text[:start] + found + text[end:], encoding="utf-8")
    print(f"записано в {path}")
