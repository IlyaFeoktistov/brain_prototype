"""
Согласование анатомического мозга с частотами in vivo в контексте всего
мозга (ARCHITECTURE 11): фон каждого ядра (BrainConfig.nucleus_bias_mv) и
фон колонок по слою и типу (column_gain_external_pop) подстраиваются шагами
«стреляет больше нормы — меньше фона» (как веса фона по типам у Billeh et
al. 2020). Цели — anatomy.RATES_HZ и column_target_hz.

    PYTHONPATH=. .venv/bin/python data/tools/anatomy_fit.py [steps=20] [ms=500] [dt=0.5]

Печатает числа для config.py / anatomy.py.
"""

import math
import sys

import torch

from brain import anatomy
from brain.config import BrainConfig
from brain.mind import Mind

opts = {"steps": "20", "ms": "500", "dt": "0.5", "size": "small"}

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
    cfg.column_gain_external_pop = tuple((l, t, g) for (l, t), g in pop.items())
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
                columns.setdefault((layer, kind), []).append(float(rates[cells.slice].mean()))

    return nuclei, {key: sum(v) / len(v) for key, v in columns.items()}, cfg


base = BrainConfig()
targets = {(layer, kind): hz for layer, *v in base.column_target_hz for kind, hz in zip(KINDS, v)}
# Старт — уже найденное (anatomy.py), продолжить подбор.
bias: dict = dict(anatomy.NUCLEUS_BIAS_MV)
pop = {(l, t): g for l, t, g in anatomy.COLUMN_EXTERNAL_POP}

for step in range(int(opts["steps"])):
    nuclei, columns, cfg = measure(bias, pop)
    errors = [math.log((r + 0.5) / (anatomy.RATES_HZ[n] + 0.5)) ** 2 for n, r in nuclei.items()]
    errors += [math.log((r + 0.5) / (targets[k] + 0.5)) ** 2 for k, r in columns.items()]
    print(f"шаг {step:2d} ошибка {sum(errors) / len(errors):.3f}  "
          + " ".join(f"{n}={r:.1f}" for n, r in nuclei.items()), flush=True)
    print("        колонки " + " ".join(f"{l}{k}={r:.1f}" for (l, k), r in columns.items()), flush=True)

    for name, r in nuclei.items():
        # Ток: шаг по логарифму отношения частот (1 мВ ≈ заметная доля пути
        # до порога), ограничен ±2 мВ за шаг.
        step_mv = 2.0 * math.log((anatomy.RATES_HZ[name] + 0.5) / (r + 0.5))
        bias[name] = bias.get(name, 0.0) + max(-2.0, min(2.0, step_mv))

    for key, r in columns.items():
        pop[key] *= min(1.3, max(0.77, ((targets[key] + 0.5) / (r + 0.5)) ** 0.25))

print("nucleus_bias_mv =", tuple((n, round(v, 2)) for n, v in bias.items()))
print("column_gain_external_pop =", tuple((l, t, round(g, 3)) for (l, t), g in pop.items()))
