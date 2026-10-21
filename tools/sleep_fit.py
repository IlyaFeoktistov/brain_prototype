"""
Согласование переключателя сна по двум состояниям (дополнение к
anatomy_fit, который видит одно состояние и потому сбивает двустабильный
узел то в сон, то в бодрствование): фон ядер узла (VLPO, LC, шов, орексин,
Мейнерт) подстраивается так, чтобы днём при малом аденозине частоты были
как наяву (anatomy.RATES_HZ), а ночью при большом — как в медленном сне
(anatomy.SLEEP_RATES_HZ). Каждое состояние — с одного и того же начала.

    PYTHONPATH=. .venv/bin/python tools/sleep_fit.py [steps=10] [wake=0.3] [sleep=0.7] [write=1]

wake, sleep — занятость рецепторов аденозина в двух замерах (полдень и
ночь; 0.5 — вечер полного дня, adenosine_half). В конце записывает фон
этих ядер в NUCLEUS_BIAS_MV (brain/anatomy.py); write=0 — только печать.
"""

import math
import sys

import torch

from brain import anatomy
from brain.config import BrainConfig
from brain.mind import Mind

opts = {"steps": "10", "wake": "0.3", "sleep": "0.7", "warm": "1500", "ms": "1000", "write": "1"}

for kv in sys.argv[1:]:
    k, v = kv.split("=", 1)
    opts[k] = v

torch.manual_seed(0)
cfg = BrainConfig()
cfg.save_path = "/nonexistent/brain.pt"
cfg.autosave = False
cfg.size = "small"
cfg.dt_ms = 0.5
mind = Mind(cfg)
brain = mind.brain
brain.plasticity = False
brain.run(200.0)
names = list(anatomy.SLEEP_RATES_HZ)
bias = dict(anatomy.NUCLEUS_BIAS_MV)
adenosine = brain.chemistry_names.index("adenosine")
states = {"наяву": (float(opts["wake"]), anatomy.RATES_HZ, 14.0),
          "во сне": (float(opts["sleep"]), anatomy.SLEEP_RATES_HZ, 2.0)}


def measure(occupancy: float, hour: float) -> dict:
    level = occupancy * cfg.adenosine_half / (1.0 - occupancy)

    with brain.frozen():
        brain.time_ms = ((hour - cfg.birth_hour) % 24.0) * 3_600_000.0
        brain.chemistry[adenosine] = level
        brain.run(float(opts["warm"]))
        brain.chemistry[adenosine] = level
        rates = brain.observe(float(opts["ms"]))

    return {n: float(rates[brain.areas[n].E.slice].mean()) for n in names}


for step in range(int(opts["steps"])):
    logs = {n: [] for n in names}
    line = []

    for label, (occupancy, targets, hour) in states.items():
        rates = measure(occupancy, hour)

        for n in names:
            logs[n].append(math.log((targets[n] + 0.5) / (rates[n] + 0.5)))

        line.append(f"{label}: " + " ".join(f"{n}={rates[n]:.1f}/{targets[n]:g}" for n in names))

    error = sum(x * x for xs in logs.values() for x in xs) / (2 * len(names))
    print(f"шаг {step:2d} ошибка {error:.3f}  " + "  |  ".join(line), flush=True)

    for n in names:
        # Как anatomy_fit: шаг по логарифму отношения частот, среднее по
        # двум состояниям, не больше ±2 мВ.
        mv = max(-2.0, min(2.0, 2.0 * sum(logs[n]) / len(logs[n])))
        bias[n] = bias.get(n, 0.0) + mv
        brain.bias[brain.areas[n].E.slice] += mv

print("фон узла сна, мВ:", {n: round(bias[n], 2) for n in names})

if opts["write"] == "1":
    path = anatomy.write_fitted(tuple(bias.items()), anatomy.COLUMN_EXTERNAL_POP)
    print(f"записано в {path}")
