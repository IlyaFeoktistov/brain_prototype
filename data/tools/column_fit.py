"""
Согласование колонки с частотами in vivo (ARCHITECTURE 10, «Согласование»):
одна колонка (слуховая зона, без других зон) живёт в покое; 11 общих на тип
множителей (BrainConfig.column_gain_*: внешний вход E / PV / SOM / VIP, E →
тип, тип → E, I → I) подбираются так, чтобы частоты по слоям и типам
совпали с column_target_hz. Поиск — Нелдер–Мид по логарифмам множителей,
ошибка — средний квадрат log((r + 0.5) / (цель + 0.5)) по 16 популяциям.

    PYTHONPATH=. .venv/bin/python data/tools/column_fit.py [neurons=10000] [evals=300] [ms=1000]

Печатает лучшие множители — их переносят в config.py с пометкой.
"""

import math
import sys

import torch

from brain.config import BrainConfig, default_areas
from brain.engine import SpikingBrain

opts = {"neurons": "10000", "evals": "300", "ms": "1000", "area": "auditory", "mode": "gains", "steps": "60", "dt": "2.0"}

for kv in sys.argv[1:]:
    k, v = kv.split("=", 1)
    opts[k] = v

TYPES = ("E", "PV", "SOM", "VIP")


def build(gains: list[float], pop: dict | None = None) -> SpikingBrain:
    cfg = BrainConfig()
    cfg.layers = True
    cfg.eye = False
    cfg.compile_step = False
    cfg.size = opts["neurons"]
    cfg.dt_ms = float(opts["dt"])
    cfg.areas = [a for a in default_areas() if a.name == opts["area"]]
    cfg.pathways = []
    cfg.column_gain_external = tuple(gains[0:4])
    cfg.column_gain_from_e = tuple(gains[4:7])
    cfg.column_gain_to_e = tuple(gains[7:10])
    cfg.column_gain_ii = gains[10]
    cfg.column_gain_external_pop = tuple((l, t, g) for (l, t), g in (pop or {}).items())
    return SpikingBrain(cfg)


def measure(gains: list[float], pop: dict | None = None) -> dict:
    torch.manual_seed(0)
    brain = build(gains, pop)
    brain.plasticity = False
    brain.run(300.0)
    rates = brain.observe(float(opts["ms"]))
    area = brain.areas[opts["area"]]
    return {(layer, kind): float(rates[pop.slice].mean()) for layer, group in area.layers.items() for kind, pop in group.items()}


def loss(rates: dict, targets: dict) -> float:
    errors = [math.log((r + 0.5) / (targets[key] + 0.5)) ** 2 for key, r in rates.items()]
    return sum(errors) / len(errors)


cfg = BrainConfig()
targets = {(layer, kind): hz for layer, *v in cfg.column_target_hz for kind, hz in zip(TYPES, v)}
calls = [0]
best = [math.inf, None, None]


def f(x: list[float]) -> float:
    gains = [math.exp(v) for v in x]
    rates = measure(gains)
    value = loss(rates, targets)
    calls[0] += 1

    if value < best[0]:
        best[:] = [value, gains, rates]
        print(f"{calls[0]:4d} ошибка {value:.3f}  " + " ".join(f"{g:.2f}" for g in gains), flush=True)

    return value


def nelder_mead(f, x0: list[float], step: float, evals: int) -> None:
    n = len(x0)
    simplex = [list(x0)] + [[x0[j] + (step if j == i else 0.0) for j in range(n)] for i in range(n)]
    values = [f(x) for x in simplex]

    while calls[0] < evals:
        order = sorted(range(n + 1), key=lambda i: values[i])
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        centroid = [sum(x[j] for x in simplex[:-1]) / n for j in range(n)]
        worst = simplex[-1]
        reflected = [c + (c - w) for c, w in zip(centroid, worst)]
        fr = f(reflected)

        if fr < values[0]:
            expanded = [c + 2 * (c - w) for c, w in zip(centroid, worst)]
            fe = f(expanded)
            simplex[-1], values[-1] = (expanded, fe) if fe < fr else (reflected, fr)
        elif fr < values[-2]:
            simplex[-1], values[-1] = reflected, fr
        else:
            contracted = [c + 0.5 * (w - c) for c, w in zip(centroid, worst)]
            fc = f(contracted)

            if fc < values[-1]:
                simplex[-1], values[-1] = contracted, fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = [simplex[0][j] + 0.5 * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                    values[i] = f(simplex[i])


if opts["mode"] == "pop":
    # Фон по популяциям (Billeh et al. 2020): множитель внешнего входа
    # каждой популяции × (цель / частота)^0.3 — «стреляет больше нормы —
    # меньше фона»; общие множители — из config.py.
    base = BrainConfig()
    gains = list(base.column_gain_external) + list(base.column_gain_from_e) + list(base.column_gain_to_e) + [base.column_gain_ii]
    pop = {key: 1.0 for key in targets}

    for step in range(int(opts["steps"])):
        rates = measure(gains, pop)
        value = loss(rates, targets)
        print(f"{step:3d} ошибка {value:.3f}  " + " ".join(f"{k[0]}{k[1]}={r:.1f}" for k, r in rates.items()), flush=True)

        if value < best[0]:
            best[:] = [value, gains, rates]
            best_pop = dict(pop)

        for key, r in rates.items():
            pop[key] *= min(1.5, max(0.67, ((targets[key] + 0.5) / (r + 0.5)) ** 0.3))

    print("column_gain_external_pop =", tuple((l, t, round(g, 3)) for (l, t), g in best_pop.items()))
else:
    start = [0.0] * 11
    nelder_mead(f, start, 0.7, int(opts["evals"]))

value, gains, rates = best
print(f"\nлучшее: ошибка {value:.3f}")
print("column_gain_external =", tuple(round(g, 3) for g in gains[0:4]))
print("column_gain_from_e =", tuple(round(g, 3) for g in gains[4:7]))
print("column_gain_to_e =", tuple(round(g, 3) for g in gains[7:10]))
print("column_gain_ii =", round(gains[10], 3))

for layer, *_ in cfg.column_target_hz:
    print(f"  {layer}: " + ", ".join(f"{kind} {rates[(layer, kind)]:.1f} (цель {targets[(layer, kind)]})" for kind in TYPES))
