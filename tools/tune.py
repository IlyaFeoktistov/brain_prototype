"""
Тестер-подгонщик анатомического мозга: проверки с диапазонами из данных и
разрешённые ручки калибровки. Мозг собирается один раз, ручки меняются на
ходу (фон популяций — noise_gain, фон ядер — bias); правятся только
популяции вне своего диапазона; шаг доверительный: ухудшение больше чем на
treshold — откат к лучшему, сброс состояния сети и шаг вдвое меньше.
Останавливается, когда все проверки в диапазоне.

    PYTHONPATH=. .venv/bin/python tools/tune.py [iters=40] [knobs=e|all] [write=1] [band=2]

Проверки (версия 1 — покой наяву):
  · частота каждой популяции колонок (кроме LAMP5: in vivo не измерена) и
    каждого ядра — в [цель / band, цель × band]; цели — column_target_hz и
    anatomy.RATES_HZ (их источники — там); band 2 — разброс средних между
    работами и состояниями (например, L2/3 E человека 2.6 ± 3.1 Гц — Le Van
    Quyen et al. 2016);
  · ни одна популяция не выше 150 Гц (у бодрствующей коры таких средних нет).
Ручки: фон E колонок по зоне и слою (knobs=e) или всех типов, кроме LAMP5
(knobs=all), и фон ядер (кроме узла сна — его согласует sleep_fit).
Найденное пишется в brain/anatomy.py (write=1) после каждого лучшего замера.
"""

import math
import sys
import time

import torch

from brain import anatomy
from brain.config import BrainConfig
from brain.mind import Mind

opts = {"iters": "40", "knobs": "e", "write": "1", "band": "2", "warm": "2500", "settle": "300", "ms": "1000",
        "cap": "1.1", "cap_mv": "2", "worse": "0.15", "dt": "0.5", "size": "small"}

for kv in sys.argv[1:]:
    k, v = kv.split("=", 1)
    opts[k] = v

KINDS = ("E", "PV", "SOM", "VIP", "LAMP5")
torch.manual_seed(0)
cfg = BrainConfig()
cfg.save_path = "/nonexistent/brain.pt"
cfg.autosave = False
cfg.anatomy = True
cfg.hand = True
cfg.size = opts["size"]
cfg.dt_ms = float(opts["dt"])
bias = dict(anatomy.NUCLEUS_BIAS_MV)
pop = {tuple(e[:3]): e[3] for e in anatomy.COLUMN_EXTERNAL_POP if len(e) == 4}
cfg.nucleus_bias_mv = tuple(bias.items())
cfg.column_gain_external_pop = tuple((*key, g) for key, g in pop.items())
start = time.time()
mind = Mind(cfg)
brain = mind.brain
brain.plasticity = False
print(f"мозг собран за {time.time() - start:.0f} с, клеток {brain.n}", flush=True)
targets = {(layer, kind): hz for layer, *v in cfg.column_target_hz for kind, hz in zip(KINDS, v)}
band = float(opts["band"])

# Популяции: (имя, клетки, цель или None, ручка).
checks = []

for name, area in brain.areas.items():
    if area.layers is None:
        if name in anatomy.RATES_HZ:
            knob = None if name in anatomy.SLEEP_RATES_HZ else ("bias", name)
            checks.append((name, area.E.slice, anatomy.RATES_HZ[name], knob))
        continue

    for layer, group in area.layers.items():
        for kind, cells in group.items():
            aim = None if kind == "LAMP5" else targets[(layer, kind)]
            tuned = kind == "E" or (opts["knobs"] == "all" and kind != "LAMP5")
            key = (name, layer, kind)
            pop.setdefault(key, 1.0)
            checks.append((f"{name}.{layer}{kind}", cells.slice, aim, ("pop", key) if tuned else None))


def measure() -> dict:
    brain.run(float(opts["settle"]))
    rates = brain.observe(float(opts["ms"]))
    return {name: float(rates[cells].mean()) for name, cells, _, _ in checks}


def outside(r: float, aim: float) -> float:
    """На сколько (в логарифме) частота вне [aim/band, aim·band]; внутри — 0."""

    x = math.log((r + 0.5) / (aim + 0.5))
    edge = math.log(band)
    return max(0.0, abs(x) - edge) * (1 if x > 0 else -1)


def loss(rates: dict) -> tuple[float, list]:
    errs, fails = [], []

    for name, _, aim, _ in checks:
        r = rates[name]

        if r > 150.0:
            errs.append(9.0)
            fails.append((name, r, aim, "> 150 Гц"))
        elif aim is not None:
            d = outside(r, aim)
            errs.append(d * d)

            if d != 0:
                fails.append((name, r, aim, ""))

    return sum(errs) / len(errs), fails


def apply(knob, factor: float = 1.0, mv: float = 0.0) -> None:
    kind, key = knob

    if kind == "pop":
        zone, layer, t = key
        cells = brain.areas[zone].layers[layer][t].slice
        brain.noise_gain[cells] *= factor
        pop[key] *= factor
    else:
        brain.bias[brain.areas[key].E.slice] += mv
        bias[key] = bias.get(key, 0.0) + mv


def reset() -> None:
    """Сеть — в покой (после отката из сорвавшегося состояния)."""

    for name in brain.areas:
        brain.silence(name)

    brain.run(1000.0)


brain.run(float(opts["warm"]))
best = (math.inf, dict(bias), dict(pop))
cap, cap_mv = float(opts["cap"]), float(opts["cap_mv"])

for it in range(int(opts["iters"])):
    rates = measure()
    value, fails = loss(rates)
    print(f"{it:3d} ошибка {value:.3f}  вне диапазона {len(fails)}/{len(checks)}  шаг ×{cap:.3f}/{cap_mv:.2f} мВ  "
          f"[{time.time() - start:.0f} с]", flush=True)
    print("     " + " ".join(f"{n}={r:.1f}/{a if a is not None else '—'}{w}"
                             for n, r, a, w in sorted(fails, key=lambda f: -abs(outside(f[1], f[2] or f[1])) - (9 if f[3] else 0))[:10]),
          flush=True)

    if not fails:
        print("все проверки в диапазоне", flush=True)
        best = (value, dict(bias), dict(pop))
        break

    if value < best[0]:
        first = best[0] == math.inf
        best = (value, dict(bias), dict(pop))

        # Первый замер — старт (он и так в файле), писать только улучшения.
        if opts["write"] == "1" and not first:
            anatomy.write_fitted(tuple(bias.items()), tuple((*k, round(g, 3)) for k, g in pop.items()))
    elif value > best[0] + float(opts["worse"]):
        # Хуже лучшего больше шума замера (worse — ~двойной разброс ошибки
        # между замерами по 1–2 с) — откат ручек, сеть в покой, шаг вдвое
        # меньше. Порог в долях лучшего (×1.3) давал ложные откаты.
        print(f"     откат к лучшему ({best[0]:.3f}), шаг вдвое меньше", flush=True)

        for key, g in best[2].items():
            if pop[key] != g:
                apply(("pop", key), factor=g / pop[key])

        for key, mv in best[1].items():
            if bias.get(key, 0.0) != mv:
                apply(("bias", key), mv=mv - bias.get(key, 0.0))

        cap, cap_mv = 1.0 + (cap - 1.0) / 2, cap_mv / 2
        reset()
        continue

    # Правка только тех, кто вне диапазона: к ближней границе.
    for name, _, aim, knob in checks:
        if knob is None or aim is None:
            continue

        r = rates[name]
        d = outside(r, aim)

        if d == 0 and r <= 150.0:
            continue

        if knob[0] == "pop":
            factor = math.exp(-0.5 * d) if r <= 150.0 else 1.0 / cap
            apply(knob, factor=min(cap, max(1.0 / cap, factor)))
        else:
            apply(knob, mv=min(cap_mv, max(-cap_mv, -2.0 * d)))

value, bias_best, pop_best = best
print(f"лучшее: ошибка {value:.3f}")

if opts["write"] == "1":
    path = anatomy.write_fitted(tuple(bias_best.items()), tuple((*k, round(g, 3)) for k, g in pop_best.items()))
    print(f"записано в {path}")
