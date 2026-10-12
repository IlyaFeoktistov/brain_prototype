"""
Переключатель сна: кто стреляет в разное время суток при разном аденозине —
часы (SCN), голубое пятно (бодрствование) или VLPO (сон).

    PYTHONPATH=. .venv/bin/python data/tools/sleep_switch.py [МОЗГ] [key=value ...]

Без МОЗГ — новый мозг small (без чтения: переключателю слова не нужны).
key=value — поля конфига на пробу (receptors не задать — правьте конфиг).
Мозг заморожен: веса и состояние после замера возвращаются.
"""

import sys

from brain.config import BrainConfig
from brain.mind import Mind

args = [a for a in sys.argv[1:] if "=" not in a]
overrides = [a.split("=", 1) for a in sys.argv[1:] if "=" in a]

if args:
    mind = Mind.load(args[0])
    cfg = mind.cfg
else:
    cfg = BrainConfig()
    cfg.size = "small"
    cfg.autosave = False
    cfg.save_path = "/nonexistent/brain.pt"

for k, v in overrides:
    old = getattr(cfg, k)
    setattr(cfg, k, type(old)(v))
    print(f"{k} = {getattr(cfg, k)}")

if not args:
    mind = Mind(cfg)

cfg.autosave = False
brain = mind.brain
mind.brain.recompile()
adenosine = brain.chemistry_names.index("adenosine")
half = cfg.adenosine_half


def rate(rates, area):
    E = brain.areas[area].E
    return float(rates[E.slice].mean())


print(f"{'час':>4} {'аденозин':>9} | {'SCN Гц':>7} {'LC Гц':>7} {'VLPO Гц':>8} | спит?")

for hour in (8.0, 10.0, 16.0, 22.0, 2.0):
    for occupancy in (0.05, 0.3, 0.5, 0.7, 0.9):
        with brain.frozen():
            brain.time_ms = ((hour - cfg.birth_hour) % 24.0) * 3_600_000.0
            level = occupancy * half / (1.0 - occupancy)
            brain.chemistry[adenosine] = level
            brain.run(1000.0)
            # Аденозин за секунду не меняется — держим тот же уровень.
            brain.chemistry[adenosine] = level
            rates = brain.observe(2000.0)
            scn, lc, vlpo = rate(rates, "scn"), rate(rates, "lc"), rate(rates, "vlpo")
            print(f"{hour:4.0f} {occupancy:9.2f} | {scn:7.2f} {lc:7.2f} {vlpo:8.2f} | {'да' if vlpo > lc else 'нет'}")
