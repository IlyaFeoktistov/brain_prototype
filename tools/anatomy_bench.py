"""
Анатомический мозг в покое (ARCHITECTURE 11): новый мозг с cfg.anatomy живёт
без входа; частота каждого ядра (главные клетки) и каждого слоя колонок
сравнивается с записями у бодрствующих (anatomy.RATES_HZ,
BrainConfig.column_target_hz). Первый стенд «по кусочкам»: что молчит, что
перевозбуждено.

    PYTHONPATH=. .venv/bin/python tools/anatomy_bench.py [size=small] [ms=1000] [key=value ...]
"""

import sys

from brain import anatomy
from brain.config import BrainConfig
from brain.engine import KINDS
from brain.mind import Mind

opts = {"size": "small", "ms": "1000"}
cfg = BrainConfig()
cfg.save_path = "/nonexistent/brain.pt"
cfg.autosave = False
cfg.anatomy = True
cfg.hand = True

for kv in sys.argv[1:]:
    k, v = kv.split("=", 1)

    if k in opts:
        opts[k] = v
    else:
        old = getattr(cfg, k)
        setattr(cfg, k, (v == "1") if isinstance(old, bool) else type(old)(v))

cfg.size = opts["size"]
mind = Mind(cfg)
brain = mind.brain
print(f"{brain.n:,} нейронов, {brain.synapse_count():,} синапсов, зон {len(brain.areas)}", flush=True)
brain.plasticity = False
brain.run(300.0)
rates = brain.observe(float(opts["ms"]))
targets = {layer: dict(zip(KINDS, v)) for layer, *v in cfg.column_target_hz}


def mark(r: float, t: float) -> str:
    if t <= 0:
        return ""
    ratio = (r + 0.5) / (t + 0.5)
    return "  ок" if 0.5 <= ratio <= 2 else ("  МАЛО" if ratio < 0.5 else "  МНОГО")


print("\nядра (главные клетки), Гц — у нас / запись:")

for name, area in brain.areas.items():
    if area.layers is not None:
        continue

    r = float(rates[area.E.slice].mean())
    t = anatomy.RATES_HZ.get(name)
    print(f"  {name:12s} {r:6.1f}" + (f" / {t:<5}{mark(r, t)}" if t is not None else "   (нет записи)") + f"   {area.title}")

print("\nколонки, E / PV / SOM / VIP по слоям, Гц (цель — column_target_hz):")

for name, area in brain.areas.items():
    if area.layers is None:
        continue

    parts = []

    for layer, group in area.layers.items():
        parts.append(f"{layer} " + "/".join(f"{float(rates[pop.slice].mean()):.1f}" for pop in group.values()))

    print(f"  {name:12s} " + " | ".join(parts))

print("  цель         " + " | ".join(f"{layer} " + "/".join(f"{v:g}" for v in t.values()) for layer, t in targets.items()))
