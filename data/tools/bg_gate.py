"""
Ворота «говори» из базальных ганглиев: что стреляет на последнем слове
вопроса (клетки окончания звука → стриатум D1/D2 → GPe/GPi → таламус →
смысловая кора) и какой ответ — против ровного тока кодом на том же мозге.

    PYTHONPATH=. .venv/bin/python data/tools/bg_gate.py [МОЗГ] [key=value ...]

Без МОЗГ — новый мозг small читает train-data.md один раз (без сна).
Мозг заморожен на каждый вопрос: ответы не учат его.
"""

import sys
from pathlib import Path

from brain.config import BrainConfig
from brain.mind import Mind

args = [a for a in sys.argv[1:] if "=" not in a]
overrides = [a.split("=", 1) for a in sys.argv[1:] if "=" in a]

if args:
    mind = Mind.load(args[0])
else:
    cfg = BrainConfig()
    cfg.size = "small"
    cfg.save_path = "/nonexistent/brain.pt"

    for k, v in overrides:
        setattr(cfg, k, type(getattr(cfg, k))(v))

    mind = Mind(cfg)
    mind.calibrate()
    mind.cfg.autosave = False
    mind.language.read_text(Path("train-data.md").read_text(encoding="utf-8"), nap=False)

mind.cfg.autosave = False
lang, brain = mind.language, mind.brain
AREAS = ("offset", "striatum_d1", "striatum_d2", "gpe", "gpi", "thalamus", "motor")
seen = []
original = lang._listen_and_prepare


def traced(sound, place, token):
    rates = original(sound, place, token)
    hz = {a: float(rates[brain.areas[a].E.slice].mean()) for a in AREAS if a in brain.areas}
    hz["кора"] = float(rates[lang.meaning.E.slice].mean())
    seen.append((token, hz))
    return rates


lang._listen_and_prepare = traced

# Покой: таламус должен быть заперт тоническим GPi (у человека GPi 60–80 Гц).
with brain.frozen():
    rest = brain.observe(1000.0)
print("покой, Гц: " + " ".join(f"{a} {float(rest[brain.areas[a].E.slice].mean()):.1f}" for a in AREAS if a in brain.areas))
questions = ["Столица Франции —", "Столица Италии —", "Земля вращается вокруг", "Человек пьёт", "какая столица франции?"]
print(f"{'':34s} " + " ".join(f"{a[:8]:>8s}" for a in AREAS + ("кора",)))

for basal in (True, False):
    lang._has_basal_ganglia = (lambda: True) if basal else (lambda: False)
    print("ворота нейронами" if basal else "ровный ток кодом")

    for q in questions:
        seen.clear()

        with brain.frozen():
            reply = lang.converse(q)

        first = seen[0][1] if seen else {}
        row = " ".join(f"{first.get(a, 0):8.1f}" for a in AREAS + ("кора",))
        print(f"  {q[:22]:22s} → {reply.text[:40]}")
        print(f"  {'частоты на последнем слове, Гц':32s} {row}")
