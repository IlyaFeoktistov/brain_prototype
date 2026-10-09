"""
Что сон вспышками делает с корой: экзамен до и после N вспышек.

    .venv/bin/python data/tools/ripple_sleep.py МОЗГ N [key=value ...]

Мозг — с нейронным гиппокампом (например data/measure/nh4-read.pt:
прочитал train-data, спал старым сном). key=value — поля конфига на
время сна (sleep_downscale=0, replay_dopamine=0, ripple_mv=…).
"""

import sys
import time
from pathlib import Path

from brain import evaluate
from brain.mind import Mind

mind = Mind.load(sys.argv[1])
mind.cfg.autosave = False
mind.cfg.neural_replay = True

for kv in sys.argv[3:]:
    k, v = kv.split("=", 1)
    old = getattr(mind.cfg, k)
    setattr(mind.cfg, k, (v == "1") if isinstance(old, bool) else type(old)(v))
    print(f"{k} = {getattr(mind.cfg, k)}")

mind.brain.recompile()
questions = evaluate.facts(Path("train-data.md"), mind.language.tokenize)


def exam() -> str:
    score = evaluate.run(mind, questions, 0)["факты"]
    return f"{score.first}/{score.total} (в тройке {score.top3})"


before = exam()
t = time.time()
mind.language.sleep(int(sys.argv[2]))
print(f"до {before} → после {exam()}, сон {time.time() - t:.0f} с; приснилось: {' | '.join(list(mind.language.dreams)[-3:])}")
