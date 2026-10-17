"""
Удержание прочитанного: новый мозг читает train-data ×2 и спит → экзамен;
читает N пар корпуса вопрос–ответ и спит → экзамен снова.

    .venv/bin/python tools/retention.py ИМЯ N [key=value ...]
"""

import sys
import time
from pathlib import Path

from brain import evaluate
from brain.commands import Session
from brain.config import BrainConfig
from brain.mind import Mind

name, n = sys.argv[1], int(sys.argv[2])
cfg = BrainConfig()
cfg.size = "small"
cfg.autosave = False
cfg.save_path = "/nonexistent/brain.pt"

for kv in sys.argv[3:]:
    k, v = kv.split("=", 1)
    old = getattr(cfg, k)
    setattr(cfg, k, (v == "1") if isinstance(old, bool) else type(old)(v))
    print(f"{k} = {getattr(cfg, k)}")

mind = Mind(cfg)
mind.calibrate()
session = Session(mind, lambda text, kind: None)
questions = evaluate.facts(Path("train-data.md"), mind.language.tokenize)
t0 = time.time()


def exam(tag: str) -> None:
    score = evaluate.run(mind, questions, 0)["факты"]
    used = (mind.language.concepts.usage > 0).float().mean()
    print(f"{tag:28s} первое {score.first}/{score.total}, в тройке {score.top3}, "
          f"эпизодов {len(mind.language.hippocampus)}, места {used:.0%}, {time.time() - t0:.0f} с", flush=True)


text = Path("train-data.md").read_text(encoding="utf-8")
for _ in range(2):
    session.read_text("train-data", text)
mind.language.sleep_until_rested()
exam("train-data ×2 + сон")
mind.save(f"{name}-read.pt")

lines = Path("data/corpus/qa/вопрос-ответ.txt").read_text(encoding="utf-8").splitlines()[:n]
session.read_text("qa", "\n".join(lines))
mind.language.sleep_until_rested()
exam(f"+ {n} пар корпуса + сон")
mind.save(f"{name}-qa.pt")
