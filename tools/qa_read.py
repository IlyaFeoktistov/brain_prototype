"""
Мозг после чтения + N пар корпуса вопрос–ответ + сон.

    .venv/bin/python tools/qa_read.py МОЗГ N ВЫХОД
"""

import sys
import time
from pathlib import Path

from brain.commands import Session
from brain.mind import Mind

mind = Mind.load(sys.argv[1])
mind.cfg.autosave = False
mind.brain.recompile()
n = int(sys.argv[2])
lines = Path("data/corpus/qa/вопрос-ответ.txt").read_text(encoding="utf-8").splitlines()[:n]
t = time.time()
Session(mind, lambda text, kind: None).read_text("qa", "\n".join(lines))
print(f"прочитал {n} пар за {time.time() - t:.0f} с", flush=True)
t = time.time()
mind.language.sleep_until_rested()
print(f"сон {time.time() - t:.0f} с", flush=True)
mind.save(sys.argv[3])
