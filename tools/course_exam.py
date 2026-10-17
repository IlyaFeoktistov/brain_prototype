"""
Экзамен по курсу data/course на контрольных наборах data/course/проверка
(в учёбе их нет): пластичность выключена, вкуса-награды нет, порядок
перемешан.

- 01 (буквы): различает ли зона букв все буквы — местные клетки L4 и L2/3
  (как letters_local), сходство ответов на одну букву против разных; буква
  в ямке, без указки.
- шаг qa (06, 07): вопрос — ответ рукой; верно, если есть последнее слово
  ответа учителя (первые 4 буквы, как в курсе).
- 02–04: считывания слога и слова у мозга пока нет — наборы на будущее.

    PYTHONPATH=. .venv/bin/python tools/course_exam.py МОЗГ [parts=01,06,07] [reps=2] [half=8] [seed=0]
"""

import random
import re
import sys
from pathlib import Path

import torch

from brain.mind import Mind
from brain.session import Session

opts = {"parts": "01,06,07", "reps": "2", "half": "8", "seed": "0"}
opts.update(dict(kv.split("=", 1) for kv in sys.argv[2:]))
mind = Mind.load(sys.argv[1])
mind.cfg.autosave = False
b = mind.brain
b.plasticity = False
s = Session(mind)
rng = random.Random(int(opts["seed"]))
checks = {p.name.split("-")[0]: p for p in sorted(Path("data/course/проверка").glob("[0-9]*.txt"))}


def clean(line: str) -> str:
    line = re.sub(r"[^а-яё ,.?!-]", " ", line.lower())
    return re.sub(r"\s+", " ", line).strip()


def body(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]


def letters(path: Path) -> None:
    half = float(opts["half"])
    mid = b.cfg.fovea_col + 4.0
    pops = {l: b.areas["letters"].layers[l]["E"] for l in ("L4", "L23")}
    cells = {l: torch.arange(p.start, p.stop, device=b.device)[(b.field_cols[p.name] - mid).abs() <= half] for l, p in pops.items()}
    show = [clean(x) for x in body(path)]
    shown, Z = [], {l: [] for l in cells}
    s.pointer = None

    for rep in range(int(opts["reps"])):
        order = show[:] if rep == 0 else rng.sample(show, len(show))

        for letter in order:
            s.line = ""
            b.observe(300, s._view(300))
            rest = b.observe(300, s._view(300))
            s.line, s.gaze = letter, 0.0
            seen = b.observe(300, s._view(300))
            shown.append(letter)

            for l, c in cells.items():
                Z[l].append((seen[c] - rest[c]) * 0.3 / ((rest[c] + 0.5) * 0.3).sqrt())

    s.line = ""
    same = torch.tensor([[a == x for x in shown] for a in shown])
    eye_ = torch.eye(len(shown), dtype=torch.bool)

    for l, zs in Z.items():
        R = torch.stack(zs).float().cpu()
        R = R - R.mean(1, keepdim=True)
        R = R / R.norm(dim=1, keepdim=True).clamp(min=1e-6)
        C = R @ R.t()
        print(f"  буквы {l}: одна {float(C[same & ~eye_].mean()):.3f} разные {float(C[~same].mean()):.3f} разница {float(C[same & ~eye_].mean() - C[~same].mean()):+.3f}")


def questions(path: Path) -> None:
    rows = [line.split("|")[:2] for line in body(path)]
    right = 0

    for question, answer in rng.sample(rows, len(rows)):
        reply = s.reply(clean(question), max_ms=10000.0)
        good = answer.strip().split()[-1][:4] in reply
        right += good
        print(f"  {clean(question)} → {reply!r} ({'верно' if good else answer.strip()})", flush=True)

    print(f"  верно {right}/{len(rows)}")


b.observe(2500, s._view(2500))

for part in opts["parts"].split(","):
    path = checks[part]
    step = path.read_text(encoding="utf-8").splitlines()[0].split(":", 1)[1].strip()
    print(f"{path.name} ({step}):", flush=True)

    if part == "01":
        letters(path)
    elif step == "qa":
        questions(path)
    else:
        print("  считывания пока нет — набор на будущее")
