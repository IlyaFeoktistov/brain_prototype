"""
Обучение анатомического мозга через мир (brain.session), как ребёнка:
только экран, указка учителя, вкус-награда. Решения учителя (что показать,
куда вести указку, что похвалить) — мир; что увидеть, куда посмотреть,
что нажать и что выучить — мозг (веса меняются сами, engine.
chemical_plasticity).

Ступени:
  letters  — азбука: буква на экране, указка под ней;
  read     — строки train-data.md с указкой: по буквам, по слогам, по словам
             (step=letter|syllable|word);
  babble   — лепет: пустой экран, что рука нажала — видно, указка у символа;
  copy     — учитель пишет букву, мозг печатает: совпало — сладкое, нет — горькое;
  talk     — вопрос из teach-data.txt, ответ рукой, учитель оценивает;
  course   — курс data/course по номерам файлов (азбука → … → вопросы):
             шаг показа — из заголовка файла; first= / last= — номера
             файлов, n — сколько раз пройти курс.

    PYTHONPATH=. .venv/bin/python tools/train_anatomy.py stage=letters [n=200] [alphabet=отл] [load=МОЗГ] [save=МОЗГ]
    PYTHONPATH=. .venv/bin/python tools/train_anatomy.py stage=course [n=1] [first=1] [last=7] [save=МОЗГ]
"""

import random
import re
import sys
import time
from pathlib import Path

from brain.config import BrainConfig
from brain.mind import Mind
from brain.session import Session

opts = {"stage": "letters", "n": "200", "first": "1", "last": "99", "load": "", "save": "", "step": "letter", "dwell": "300", "seed": "0", "save_every": "0", "alphabet": ""}
cfg = BrainConfig()
cfg.save_path = "/nonexistent/brain.pt"
cfg.autosave = False
cfg.anatomy = True
cfg.hand = True
cfg.dt_ms = 0.5

for kv in sys.argv[1:]:
    k, v = kv.split("=", 1)

    if k in opts:
        opts[k] = v
    else:
        old = getattr(cfg, k)
        setattr(cfg, k, (v == "1") if isinstance(old, bool) else type(old)(v))

mind = Mind.load(opts["load"]) if opts["load"] else Mind(cfg)
mind.cfg.autosave = False
session = Session(mind)
rng = random.Random(int(opts["seed"]))
ALPHABET = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
VOWELS = set("аеёиоуыэюя")
dwell = float(opts["dwell"])


def syllables(word: str) -> list[str]:
    """Слоги — как их показывает учитель: открытый слог до гласной."""

    out, cur = [], ""

    for i, c in enumerate(word):
        cur += c

        if c in VOWELS and any(x in VOWELS for x in word[i + 1:]):
            out.append(cur)
            cur = ""

    return out + ([cur] if cur else []) if out else [word]


def point(text: str, step: str) -> None:
    """Учитель ведёт указку под строкой: куски по step, на каждом dwell мс."""

    session.line, session.gaze = text, 0.0
    x = 0.0

    for word in text.split(" "):
        parts = list(word) if step == "letter" else syllables(word) if step == "syllable" else [word]

        for part in parts:
            width = session.screen.width(part)
            session.pointer = x + width / 2
            session.live(dwell)
            x += width

        x += session.screen.width(" ")

    session.pointer = None


def clean(line: str) -> str:
    """Строка для экрана: строчные буквы и знаки, которые видит глаз."""

    line = re.sub(r"[^а-яё ,.?!-]", " ", line.lower())
    return re.sub(r"\s+", " ", line).strip()


def course() -> list[tuple[str, str, list[str]]]:
    """Файлы курса data/course по номерам: (имя, шаг показа, куски)."""

    out = []

    for path in sorted(Path("data/course").glob("[0-9]*.txt")):
        number = int(path.name.split("-")[0])

        if not int(opts["first"]) <= number <= int(opts["last"]):
            continue

        lines = path.read_text(encoding="utf-8").splitlines()
        step = lines[0].split(":", 1)[1].strip()
        body = [line for line in lines if line.strip() and not line.startswith("#")]

        if step == "qa":
            items = body
        elif step == "letter":
            items = [clean(line) for line in body]
        else:
            # «буква: слово, слово» (03) и «слог слог» (02) — по одному куску.
            items = [clean(x) for line in body for x in re.split(r"[,\s]+", line.split(":", 1)[-1]) if clean(x)] \
                if step == "syllable" else [clean(line) for line in body]

        out.append((path.name, step, items))

    return out


def corpus() -> list[str]:
    text = Path("train-data.md").read_text(encoding="utf-8").lower()
    lines = [re.sub(r"[^а-яё ,.?!-]", " ", line) for line in text.splitlines() if line.strip() and not line.startswith("#")]
    return [re.sub(r"\s+", " ", line).strip() for line in lines if len(line) > 3]


start = time.time()
n = int(opts["n"])
stage = opts["stage"]

if stage == "letters":
    # alphabet= — свой набор букв (малый опыт: учится ли зона различать несколько букв).
    shown = opts["alphabet"] or ALPHABET

    for i in range(n):
        point(rng.choice(shown), "letter")
        session.live(200.0)

        if i % 10 == 9:
            print(f"{i + 1}/{n} букв, {time.time() - start:.0f} с, саккад {len(session.saccades)}", flush=True)

        every = int(opts.get("save_every", "0"))

        if opts["save"] and every and (i + 1) % every == 0:
            # Промежуточный мозг — для проверки обучения по ходу.
            mind.save(opts["save"].replace(".pt", f"-{i + 1}.pt"))
elif stage == "read":
    lines = corpus()

    for i in range(n):
        point(rng.choice(lines), opts["step"])
elif stage == "babble":
    session.line, session.gaze = "", 0.0
    typed = ""

    for i in range(n):
        new = session.live(500.0)
        typed += new

        if new:
            # Указка — у нового символа: взгляд туда, куда писала рука.
            session.pointer = session.screen.width(session.line) - 3

    print(f"напечатано: {typed!r}")
elif stage == "copy":
    right = 0

    for i in range(n):
        target = rng.choice(ALPHABET)
        answer = session.reply(target + " ", max_ms=3000.0, silence_ms=800.0)
        good = answer[:1] == target
        right += good
        session.praise(good)

        if i % 20 == 19:
            print(f"{i + 1}: верно {right}/{i + 1}, последний «{target}» → {answer!r}", flush=True)
elif stage == "course":
    for lap in range(n):
        for name, step, items in course():
            right = 0

            for item in items:
                if step == "qa":
                    question, answer = item.split("|")[:2]
                    reply = session.reply(clean(question), max_ms=10000.0)
                    good = answer.strip().split()[-1][:4] in reply
                    right += good
                    session.praise(good)
                else:
                    point(item, step)
                    session.live(200.0)

            tail = f", верно {right}/{len(items)}" if step == "qa" else ""
            print(f"курс {lap + 1}/{n}: {name} ({step}, {len(items)}){tail}, {time.time() - start:.0f} с", flush=True)
elif stage == "talk":
    rows = [line.split("|") for line in Path("teach-data.txt").read_text(encoding="utf-8").splitlines()
            if "|" in line and not line.startswith("#")]
    right = 0

    for i in range(n):
        question, answer = rows[rng.randrange(len(rows))][:2]
        reply = session.reply(question.strip(), max_ms=10000.0)
        good = answer.strip().split()[-1][:4] in reply
        right += good
        session.praise(good)
        print(f"{question.strip()} → {reply!r} ({'верно' if good else answer.strip()})", flush=True)

    print(f"верно {right}/{n}")

print(f"ступень {stage}: {n} за {time.time() - start:.0f} с; саккад {len(session.saccades)}")

if opts["save"]:
    mind.save(opts["save"])
    print(f"сохранено: {opts['save']}")
