"""
Обучение анатомического мозга через мир (brain.session), как ребёнка:
только экран, указка учителя, вкус-награда. Решения учителя (что показать,
куда вести указку, что похвалить) — мир; что увидеть, куда посмотреть,
что нажать и что выучить — мозг (веса меняются сами, engine.
chemical_plasticity).

Ступени:
  letters  — азбука: буква на экране, указка под ней;
  babble   — лепет: пустой экран, что рука нажала — видно, указка у символа;
  copy     — учитель пишет букву, мозг печатает: совпало — сладкое, нет — горькое;
  course   — курс data/course по номерам файлов (азбука → … → вопросы):
             шаг показа — из заголовка файла; first= / last= — номера
             файлов, n — сколько раз пройти курс.

    PYTHONPATH=. .venv/bin/python tools/train_anatomy.py stage=letters [n=200] [alphabet=отл] [load=МОЗГ] [save=МОЗГ]
    PYTHONPATH=. .venv/bin/python tools/train_anatomy.py stage=course [n=1] [first=1] [last=7] [save=МОЗГ]
"""

import sys
import time

from brain.config import BrainConfig
from brain.mind import Mind
from brain.session import Session
from brain.teacher import Teacher

opts = {"stage": "letters", "n": "200", "first": "1", "last": "99", "load": "", "save": "", "dwell": "300", "seed": "0", "save_every": "0", "alphabet": ""}
cfg = BrainConfig()
cfg.save_path = "/nonexistent/brain.pt"
cfg.autosave = False

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
teacher = Teacher(session, seed=int(opts["seed"]), dwell=float(opts["dwell"]), report=lambda text: print(text, flush=True))
start = time.time()
n = int(opts["n"])
stage = opts["stage"]

if stage == "letters":
    every = int(opts["save_every"])

    for done in range(0, n, every or n):
        teacher.letters(min(every or n, n - done), opts["alphabet"])

        if opts["save"] and every:
            # Промежуточный мозг — для проверки обучения по ходу.
            mind.save(opts["save"].replace(".pt", f"-{done + every}.pt"))
elif stage == "babble":
    teacher.babble(n)
elif stage == "copy":
    teacher.copy(n)
elif stage == "course":
    teacher.course(n, int(opts["first"]), int(opts["last"]))

print(f"ступень {stage}: {n} за {time.time() - start:.0f} с; саккад {len(session.saccades)}")

if opts["save"]:
    mind.save(opts["save"])
    print(f"сохранено: {opts['save']}")
