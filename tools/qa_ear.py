"""
Середина цепи «вопрос → факт → ответ» в обход зрения и руки (как в Spaun —
восприятие → рабочая память → выбор, — но веса учатся, а не задаются).
Слова приходят на ухо (код улитки — brain.language.Ear), ответ считывается
по сходству узоров (измерение, не часть мозга):

1. отпечатки: каждое слово словаря звучит reps раз, узор ответа L2/3 E
   ассоциативной коры и ПФК — отпечаток слова (минус средний по словарю);
2. проверка до учёбы: вопрос на слух, затем recall мс тишины; узор в
   тишине сравнивается с отпечатками всех ответов (последние слова ответов
   07 и проверка/07) — место верного среди них;
3. учёба: факты 06 на слух, слово за словом, пластичность включена,
   passes проходов, между фразами pause мс тишины;
4. проверка после учёбы: те же вопросы 07 и перефразированные
   проверка/07.

Случайный уровень — 1 / число ответов (верхнее место), средний ранг —
половина списка.

    PYTHONPATH=. .venv/bin/python tools/qa_ear.py [passes=5] [reps=3] [recall=300] [pause=500] [areas=association,prefrontal] [seed=0] [save=имя.pt]
"""

import re
import sys
from pathlib import Path

import torch

from brain.config import BrainConfig
from brain.language import Ear
from brain.mind import Mind

opts = {"passes": "5", "reps": "3", "recall": "300", "pause": "500", "areas": "association,prefrontal", "seed": "0", "save": ""}
opts.update(dict(kv.split("=", 1) for kv in sys.argv[1:]))
cfg = BrainConfig()
cfg.save_path = "/nonexistent/brain.pt"
cfg.autosave = False
cfg.anatomy = True
cfg.hand = True
cfg.dt_ms = 0.5
cfg.seed = int(opts["seed"])
torch.manual_seed(cfg.seed)
mind = Mind(cfg)
b = mind.brain
ear = Ear(cfg, b.device)
areas = opts["areas"].split(",")
cells = {a: b.areas[a].layers["L23"]["E"].slice for a in areas}


def words(line: str) -> list[str]:
    return re.sub(r"[^а-яё ]", " ", line.lower()).split()


def body(path: str) -> list[str]:
    return [l for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]


def hear(word: str) -> torch.Tensor:
    """Слово на ухо (ухо — первый орган, участок с нуля); частоты за слово."""

    raster = ear.encode(word)
    sensory = torch.zeros(raster.shape[0], b.n_receptors, device=b.device)
    sensory[:, : raster.shape[1]] = raster
    return b.observe(raster.shape[0] * cfg.dt_ms, sensory)


def pattern(rates: torch.Tensor) -> dict[str, torch.Tensor]:
    return {a: rates[c].float() for a, c in cells.items()}


facts = [words(l) for l in body("data/course/06-факты.txt")]
qa = [(words(q), words(a)[-1]) for q, a in (l.split("|")[:2] for l in body("data/course/07-вопрос-ответ.txt"))]
para = [(words(q), words(a)[-1]) for q, a in (l.split("|")[:2] for l in body("data/course/проверка/07-вопрос-ответ.txt"))]
answers = sorted({a for _, a in qa + para})
vocab = sorted({w for f in facts for w in f} | set(answers) | {w for q, _ in qa + para for w in q})
print(f"фактов {len(facts)}, вопросов {len(qa)} + перефразированных {len(para)}, ответов-кандидатов {len(answers)}, словарь {len(vocab)}")

b.plasticity = False
b.observe(2500)
prints = {a: {} for a in areas}

for w in answers:
    acc = {a: 0.0 for a in areas}

    for _ in range(int(opts["reps"])):
        b.observe(300)
        got = pattern(hear(w))

        for a in areas:
            acc[a] = acc[a] + got[a] / int(opts["reps"])

    for a in areas:
        prints[a][w] = acc[a]

for a in areas:
    mean = torch.stack(list(prints[a].values())).mean(0)
    prints[a] = {w: p - mean for w, p in prints[a].items()}


def test(tag: str, questions: list[tuple[list[str], str]]) -> None:
    was = b.plasticity
    b.plasticity = False
    ranks = {a: [] for a in areas}
    rest = 0.0

    for q, answer in questions:
        b.observe(500)

        for w in q:
            hear(w)

        got = pattern(b.observe(float(opts["recall"])))

        for a in areas:
            v = got[a] - got[a].mean()
            sims = {w: float(torch.nn.functional.cosine_similarity(v, p, dim=0)) for w, p in prints[a].items()}
            order = sorted(sims, key=lambda w: -sims[w])
            ranks[a].append(order.index(answer) + 1)

    b.plasticity = was
    n = len(answers)

    for a in areas:
        r = ranks[a]
        print(f"  {tag:28s} {a:12s}: верный ответ первым {sum(x == 1 for x in r)}/{len(r)} (случайно ~{len(r) / n:.1f}), "
              f"в тройке {sum(x <= 3 for x in r)}/{len(r)} (~{3 * len(r) / n:.1f}), средний ранг {sum(r) / len(r):.1f} из {n} (~{(n + 1) / 2:.1f})", flush=True)


test("до учёбы, вопросы 07", qa)
b.plasticity = True

for p in range(int(opts["passes"])):
    for f in facts:
        for w in f:
            hear(w)

        b.observe(float(opts["pause"]))

    print(f"проход {p + 1}/{opts['passes']} по фактам", flush=True)

test("после учёбы, вопросы 07", qa)
test("после учёбы, перефразированные", para)

if opts["save"]:
    mind.save(opts["save"])
    print(f"сохранено: {opts['save']}")
