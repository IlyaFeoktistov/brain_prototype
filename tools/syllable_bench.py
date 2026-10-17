"""
Слоговарь (зона «syllable»): захватили ли клетки частые сочетания звуков.
Мозг слушает слово (заморожен); клетки слоговаря первых двух окон, где
он ответил, — его начальный слог. Слова с общим началом
(«может» / «москва» — «мо») должны делить клетки, без общего — нет.

    PYTHONPATH=. .venv/bin/python tools/syllable_bench.py МОЗГ
"""

import sys

import torch

from brain.mind import Mind

mind = Mind.load(sys.argv[1])
mind.cfg.autosave = False
lang, brain = mind.language, mind.brain
S = brain.areas["syllable"].E
bins: list[torch.Tensor] = []


def chunk(fired, _t):
    cells = fired[(fired >= S.start) & (fired < S.stop)] - S.start
    v = torch.zeros(S.size, dtype=torch.bool)
    v[cells] = True
    bins.append(v)


def onset(word: str, first: int = 2) -> tuple[torch.Tensor, list[int]]:
    bins.clear()
    brain.on_chunk = chunk

    with brain.frozen():
        brain.run(300.0)
        bins.clear()
        lang.experience(lang.tokenize(word), learn=False)
        lang.rest()

    brain.on_chunk = None
    # Начальный слог — первые окна, где слоговарь ответил (путь слух → рот →
    # выбор → слоговарь идёт с задержкой).
    head = torch.zeros(S.size, dtype=torch.bool)

    for b in [b for b in bins if b.any()][:first]:
        head |= b

    return head, [int(b.sum()) for b in bins[:8]]


def jaccard(a, b) -> float:
    u = int((a | b).sum())
    return int((a & b).sum()) / u if u else 0.0


# Пары из прочитанных слов (у них есть места): общие первые две буквы
# против разных первых букв.
import random

known = sorted(w for w in lang.memory.places if len(w) >= 4 and w.isalpha())
rng = random.Random(0)
by_head: dict[str, list[str]] = {}

for w in known:
    by_head.setdefault(w[:2], []).append(w)

PAIRS = [tuple(rng.sample(ws, 2)) for ws in by_head.values() if len(ws) >= 2]
rng.shuffle(PAIRS)
PAIRS = PAIRS[:12]
OTHER = []

while len(OTHER) < len(PAIRS):
    a, b = rng.sample(known, 2)
    if a[0] != b[0]:
        OTHER.append((a, b))

cache = {}

for a, b in PAIRS + OTHER:
    for w in (a, b):
        if w not in cache:
            cache[w] = onset(w)

silent = [w for w, (head, _) in cache.items() if not head.any()]
print(f"слов {len(cache)}, слоговарь молчит на {len(silent)}: {', '.join(silent[:10])}")

same = [jaccard(cache[a][0], cache[b][0]) for a, b in PAIRS]
diff = [jaccard(cache[a][0], cache[b][0]) for a, b in OTHER]
print("общее начало: " + ", ".join(f"{a}/{b} {100 * j:.0f}%" for (a, b), j in zip(PAIRS, same)))
print("без общего:   " + ", ".join(f"{a}/{b} {100 * j:.0f}%" for (a, b), j in zip(OTHER, diff)))
print(f"среднее: общее начало {100 * sum(same) / len(same):.0f}%, без общего {100 * sum(diff) / len(diff):.0f}%")
