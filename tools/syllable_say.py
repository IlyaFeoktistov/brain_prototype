"""
Слово → очередь слогов: держа слово в памяти (место в смысле и в ПФК), с
тета-ритмом в слоговарь — загораются ли слоги слова по порядку. Каждое окно
произнесения сравнивается с окнами слоговаря при слушании того же слова:
номер самого похожего окна слушания должен расти (1-й слог, 2-й, …).

    PYTHONPATH=. .venv/bin/python tools/syllable_say.py МОЗГ [go=18] [pulse=150/40] [слова…]
"""

import sys

import torch

from brain.mind import Mind

args = [a for a in sys.argv[2:] if "=" not in a]
opts = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
go = float(opts.get("go", 18))
period, width = (float(x) for x in opts.get("pulse", "150/40").split("/"))
mind = Mind.load(sys.argv[1])
mind.cfg.autosave = False
lang, brain, cfg = mind.language, mind.brain, mind.cfg
S = brain.areas["syllable"].E
bins: list[torch.Tensor] = []


def chunk(fired, _t):
    cells = fired[(fired >= S.start) & (fired < S.stop)] - S.start
    v = torch.zeros(S.size, dtype=torch.bool)
    v[cells] = True
    bins.append(v)


def hear(word: str) -> list[torch.Tensor]:
    bins.clear()
    brain.on_chunk = chunk

    with brain.frozen():
        brain.run(300.0)
        bins.clear()
        lang.experience(lang.tokenize(word), learn=False)

    brain.on_chunk = None
    return [b.clone() for b in bins]


def say(word: str) -> list[torch.Tensor]:
    held = lang.memory.get(word, allocate=False)
    place = lang.concepts.get(word, allocate=False)
    trace = lang._memory_trace()
    trace.zero_()
    trace[held] = cfg.memory_mv
    lang._hold_memory()
    brain.set_acetylcholine(0.0)
    everyone = torch.arange(S.size, device=brain.device)

    with brain.frozen():
        brain.run(2500.0)  # отдых: усталость слогов (τ 2 с)
        brain.feel("association", place, 30.0)
        brain.run(100.0)
        bins.clear()
        brain.on_chunk = chunk
        t = 0.0

        while t < 600.0:
            steps = max(1, int(width // 4))

            for i in range(steps):
                brain.feel("syllable", everyone, go * (i + 1) / steps)
                brain.run(4.0)

            brain.feel("syllable", None, 0.0)
            brain.run(period - width)
            t += period

        brain.on_chunk = None
        brain.feel("association", None, 0.0)

    trace.zero_()
    lang._hold_memory()
    return [b.clone() for b in bins]


def jaccard(a, b) -> float:
    u = int((a | b).sum())
    return int((a & b).sum()) / u if u else 0.0


words = args or [w for w in ("является", "находится", "человек", "столица", "результат", "окружающие",
                             "ожидание", "изменения") if w in lang.memory.places]
good = 0

for w in words:
    heard = [b for b in hear(w) if b.any()]
    spoken = [b for b in say(w) if b.any()]
    order = []

    for b in spoken:
        scores = [jaccard(b, h) for h in heard]
        best = max(range(len(scores)), key=scores.__getitem__) if scores else -1
        order.append(f"{best + 1}({scores[best]:.1f})" if best >= 0 else "?")

    rising = [int(o.split("(")[0]) for o in order if o != "?"]
    ok = len(rising) >= 2 and all(a <= b for a, b in zip(rising, rising[1:])) and rising[-1] > rising[0]
    good += ok
    print(f"  {w:12s} слушание: {len(heard)} окон; произнесение: {len(spoken)} окон → {' '.join(order)}{'  ✓' if ok else ''}")

print(f"по порядку: {good}/{len(words)}")
