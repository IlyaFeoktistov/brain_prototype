"""
Рот по буквам: в каком порядке во времени горят буквы плана (повтор про
себя) и выбора, пока мозг слушает слово, — окна 20 мс. Цепочка выбор →
план учится, если блок выбора буквы стреляет чуть раньше плана следующей.

    PYTHONPATH=. .venv/bin/python data/tools/mouth_timing.py МОЗГ слово ...
"""

import sys

import torch

from brain.mind import Mind

mind = Mind.load(sys.argv[1])
mind.cfg.autosave = False
lang, brain = mind.language, mind.brain
M, A = brain.areas["motor"].E, brain.areas["articulator"].E
art = lang.articulators
plan, out = [], []


def chunk(fired, _t):
    for pop, store in ((M, plan), (A, out)):
        cells = fired[(fired >= pop.start) & (fired < pop.stop)] - pop.start
        v = torch.zeros(pop.size)
        v[cells] = 1.0
        store.append(v)


def letters(v: torch.Tensor) -> str:
    return "".join(s for s, _ in art.said(v.to(brain.device), 0.5)[:3]) or "·"


for word in sys.argv[2:]:
    plan.clear()
    out.clear()
    brain.on_chunk = chunk

    with brain.frozen():
        lang.experience(lang.tokenize(word), learn=False)
        lang.rest()

    brain.on_chunk = None
    n = len(plan)
    print(f"{word}:")
    print("  план:  " + " | ".join(letters(v) for v in plan[:12]))
    print("  выбор: " + " | ".join(letters(v) for v in out[:12]))
