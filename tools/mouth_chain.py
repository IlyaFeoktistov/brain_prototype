"""
Рот по буквам: цепочка выбор → план внутри слов — вход на клетку следующей
буквы от блока предыдущей (выученный переход) против случайной пары.

    PYTHONPATH=. .venv/bin/python tools/mouth_chain.py МОЗГ
"""
import sys, torch
from brain.mind import Mind
mind = Mind.load(sys.argv[1]); mind.cfg.autosave = False
lang, brain = mind.language, mind.brain
M = brain.areas["motor"].E; A = brain.areas["articulator"].E; art = lang.articulators
am = next(p for p in brain.plastic if p.name == sys.argv[2]) if len(sys.argv) > 2 else next(p for p in brain.plastic if p.name == "articulator -> motor")
SRC = A if am.name.startswith("articulator") else M
def link(a, b):
    pa, pb = art.code(a), art.code(b)
    hit = torch.isin(am.pre[pb], (SRC.start + pa).to(am.pre.dtype))
    return float((am.weight[pb] * hit).sum(1).mean())   # вход на клетку b от блока a
for w in ["человек", "может", "находится", "если", "является"]:
    print(w, " ".join(f"{a}→{b}:{link(a,b):.1f}" for a, b in zip(w, w[1:])), "| случайно:", f"{link(w[0], 'щ'):.1f}")
