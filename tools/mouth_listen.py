"""
Рот при слушании: стреляет ли моторная зона на услышанное слово и у
скольких её клеток есть вход от места слова.

    PYTHONPATH=. .venv/bin/python tools/mouth_listen.py МОЗГ
"""
import sys, torch
from brain.mind import Mind
mind = Mind.load(sys.argv[1]); mind.cfg.autosave = False
lang, brain = mind.language, mind.brain
E = brain.areas["association"].E; M = brain.areas["motor"].E
proj = brain.projection("association -> motor")
print("смысл→рот: форма весов", tuple(proj.weight.shape), "веса", float(proj.weight.min()), float(proj.weight.max()))
pre = proj.pre.long() - E.start
import os

orig = lang.perceive
def traced(token, *a, **kw):
    r = orig(token, *a, **kw)
    place = lang.concepts.get(lang.word(token), allocate=False)
    if place is not None:
        conn = torch.isin(pre, place).sum(1)          # синапсов от места к каждой клетке рта
        fired = r[M.slice] > 0
        print(f"  {token:10s} место {float(r[E.start+place].mean()):.0f} Гц; рот стреляет {int(fired.sum())} кл., из них со входом от места {int((fired & (conn>0)).sum())} (всего со входом {int((conn>0).sum())})")
    return r
lang.perceive = traced
with brain.frozen():
    lang.experience(lang.tokenize("Человек пьёт воду ."), learn=False)
