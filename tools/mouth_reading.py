"""
Рот по буквам: что делает рот, пока мозг слушает фразу (клеток, Гц,
какие артикуляции выстрелили).

    PYTHONPATH=. .venv/bin/python tools/mouth_reading.py МОЗГ "Фраза."
"""
import sys, torch
from brain.mind import Mind
mind = Mind.load(sys.argv[1]); mind.cfg.autosave = False
lang, brain = mind.language, mind.brain
M = brain.areas["motor"].E; art = lang.articulators
orig = lang.perceive
def traced(token, *a, **kw):
    r = orig(token, *a, **kw)
    m = r[M.slice]
    said = art.said(m, 0.5)[:6]
    print(f"  {token:10s} рот: {int((m>0).sum())} кл., {float(m[m>0].mean()) if (m>0).any() else 0:.0f} Гц; буквы {' '.join(s for s,_ in said)}")
    return r
lang.perceive = traced
with brain.frozen():
    lang.experience(lang._ended(lang.tokenize(sys.argv[2])), learn=False)
