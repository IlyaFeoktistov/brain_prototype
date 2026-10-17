"""
Рот по буквам: градиент первичности — средний вход от места слова (смысл →
рот) к клеткам каждой его буквы по порядку против чужих клеток рта.

    PYTHONPATH=. .venv/bin/python tools/mouth_primacy.py МОЗГ
"""
import sys, torch
from brain.mind import Mind
mind = Mind.load(sys.argv[1]); mind.cfg.autosave = False
lang, brain = mind.language, mind.brain
E = brain.areas["association"].E; art = lang.articulators
am = next(p for p in brain.plastic if p.name == "association -> motor")
for w in ["человек", "может", "находится", "вода", "москва", "столица"]:
    place = lang.concepts.get(w, allocate=False)
    hit = torch.isin(am.pre, (E.start + place).to(am.pre.dtype))
    sums = (am.weight * hit).sum(1)          # вход от места на каждую клетку рта
    row = " ".join(f"{c}:{float(sums[art.code(c)].mean()):.1f}" for c in w)
    others = torch.ones(len(sums), dtype=torch.bool, device=sums.device)
    for c in set(w): others[art.code(c)] = False
    print(f"{w:10s} {row}   | чужие {float(sums[others].mean()):.2f}")
