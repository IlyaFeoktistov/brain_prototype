"""
Рот по буквам: выучил ли рот порядок букв — средний вес связей «буква а →
буква б» вперёд и назад для частых пар train-data против случайных пар.

    PYTHONPATH=. .venv/bin/python data/tools/mouth_bigrams.py МОЗГ
"""
import sys, torch
from collections import Counter
from pathlib import Path
from brain.mind import Mind
mind = Mind.load(sys.argv[1]); mind.cfg.autosave = False
lang, brain = mind.language, mind.brain
M = brain.areas["motor"].E; art = lang.articulators
ee = next(p for p in brain.plastic if p.name == "motor: E->E")
def link(a, b):
    pa, pb = art.code(a), art.code(b)
    hit = torch.isin(ee.pre[pb], (M.start + pa).to(ee.pre.dtype))
    return float((ee.weight[pb] * hit).sum() / hit.sum().clamp(min=1))
text = Path("train-data.md").read_text(encoding="utf-8").lower()
pairs = Counter(w[i:i+2] for w in lang.tokenize(text) if w.isalpha() for i in range(len(w)-1))
fw = bw = n = 0
for bg, c in pairs.most_common(15):
    a, b = bg
    f, r = link(a, b), link(b, a)
    print(f"  {a}→{b} ({c}): вперёд {f:.2f}, назад {r:.2f}")
rnd = [link(a, b) for a in "абвгд" for b in "жзийк"]
print("  случайные пары (не соседи):", round(sum(rnd)/len(rnd), 2))
