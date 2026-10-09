"""
Стенд сна вспышками: что происходит во вспышке CA3 (без обучения).

    .venv/bin/python data/tools/ripple_bench.py МОЗГ [key=value ...]

Доля активных клеток по зонам гиппокампа и коры и какие места слов
вспыхнули — похоже ли это на эпизоды (строки прочитанного) или на шум.
"""

import sys

import torch

from brain.mind import Mind

mind = Mind.load(sys.argv[1])
mind.cfg.autosave = False

for kv in sys.argv[2:]:
    k, v = kv.split("=", 1)
    old = getattr(mind.cfg, k)
    setattr(mind.cfg, k, (v == "1") if isinstance(old, bool) else type(old)(v))

mind.brain.recompile()
lang, brain = mind.language, mind.brain
ca3 = brain.areas["ca3"].E
everyone = torch.arange(ca3.size, device=brain.device)
tokens, cells = lang.concepts.table()

with brain.frozen():
    for _ in range(6):
        lang.rest()
        brain.feel("ca3", everyone, mind.cfg.ripple_mv)
        rates = brain.observe(mind.cfg.ripple_ms)
        brain.feel("ca3", None, 0.0)
        share = {n: float((rates[brain.areas[n].E.slice] > 0).float().mean()) for n in ("ca3", "ca1", "association", "entorhinal")}
        counts = lang._area(rates, lang.meaning)[cells].sum(1)
        top = counts.topk(8)
        words = [f"{tokens[i]}:{int(c)}" for c, i in zip(top.values.tolist(), top.indices.tolist()) if c > 0]
        print(" ".join(f"{n} {v:.1%}" for n, v in share.items()), "|", " ".join(words))
