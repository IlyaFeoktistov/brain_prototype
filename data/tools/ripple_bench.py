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
    # Как во сне: ацетилхолина нет — CA3 и выход гиппокампа свободны.
    brain.set_acetylcholine(0.0)
    for _ in range(6):
        lang.rest()
        rates = lang._ripple(ca3, everyone)
        share = {n: float((rates[brain.areas[n].E.slice] > 0).float().mean()) for n in ("entorhinal", "dentate", "ca3", "ca1", "association")}
        counts = lang._area(rates, lang.meaning)[cells].sum(1)
        top = counts.topk(8)
        words = [f"{tokens[i]}:{int(c)}" for c, i in zip(top.values.tolist(), top.indices.tolist()) if c > 0]
        print(" ".join(f"{n} {v:.1%}" for n, v in share.items()), "|", " ".join(words))
