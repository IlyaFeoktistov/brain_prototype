"""
Стенд CA3: совпадает ли узор CA3 на «?» при подсказке с записанным при
уроке (своим эпизодом) — и насколько похож на чужие.

    .venv/bin/python data/tools/hippo_overlap.py [key=value ...]
"""

import sys
from pathlib import Path

import torch

from brain.config import BrainConfig
from brain.mind import Mind
from brain.teach import load_facts

cfg = BrainConfig()
cfg.size = "small"
cfg.autosave = False
cfg.save_path = "/nonexistent/brain.pt"
cfg.neural_hippocampus = True

for kv in sys.argv[1:]:
    k, v = kv.split("=", 1)
    old = getattr(cfg, k)
    setattr(cfg, k, (v == "1") if isinstance(old, bool) else type(old)(v))

mind = Mind(cfg)
mind.calibrate()
lang, brain = mind.language, mind.brain
facts = [f for f in load_facts(Path("teach-data.txt")) if not f.held_out]
ca3, dg = brain.areas["ca3"].E, brain.areas["dentate"].E


def pattern(rates, pop):
    return rates[pop.slice] > 0


stored = {"ca3": [], "dentate": []}

for fact in facts:
    lang.rest()
    lang.experience(lang.tokenize(fact.question), learn=True)
    stored["ca3"].append(pattern(lang.last_rates, ca3))
    stored["dentate"].append(pattern(lang.last_rates, dg))
    lang.experience(lang.tokenize(fact.answer), learn=True)

lang.rest()


def jaccard(a, b):
    return float((a & b).sum()) / max(1.0, float((a | b).sum()))


for name, pop in (("dentate", dg), ("ca3", ca3)):
    own, other = [], []

    with brain.frozen():
        for i, fact in enumerate(facts):
            lang.rest()
            lang.experience(lang.tokenize(fact.question), learn=False)
            now = pattern(lang.last_rates, pop)
            own.append(jaccard(now, stored[name][i]))
            other.append(max(jaccard(now, stored[name][j]) for j in range(len(facts)) if j != i))

    # Похожесть записанных узоров между собой — разделение похожего.
    between = [jaccard(stored[name][i], stored[name][j]) for i in range(len(facts)) for j in range(i)]
    print(f"{name:8s} подсказка ↔ свой {sum(own) / len(own):.2f}, ↔ лучший чужой {sum(other) / len(other):.2f}, "
          f"записанные между собой {sum(between) / len(between):.2f}")

# Тот же вопрос дважды без обучения: насколько узор воспроизводится (шум).
for name, pop in (("entorhinal", brain.areas["entorhinal"].E), ("dentate", dg), ("ca3", ca3), ("association", brain.areas["association"].E)):
    same, top = [], []

    with brain.frozen():
        for fact in facts[:10]:
            got, counts = [], []

            for _ in range(2):
                lang.rest()
                lang.experience(lang.tokenize(fact.question), learn=False)
                got.append(pattern(lang.last_rates, pop))
                # Самые активные клетки — по числу спайков, столько же, сколько
                # в среднем активно в зоне, но не меньше 1%: сильный код.
                rates = lang.last_rates[pop.slice]
                k = max(1, int(0.25 * float((rates > 0).sum())))
                mask = torch.zeros_like(rates, dtype=torch.bool)
                mask[rates.topk(k).indices] = True
                counts.append(mask)

            same.append(jaccard(*got))
            top.append(jaccard(*counts))

    print(f"{name:11s} тот же вопрос дважды {sum(same) / len(same):.2f}, самые активные {sum(top) / len(top):.2f}, "
          f"активны {float(got[0].float().mean()):.2%}")
