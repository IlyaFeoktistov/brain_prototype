"""
Стенд нейронного гиппокампа: запоминает ли CA3 эпизод с одного раза и
достраивает ли его по части.

    .venv/bin/python tools/hippo_bench.py [key=value ...]

Новый мозг (20 тыс., neural_hippocampus) слышит 20 строк урока «вопрос?
ответ» по одному разу. Потом без обучения — вопрос, и смотрим, какое
место смысла сильнее всех получает выученный ток от CA1 на «?»: верный
ответ первым — достроил. Плюс разреженность зубчатой извилины и CA3.
"""

import sys
import time
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
    print(f"{k} = {getattr(cfg, k)}")

t0 = time.time()
mind = Mind(cfg)
mind.calibrate()
lang, brain = mind.language, mind.brain
facts = [f for f in load_facts(Path("teach-data.txt")) if not f.held_out]

for fact in facts:
    lang.read_text(f"{fact.question} {fact.answer}")

areas = {name: brain.areas[name].E for name in ("entorhinal", "dentate", "ca3", "ca1")}
top1, ranks, active = 0, [], {name: [] for name in areas}

with brain.frozen():
    for fact in facts:
        lang.rest()
        lang.experience(lang.tokenize(fact.question), learn=False)
        rates = lang.last_rates

        for name, pop in areas.items():
            active[name].append(float((rates[pop.slice] > 0).float().mean()))

        drive = brain.learned_drive(rates, "association", source="ca1")
        tokens, cells = lang.concepts.table()
        score = drive[cells].mean(1)
        answer = lang.word(fact.answer.split()[-1])
        order = [tokens[i] for i in score.argsort(descending=True).tolist()]
        rank = order.index(answer) if answer in order else len(order)
        ranks.append(rank)
        top1 += rank == 0
        print(f"  {fact.question:42s} → {order[:3]}  (верный на {rank + 1})")

print(f"верный первым {top1}/{len(facts)}, средний ранг {sum(ranks) / len(ranks) + 1:.1f} из {len(tokens)}")
print("активны на «?»: " + ", ".join(f"{n} {sum(a) / len(a):.1%}" for n, a in active.items()))
print("смещение возбудимости: " + ", ".join(
    f"{n} {float(brain.bias[brain.areas[n].E.slice].mean()):.1f}" for n in ("entorhinal", "dentate", "ca3", "ca1")))
print(f"{time.time() - t0:.0f} с")

# Вспышки CA3 (сон) на этом же мозге: разреженны ли и будят ли слова урока.
ca3 = brain.areas["ca3"].E
everyone = torch.arange(ca3.size, device=brain.device)
answers = {lang.word(f.answer.split()[-1]) for f in facts}

with brain.frozen():
    for mv in (4.0, 6.0, 8.0, 12.0):
        shares, hits = {n: [] for n in ("ca3", "ca1", "association")}, []
        cfg.ripple_mv = mv

        for _ in range(4):
            lang.rest()
            rates = lang._ripple(ca3, everyone)

            for n in shares:
                shares[n].append(float((rates[brain.areas[n].E.slice] > 0).float().mean()))

            counts = lang._area(rates, lang.meaning)[cells].sum(1)
            top = [tokens[i] for c, i in zip(*counts.topk(5)) if c > 0]
            hits.append(" ".join(top))

        print(f"вспышка {mv:.0f} мВ: " + ", ".join(f"{n} {sum(v) / len(v):.1%}" for n, v in shares.items())
              + " | " + " / ".join(hits[:2]))
