"""
Узнавание слова по звуку: мозг слышит только звучание (ухо по буквам),
место смысла ему не подсказывают. Какое место загорается сильнее всех —
его ли это слово? Для точной формы, другого регистра, другой формы слова
и опечатки.

    .venv/bin/python tools/recognition.py МОЗГ

Ток — выученный путь «слух → смысл» (recognize); спайки — место, которое
выстрелило само при открытых воротах «говори» (ровный ток на кору).
"""

import sys

import torch

from brain.mind import Mind

mind = Mind.load(sys.argv[1])
mind.cfg.autosave = False
lang, brain = mind.language, mind.brain
tokens, cells = lang.concepts.table()
index = {t: i for i, t in enumerate(tokens)}
assoc = lang.meaning.E

# (как прозвучало, чьё это место)
probes = [
    ("франции", "франции"), ("Франции", "франции"), ("франция", "франции"),
    ("францию", "франции"), ("францыи", "франции"),
    ("столица", "столица"), ("Столица", "столица"), ("столицей", "столица"), ("стлица", "столица"),
    ("париж", "париж"), ("Париж", "париж"), ("парижа", "париж"), ("пориж", "париж"),
    ("земля", "земля"), ("земли", "земля"), ("землю", "земля"), ("зимля", "земля"),
    ("вода", "вода"), ("воды", "вода"), ("воду", "вода"),
    ("солнце", "солнце"), ("солнца", "солнце"), ("сонце", "солнце"),
]


def rank(score: torch.Tensor, target: str) -> int:
    i = index[target]
    return int((score > score[i]).sum()) + 1


rows = []
with brain.frozen():
    for form, target in probes:
        if target not in index:
            continue
        lang.rest()
        sound = lang.ear.encode(form)
        rates = brain.observe(mind.cfg.stimulus_ms, sound)
        drive = lang.recognize(rates)
        by_drive = drive[cells].mean(1)
        # Ворота «говори»: ровный ток всей коре — какое место выстрелит само.
        lang.rest()
        brain.feel("association", torch.arange(assoc.size, device=brain.device), mind.cfg.speak_gate_mv)
        fired = brain.observe(mind.cfg.stimulus_ms, sound)
        brain.feel("association", None, 0.0)
        by_spikes = fired[assoc.slice][cells].sum(1)
        top = tokens[int(by_drive.argmax())]
        spiking = tokens[int(by_spikes.argmax())] if by_spikes.max() > 0 else "—"
        known = "знакомо" if form.casefold() in index else "новая форма"
        rows.append((form, target, rank(by_drive, target), top, spiking, known))

print(f"{len(tokens)} мест слов")
print(f"{'звук':12s} {'место':10s} {'ранг по току':>12s}  {'первым по току':16s} {'выстрелило':14s}")
for form, target, r, top, spiking, known in rows:
    print(f"{form:12s} {target:10s} {r:12d}  {top:16s} {spiking:14s} {known}")

for label, picked in (("знакомые формы", [r for r in rows if r[5] == "знакомо"]),
                      ("новые формы и опечатки", [r for r in rows if r[5] != "знакомо"])):
    if picked:
        ranks = [r[2] for r in picked]
        first = sum(r[1] == r[4] for r in picked)
        print(f"{label}: средний ранг {sum(ranks) / len(ranks):.1f}, первым по току {sum(x == 1 for x in ranks)}/{len(ranks)}, "
              f"выстрелило своё {first}/{len(ranks)}")
