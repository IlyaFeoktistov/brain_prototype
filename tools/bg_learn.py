"""
Базальные ганглии учатся, когда говорить: мозг отвечает на вопрос, человек
ругает («-») — дофамин проваливается, пути «вопрос → D2 (стой)» крепнут.
Через несколько раз на этом вопросе мозг должен промолчать, а на другом —
отвечать как раньше. Похвала («+») — наоборот, «иди» крепнет.

    PYTHONPATH=. .venv/bin/python tools/bg_learn.py [раз=6] [key=value ...]

Новый мозг small читает train-data.md один раз (без сна).
"""

import sys
from pathlib import Path

from brain.config import BrainConfig
from brain.mind import Mind

times = int(next((a for a in sys.argv[1:] if "=" not in a), "6"))
cfg = BrainConfig()
cfg.size = "small"
cfg.save_path = "/nonexistent/brain.pt"

for k, v in (a.split("=", 1) for a in sys.argv[1:] if "=" in a):
    setattr(cfg, k, type(getattr(cfg, k))(v))

mind = Mind(cfg)
mind.calibrate()
mind.cfg.autosave = False
lang, brain = mind.language, mind.brain
lang.read_text(Path("train-data.md").read_text(encoding="utf-8"), nap=False)

rest = brain.observe(1000.0)
hz = lambda rates, area: float(rates[brain.areas[area].E.slice].mean())
print(f"покой: VTA {hz(rest, 'vta'):.1f} Гц, дофамин {brain.level('dopamine'):.2f} "
      f"(рецепторы {brain.occupancy('dopamine'):.2f}), GPi {hz(rest, 'gpi'):.0f} Гц, таламус {hz(rest, 'thalamus'):.1f} Гц")

gaps = []
spoken = lang._articulated


def traced(rates, heard):
    choice = spoken(rates, heard)
    gaps.append({a: hz(rates, a) for a in ("striatum_d1", "striatum_d2", "gpi", "thalamus")})
    return choice


lang._articulated = traced


def ask(question):
    gaps.clear()
    text = lang.converse(question).text
    g = gaps[0] if gaps else {}
    loop = f"D1 {g.get('striatum_d1', 0):.0f} D2 {g.get('striatum_d2', 0):.0f} GPi {g.get('gpi', 0):.0f} таламус {g.get('thalamus', 0):.0f}"
    return text, loop


for feedback, question in (("-", "Столица Франции —"), ("+", "Земля вращается вокруг")):
    print(f"\n«{question}», после ответа — «{feedback}»:")

    for i in range(times):
        text, loop = ask(question)
        print(f"  {i + 1}. {text[:30]:30s} | {loop}")

        if "Промолчал" not in text:
            lang.converse(feedback)

    for control in ("Человек пьёт", "Столица Италии —"):
        text, loop = ask(control)
        print(f"  контроль «{control}» → {text[:30]:30s} | {loop}")
