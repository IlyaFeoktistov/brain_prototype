"""
Слой выбора рта (зона «articulator», competitive queuing): выдаёт ли он
буквы по одной в порядке подготовки. Блокам пяти символов подаётся
постоянный ток плана (по убыванию), сверху — ритм «давай» нарастающими
толчками; окна 20 мс — сколько блоков выстрелило и какие.

    PYTHONPATH=. .venv/bin/python data/tools/choice_bench.py [inh=1] [ei=1] [col=1] [go=10] [ahp=8] [ahp_tau=1000] [plan=12/10/8/6/4] [pulse=60/40]

inh / ei / col — множители путей зоны на стенде: I → E, E → I, колонки (E → E).
Новый мозг small с ртом по буквам, без чтения (выбору учиться нечему).
"""

import sys

import torch

from brain.config import BrainConfig
from brain.mind import Mind

opts = {"inh": 1.0, "ei": 1.0, "col": 1.0, "go": 10.0, "ahp": 8.0, "ahp_tau": 1000.0}
plan = [12.0, 10.0, 8.0, 6.0, 4.0]
pulse = [60.0, 40.0]

for kv in sys.argv[1:]:
    k, v = kv.split("=", 1)

    if k == "plan":
        plan = [float(x) for x in v.split("/")]
    elif k == "pulse":
        pulse = [float(x) for x in v.split("/")]
    else:
        opts[k] = float(v)

cfg = BrainConfig()
cfg.size = "small"
cfg.save_path = "/nonexistent/brain.pt"
cfg.mouth_letters = True
cfg.letter_ms = 25.0
mind = Mind(cfg)
mind.calibrate()
mind.cfg.autosave = False
lang, brain = mind.language, mind.brain
A = brain.areas["articulator"].E
art = lang.articulators

for name, key in (("articulator: I->E", "inh"), ("articulator: E->I", "ei"), ("articulator: E->E", "col")):
    brain.projection(name).weight.mul_(opts[key])

import math

brain.slow_ahp_b[A.slice] = opts["ahp"]
brain.decay_slow_ahp[A.slice] = math.exp(-cfg.dt_ms / opts["ahp_tau"])
letters = "абвгд"[: len(plan)]
drive = torch.zeros(A.size, device=brain.device)

for c, mv in zip(letters, plan):
    drive[art.code(c)] = mv

for c in "еёжзийклмнопрст":
    art.code(c)

bins: list[torch.Tensor] = []


def chunk(fired, _t):
    cells = fired[(fired >= A.start) & (fired < A.stop)] - A.start
    out = torch.zeros(A.size)
    out[cells] = 1.0
    bins.append(out)


everyone = torch.arange(A.size, device=brain.device)

with brain.frozen():
    brain.run(500.0)
    brain.on_chunk = chunk
    t = 0.0

    while t < 600.0:
        steps = max(1, int(pulse[1] // 4))

        for i in range(steps):
            brain.feel("articulator", everyone, drive + opts["go"] * (i + 1) / steps)
            brain.run(4.0)

        brain.feel("articulator", everyone, drive)
        brain.run(pulse[0] - pulse[1])
        t += pulse[0]

    brain.feel("articulator", None, 0.0)
    brain.on_chunk = None

row, order = [], []

for b in bins:
    said = art.said(b.to(brain.device), 0.5)
    row.append(f"{int(b.sum())}:" + "".join(s for s, _ in said))

    for s, _ in said:
        if not order or order[-1] != s:
            order.append(s)

print(" | ".join(row))
print(f"порядок: {''.join(order)}  (ждали {letters}); окон с >1 буквой: "
      f"{sum(len(art.said(b.to(brain.device), 0.5)) > 1 for b in bins)}")
