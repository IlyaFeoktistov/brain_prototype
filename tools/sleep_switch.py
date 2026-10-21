"""
Переключатель сна (Saper, Chou & Scammell 2001): держит ли мозг
бодрствование, пока растёт аденозин, засыпает ли по нему и не перескакивает
ли сам. Аденозин (занятость рецепторов) идёт лесенкой вверх, потом вниз; на
каждой ступени — частоты ядер бодрствования и VLPO по окнам и число
перескоков. Переключатель «шлёпанец»: внутри ступени перескоков нет, вверх
засыпает позже, чем вниз просыпается (гистерезис).

    PYTHONPATH=. .venv/bin/python tools/sleep_switch.py [МОЗГ] [hour=14 ...]

Без МОЗГ — новый мозг small. hour — время суток (часы SCN), step_ms —
длина ступени, win_ms — окно частот. Мозг заморожен.
"""

import sys

from brain.config import BrainConfig
from brain.mind import Mind

args = [a for a in sys.argv[1:] if "=" not in a]
opts = {"hour": "14", "step_ms": "3000", "win_ms": "500", "levels": "0.05,0.2,0.35,0.5,0.6,0.7,0.8,0.9"}
overrides = []

for kv in sys.argv[1:]:
    if "=" not in kv:
        continue

    k, v = kv.split("=", 1)

    if k in opts:
        opts[k] = v
    else:
        overrides.append((k, v))

if args:
    mind = Mind.load(args[0])
    cfg = mind.cfg
else:
    cfg = BrainConfig()
    cfg.size = "small"
    cfg.autosave = False
    cfg.save_path = "/nonexistent/brain.pt"

for k, v in overrides:
    old = getattr(cfg, k)
    setattr(cfg, k, (v == "1") if isinstance(old, bool) else type(old)(v))
    print(f"{k} = {getattr(cfg, k)}")

if not args:
    mind = Mind(cfg)

cfg.autosave = False
brain = mind.brain
brain.plasticity = False
adenosine = brain.chemistry_names.index("adenosine")
half = cfg.adenosine_half
wake = [a for a in ("lc", "raphe", "orexin", "nbm") if a in brain.areas]
names = ["scn"] * ("scn" in brain.areas) + wake + ["vlpo"]
levels = [float(x) for x in opts["levels"].split(",")]
step, win = float(opts["step_ms"]), float(opts["win_ms"])


def rate(rates, area):
    return float(rates[brain.areas[area].E.slice].mean())


def asleep() -> bool:
    return brain.level("galanin") > brain.level("noradrenaline")


print(f"{brain.n:,} нейронов; час {opts['hour']}; ступень {step:g} мс, окно {win:g} мс")
print(f"{'аденозин':>8} | " + " ".join(f"{n:>7}" for n in names) + " | сон по окнам        перескоков")
flips_total = 0

with brain.frozen():
    brain.time_ms = ((float(opts["hour"]) - cfg.birth_hour) % 24.0) * 3_600_000.0
    brain.run(500.0)

    for occupancy in levels + levels[-2::-1]:
        level = occupancy * half / (1.0 - occupancy)
        states, sums = [], {n: 0.0 for n in names}
        windows = int(step // win)

        for _ in range(windows):
            # Аденозин за секунды не меняется — держим ступень.
            brain.chemistry[adenosine] = level
            rates = brain.observe(win)

            for n in names:
                sums[n] += rate(rates, n) / windows

            states.append(asleep())

        flips = sum(a != b for a, b in zip(states, states[1:]))
        flips_total += flips
        marks = "".join("z" if s else "." for s in states)
        print(f"{occupancy:8.2f} | " + " ".join(f"{sums[n]:7.2f}" for n in names) + f" | {marks:20s} {flips}")

print(f"перескоков внутри ступеней всего: {flips_total}")
