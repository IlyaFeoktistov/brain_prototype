"""
Рот по буквам, шаги 2–3 (ROADMAP чат-минимум 7): слово учит свои буквы
(рот повторяет услышанное про себя, ПФК → рот на дендриты, буква → буква
внутри рта) и произносится из рабочей памяти.

    PYTHONPATH=. .venv/bin/python data/tools/spell_bench.py [reads=3] [go=6] [save=путь | load=путь] [show=N — окна рта у N слов] [ahp=мВ/мс — усталость рта] [inh=× торможение рта] [place=мВ — место слова в смысле] [pulse=период/ширина[/1 — нарастающий] мс — ритм толчков] [key=value ...]

Новый мозг small: лепет (5 кругов) → чтение train-data reads раз без сна →
проверка: в ПФК — только слово (без звука), рту — ровный ток go мВ (роль
ворот базальных ганглиев на стенде), окна 20 мс — какие буквы сказаны.
"""

import sys
from collections import Counter
from pathlib import Path

import torch

from brain.config import BrainConfig
from brain.language import Articulators
from brain.mind import Mind

reads, go = 3, 6.0
show = 0
ahp = None
inh = 1.0
place_mv = 30.0
pulse = None
load = save = None
overrides = []

for kv in sys.argv[1:]:
    k, v = kv.split("=", 1)

    if k == "reads":
        reads = int(v)
    elif k == "go":
        go = float(v)
    elif k == "load":
        load = v
    elif k == "show":
        show = int(v)
    elif k == "inh":
        inh = float(v)
    elif k == "place":
        place_mv = float(v)
    elif k == "pulse":
        pulse = [float(x) for x in v.split("/")]
    elif k == "ahp":
        ahp = [float(x) for x in v.split("/")]
    elif k == "save":
        save = v
    else:
        overrides.append((k, v))

cfg = BrainConfig()
cfg.size = "small"
cfg.save_path = "/nonexistent/brain.pt"
cfg.mouth_letters = True

for k, v in overrides:
    old = getattr(cfg, k)
    setattr(cfg, k, (v.lower() in ("1", "true")) if isinstance(old, bool) else type(old)(v))

mind = Mind.load(load) if load else Mind(cfg)

if not load:
    mind.calibrate()

mind.cfg.autosave = False
cfg = mind.cfg
lang, brain = mind.language, mind.brain
M = brain.areas["motor"].E
art = lang.articulators

if inh != 1.0:
    # Торможение рта на стенде: E → I и I → E × inh (конкуренция букв).
    for name in ("motor: E->I", "motor: I->E"):
        brain.projection(name).weight.mul_(inh)

if ahp:
    # Усталость клеток рта (sAHP) на стенде: ahp=мВ/мс.
    import math

    brain.slow_ahp_b[M.slice] = ahp[0]
    brain.decay_slow_ahp[M.slice] = math.exp(-cfg.dt_ms / ahp[1])
text = Path("train-data.md").read_text(encoding="utf-8")
alphabet = sorted({c.lower() for c in text if not c.isspace()})

for s in alphabet + [art.LOUD]:
    art.code(s)

if not load:
    lang.babble(alphabet, rounds=5)

pause = art.code(Articulators.PAUSE)

with brain.frozen():
    quiet = brain.observe(200.0)

print(f"тишина: пауза рта {float(quiet[M.start + pause].mean()):.0f} Гц, остальной рот "
      f"{float(quiet[M.slice].sum() - quiet[M.start + pause].sum()) / (M.size - len(pause)):.1f} Гц")

for _ in range(0 if load else reads):
    lang.read_text(text, nap=False)

if save:
    mind.save(save)

ee = next(p for p in brain.plastic if p.name == "motor: E->E")
pm = next(p for p in brain.plastic if p.name == "prefrontal -> motor")
print(f"рот E→E: макс {float(ee.weight.max()):.2f}, выше 0.5 — {int((ee.weight > 0.5).sum())}; "
      f"ПФК → рот: макс {float(pm.weight.max()):.2f}, выше 2 — {int((pm.weight > 2).sum())}")

words = Counter(lang.word(t) for t in lang.tokenize(text) if t.isalpha())
bins: list[torch.Tensor] = []


def chunk(fired, _t):
    cells = fired[(fired >= M.start) & (fired < M.stop)] - M.start
    motor = torch.zeros(M.size)
    motor[cells] = 1.0
    bins.append(motor)


def say(word: str) -> str:
    held = lang.memory.get(word, allocate=False)

    if held is None:
        return "?"

    bins.clear()
    trace = lang._memory_trace()
    trace.zero_()
    trace[held] = cfg.memory_mv
    lang._hold_memory()
    brain.on_chunk = chunk

    place = lang.concepts.get(word, allocate=False)

    with brain.frozen():
        brain.run(1500.0)  # отдых: усталость рта от прошлого слова проходит (τ 1 с)
        brain.run(100.0)  # слово в памяти, рот молчит
        # Мозг думает это слово: горит его место в смысловой коре.
        if place is not None and place_mv > 0:
            brain.feel("association", place, place_mv)

        everyone = torch.arange(M.size, device=brain.device)

        if pulse:
            # Ритм «давай» на стенде: толчок go мВ шириной pulse[1] мс каждые
            # pulse[0] мс (идея перед генератором ритма из клеток).
            period, width = pulse[0], pulse[1]
            ramp = len(pulse) > 2 and pulse[2] > 0
            t = 0.0

            while t < 300.0:
                if ramp:
                    # Нарастающий толчок (как фаза тета): гонка к порогу.
                    steps = max(1, int(width // 4))

                    for i in range(steps):
                        brain.feel("motor", everyone, go * (i + 1) / steps)
                        brain.run(4.0)
                else:
                    brain.feel("motor", everyone, go)
                    brain.run(width)

                brain.feel("motor", None, 0.0)
                brain.run(period - width)
                t += period
        else:
            brain.feel("motor", everyone, go)
            brain.run(300.0)
            brain.feel("motor", None, 0.0)
        brain.feel("association", None, 0.0)
        branches = brain.i_dend.view(brain.n_slots, -1)[M.start: M.stop].amax(1)
        mine = torch.cat([art.code(c) for c in word.lower() if c in art.codes])
        print(f"    дендрит рта: буквы слова {float(branches[mine].mean()):.1f} мВ, все {float(branches.mean()):.1f}")

    brain.on_chunk = None
    trace.zero_()
    lang._hold_memory()
    out = []

    if show > 0:
        print("    " + " | ".join(
            f"{int(b.sum())}:" + "".join(c for c, _ in art.said(b.to(brain.device), 0.25)[:3]) for b in bins[80:95]
        ))

    for b in bins[80:]:
        said = art.said(b.to(brain.device), 0.5)
        letter = said[0][0] if said else ""

        if letter and letter != Articulators.PAUSE and (not out or out[-1] != letter):
            out.append(letter)

    return "".join(out)


right = 0
test = [w for w, _ in words.most_common(20)][: show or 20]

for w in test:
    got = say(w)
    right += got == w.lower()
    print(f"  {w:12s} → {got}")

print(f"верно {right}/{len(test)}")
