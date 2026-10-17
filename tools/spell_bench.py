"""
Рот по буквам, шаги 2–3 (ROADMAP чат-минимум 7): слово учит свои буквы
(рот повторяет услышанное про себя, ПФК → рот на дендриты, буква → буква
внутри рта) и произносится из рабочей памяти.

    PYTHONPATH=. .venv/bin/python tools/spell_bench.py [reads=3] [size=small] [go=6] [save=путь | load=путь] [show=N — окна рта у N слов] [ahp=мВ/мс — усталость рта] [inh=× торможение рта] [place=мВ — место слова в смысле] [pulse=период/ширина[/1 — нарастающий] мс — ритм толчков] [plan=× план → выбор] [nostp=1 — проверка: смысл без истощения] [dsoma=мВ — плато слова → сома плана] [chain=× цепочка выбор → план] [timing=× часы → план] [key=value ...]

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
plan_gain = 1.0
no_stp = False
dsoma = None
chain = 1.0
timing_gain = 1.0
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
    elif k == "plan":
        plan_gain = float(v)
    elif k == "nostp":
        no_stp = v == "1"
    elif k == "dsoma":
        dsoma = float(v)
    elif k == "chain":
        chain = float(v)
    elif k == "timing":
        timing_gain = float(v)
    elif k == "pulse":
        pulse = [float(x) for x in v.split("/")]
    elif k == "ahp":
        ahp = [float(x) for x in v.split("/")]
    elif k == "save":
        save = v
    else:
        overrides.append((k, v))

cfg = BrainConfig()
cfg.size = next((v for k, v in overrides if k == "size"), "small")
overrides = [(k, v) for k, v in overrides if k != "size"]
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
# Выход — слой выбора, если он есть (иначе — сам рот).
OUT = brain.areas["articulator"].E if "articulator" in brain.areas else M
GO = "articulator" if "articulator" in brain.areas else "motor"
art = lang.articulators

if inh != 1.0:
    # Торможение рта на стенде: E → I и I → E × inh (конкуренция букв).
    for name in ("motor: E->I", "motor: I->E"):
        brain.projection(name).weight.mul_(inh)

if ahp:
    # Усталость выхода рта (sAHP) на стенде: ahp=мВ/мс (слой выбора, если есть).
    import math

    brain.slow_ahp_b[OUT.slice] = ahp[0]
    brain.decay_slow_ahp[OUT.slice] = math.exp(-cfg.dt_ms / ahp[1])

if chain != 1.0 and "articulator" in brain.areas:
    # Цепочка выбор → план × chain (стенд): в одиночку ниже порога?
    brain.projection("articulator -> motor").weight.mul_(chain)

if timing_gain != 1.0 and "timing" in brain.areas:
    # Часы → план × timing (стенд): в одиночку ниже порога?
    brain.projection("timing -> motor").weight.mul_(timing_gain)

if dsoma is not None:
    # Плато слова → сома плана, мВ (стенд).
    brain.dendrite_soma[M.slice] = dsoma

if plan_gain != 1.0 and "articulator" in brain.areas:
    brain.projection("motor -> articulator").weight.mul_(plan_gain)
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

ee = next(p for p in brain.plastic if p.name in ("articulator -> motor", "motor: E->E"))
pm = next(p for p in brain.plastic if p.name == "prefrontal -> motor")
print(f"рот E→E: макс {float(ee.weight.max()):.2f}, выше 0.5 — {int((ee.weight > 0.5).sum())}; "
      f"ПФК → рот: макс {float(pm.weight.max()):.2f}, выше 2 — {int((pm.weight > 2).sum())}")

words = Counter(lang.word(t) for t in lang.tokenize(text) if t.isalpha())
bins: list[torch.Tensor] = []
plan_bins: list[torch.Tensor] = []


def chunk(fired, _t):
    cells = fired[(fired >= OUT.start) & (fired < OUT.stop)] - OUT.start
    motor = torch.zeros(OUT.size)
    motor[cells] = 1.0
    bins.append(motor)
    cells = fired[(fired >= M.start) & (fired < M.stop)] - M.start
    plan = torch.zeros(M.size)
    plan[cells] = 1.0
    plan_bins.append(plan)


def say(word: str) -> str:
    held = lang.memory.get(word, allocate=False)

    if held is None:
        return "?"

    bins.clear()
    plan_bins.clear()
    trace = lang._memory_trace()
    trace.zero_()
    trace[held] = cfg.memory_mv
    lang._hold_memory()
    brain.on_chunk = chunk

    place = lang.concepts.get(word, allocate=False)

    # Говорит — ацетилхолин низкий (вспоминание, не запись): усталость рта есть.
    brain.set_acetylcholine(0.0)

    with brain.frozen():
        brain.run(1500.0)  # отдых: усталость рта от прошлого слова проходит (τ 1 с)
        brain.run(100.0)  # слово в памяти, рот молчит
        # Мозг думает это слово: горит его место в смысловой коре.
        if place is not None and place_mv > 0:
            brain.feel("association", place, place_mv)

        everyone = torch.arange(OUT.size, device=brain.device)
        # Команда «начать»: часы последовательности запускаются заново.
        lang._start_clock()

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
                        brain.feel(GO, everyone, go * (i + 1) / steps)
                        brain.run(4.0)

                        if no_stp:
                            # Проверка гипотезы: ресурс синапсов смысла не истощается.
                            brain.stp_x[lang.meaning.E.slice] = 1.0
                else:
                    brain.feel(GO, everyone, go)
                    brain.run(width)

                brain.feel(GO, None, 0.0)
                brain.run(period - width)

                if no_stp:
                    brain.stp_x[lang.meaning.E.slice] = 1.0

                t += period
        else:
            brain.feel(GO, everyone, go)
            brain.run(300.0)
            brain.feel(GO, None, 0.0)
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

        if OUT is not M:
            mine = torch.zeros(M.size, dtype=torch.bool)

            for c in set(word.lower()):
                if c in art.codes:
                    mine[art.code(c).cpu()] = True

            print("    план, своих/чужих клеток: " + " | ".join(
                f"{int((b.bool() & mine).sum())}/{int((b.bool() & ~mine).sum())}" for b in plan_bins[80:92]
            ))
            print("    план: " + " | ".join(
                f"{int(b.sum())}:" + "".join(c for c, _ in art.said(b.to(brain.device), 0.25)[:3])
                for b in plan_bins[80:95]
            ))

    for b in bins[80:]:
        said = art.said(b.to(brain.device), 0.5)
        letter = said[0][0] if said else ""

        if letter and letter != Articulators.PAUSE and (not out or out[-1] != letter):
            out.append(letter)

    return "".join(out)


right = first = two = 0
test = [w for w, _ in words.most_common(20)][: show or 20]

for w in test:
    got = say(w)
    word = w.lower()
    right += got == word
    first += got[:1] == word[:1]
    two += len(word) >= 2 and got[:2] == word[:2]
    print(f"  {w:12s} → {got}")

longer = sum(len(w) >= 2 for w in test)
print(f"верно {right}/{len(test)}; первая буква {first}/{len(test)}, первые две {two}/{longer}")
