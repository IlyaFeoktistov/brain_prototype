"""
Рука (ROADMAP чат-минимум 7.3, шаг 1): моторный лепет и копирование.

Новый мозг small с глазом и рукой: читает глазом слова корпуса (детекторы
букв просыпаются и учатся), потом лепечет — нажимает случайную клавишу,
буква появляется у пера, взгляд на ней (взгляд при письме ведёт перо);
связь «буквы → рука» учится по совпадению (engine.instar). Проверка —
копирование: видит букву на месте взгляда, ровный ток «давай» в руку (роль
базальных ганглиев на стенде) — какую клавишу нажал.

    PYTHONPATH=. .venv/bin/python tools/hand_bench.py [words=3000] [babble=2000] [go=10,15,20] [key=value ...]
"""

import random
import re
import sys
from pathlib import Path

import torch

from brain.body import Eye, Hand
from brain.config import BrainConfig
from brain.mind import Mind
from brain.world import Keyboard, Screen

opts = {"words": "3000", "babble": "2000", "go": "10,15,20", "load": "", "save": "", "rebabble": "0"}
cfg = BrainConfig()
cfg.size = "small"
cfg.save_path = "/nonexistent/brain.pt"
cfg.eye = cfg.hand = True

for kv in sys.argv[1:]:
    k, v = kv.split("=", 1)

    if k in opts:
        opts[k] = v
    else:
        old = getattr(cfg, k)
        setattr(cfg, k, (v == "1") if isinstance(old, bool) else type(old)(v))

mind = Mind.load(opts["load"]) if opts["load"] else Mind(cfg)
mind.cfg.autosave = False
cfg, brain = mind.cfg, mind.brain
H = brain.areas["hand"].output
keyboard = Keyboard()
hand = Hand(len(keyboard), cfg.hand_cells)
assert hand.keys * hand.cells <= H.size, (hand.keys * hand.cells, H.size)
i = [s.name for s in brain.sensors].index("eye")
eye = Eye(cfg, brain.sensor_offsets[i], brain.sensors[i].size, brain.device)
screen = Screen(cfg.retina_rows, cfg.retina_cols)
PEN = 20  # место взгляда (px): буква у пера
PATHS = ("granule -> hand", "granule -> hand: торможение")
G = brain.areas["granule"].E
rng = random.Random(0)

def babble() -> None:
    """Лепет: только клавиши букв (цифры и знаки — так же, но буквы — проверка)."""

    letters = [k for k in range(len(keyboard)) if keyboard.type(k).isalpha()]

    for n in range(int(opts["babble"])):
        key = rng.choice(letters)
        brain.feel("hand", hand.code(key).to(brain.device), 30.0)
        picture = screen.picture(keyboard.type(key), PEN)
        # Палец на клавише, глаз смотрит на появившуюся букву всю фиксацию.
        rates = brain.observe(cfg.fixation_ms, eye.look(picture))
        brain.feel("hand", None, 0.0)
        brain.error_learning(PATHS, rates, hand.code(key).to(brain.device), cfg.hand_rate)
        brain.run(cfg.keystroke_ms)

    print(f"лепет: {opts['babble']} нажатий")


if opts["load"] and opts["rebabble"] == "1":
    # Тот же прочитавший мозг, связь «буквы → рука» — заново с рождения.
    for name in PATHS:
        proj = brain.projection(name)
        proj.weight.uniform_(*proj.innate_range)
    babble()

    if opts["save"]:
        mind.save(opts["save"])

if not opts["load"]:
    text = "\n".join(p.read_text(encoding="utf-8") for p in sorted(Path("data/corpus/qa").glob("*.txt")))
    vocab = [w for w in re.findall(r"[а-яё]+", text.lower()) if 2 <= len(w) <= 12]

    for n in range(int(opts["words"])):
        rates = brain.observe(cfg.fixation_ms, eye.look(screen.picture(rng.choice(vocab))))
        brain.run(30.0)
        brain.homeostasis(rates)
        brain.hebbian(cfg.hebbian_rate)
        brain.inhibitory_plasticity(rates)

        if (n + 1) % 50 == 0:
            brain.synaptic_scaling()

    print(f"прочитано глазом {opts['words']} слов")

    babble()

    if opts["save"]:
        mind.save(opts["save"])

proj = brain.projection("granule -> hand")
neg = brain.projection("granule -> hand: торможение")
print(f"гранулярные → рука: макс {float(proj.weight.max()):.2f}, выше 0.5 — {int((proj.weight > 0.5).sum())}; "
      f"торможение мин {float(neg.weight.min()):.2f}, сильнее 0.5 — {int((neg.weight < -0.5).sum())}")
TEST = "аоеиклмнвстрдпуябгжзчшщхцфыэюйьъ"
L = brain.areas["letters"].E

with brain.frozen():
    brain.run(300.0)
    g = brain.observe(cfg.fixation_ms, eye.look(screen.picture("о", PEN)))[G.slice]
print(f"гранулярные на букву: {int((g > 0).sum())} из {G.size} клеток, {float(g[g > 0].mean()) if (g > 0).any() else 0:.0f} Гц")


def seen(c: str) -> torch.Tensor:
    with brain.frozen():
        brain.run(300.0)
        return brain.observe(cfg.fixation_ms, eye.look(screen.picture(c, PEN)))[L.slice].cpu()


cos = lambda a, b: float(a @ b / (a.norm() * b.norm() + 1e-9))
refs = {c: seen(c) for c in TEST}
ceiling = sum(max(TEST, key=lambda d: cos(seen(c), refs[d])) == c for c in TEST)
print(f"потолок: зона букв различает {ceiling}/{len(TEST)} букв на месте взгляда (ближайший образец)")

for go in (float(g) for g in opts["go"].split(",")):
    right = silent = 0
    wrong = []

    for c in TEST:
        with brain.frozen():
            brain.run(300.0)
            # «Давай» нарастает (гонка к порогу): первые 100 мс взгляда — без
            # него, потом ток растёт до go за 100 мс.
            raster = eye.look(screen.picture(c, PEN))
            steps = raster.shape[0]
            everyone = torch.arange(H.size, device=brain.device)
            brain.counts.zero_()
            brain.run(100.0, raster)

            for j in range(5):
                brain.feel("hand", everyone, go * (j + 1) / 5)
                part = raster[int(steps * (100 + 25 * j) / cfg.fixation_ms):]
                brain.run(25.0, part)

            brain.feel("hand", None, 0.0)
            rates = brain.counts[: brain.n].float().clone()

        counts = rates[H.slice].cpu()
        blocks = counts[: hand.keys * hand.cells].view(hand.keys, hand.cells).sum(1)

        if not float(blocks.max()):
            silent += 1
            continue

        got = keyboard.type(int(blocks.argmax()))
        right += got == c

        if got != c:
            wrong.append(f"{c}→{got}")

    print(f"копирование, ток {go:.0f} мВ: верно {right}/{len(TEST)}, молчит {silent}; ошибки: {' '.join(wrong[:12])}")
