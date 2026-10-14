"""
Детекторы букв (зона «letters», V4/LO): выучивает ли чтение глазом клетки,
различающие буквы. Новый мозг small с глазом смотрит на слова корпуса
(картинка, фиксация, саккада), после каждой фиксации — чистый STDP
(brain.hebbian), гомеостаз, масштабирование. До и после чтения:
одна буква со сдвигом на shift px (должна остаться похожей) против разных
букв на одном месте (должны различаться).

    PYTHONPATH=. .venv/bin/python data/tools/letters_bench.py [words=1500] [rate=10] [shift=2] [key=value ...]
"""

import random
import re
import sys
from pathlib import Path

import torch

from brain.body import Eye
from brain.config import BrainConfig
from brain.mind import Mind
from brain.world import Screen

opts = {"words": "1500", "rate": "10", "shift": "2", "save": ""}
cfg = BrainConfig()
cfg.size = "small"
cfg.save_path = "/nonexistent/brain.pt"
cfg.eye = True

from dataclasses import replace

from brain.config import eye_areas, eye_pathways

areas, paths = eye_areas(), eye_pathways()

for kv in sys.argv[1:]:
    k, v = kv.split("=", 1)

    if k in opts:
        opts[k] = v
    elif k.startswith("letters."):
        # Параметр зоны букв (стенд): letters.pool_mv=6
        field = k.split(".", 1)[1]
        areas = [replace(a, **{field: type(getattr(a, field))(v)}) if a.name == "letters" else a for a in areas]
    elif k.startswith("path."):
        # Параметр пути V1 → буквы (стенд): path.cap=2, path.weight=0/2
        field = k.split(".", 1)[1]
        value = tuple(float(x) for x in v.split("/")) if "/" in v else type(getattr(paths[0], field))(v)
        paths = [replace(paths[0], **{field: value})]
    else:
        old = getattr(cfg, k)
        setattr(cfg, k, type(old)(v))

cfg.areas = cfg.areas + areas
cfg.pathways = cfg.pathways + paths
mind = Mind(cfg)
brain = mind.brain
mind.cfg.autosave = False
L = brain.areas["letters"].E
i = [s.name for s in brain.sensors].index("eye")
eye = Eye(cfg, brain.sensor_offsets[i], brain.sensors[i].size, brain.device)
screen = Screen(cfg.retina_rows, cfg.retina_cols)
shift = int(opts["shift"])


def letters(picture: torch.Tensor) -> torch.Tensor:
    with brain.frozen():
        brain.run(100.0)
        rates = brain.observe(cfg.fixation_ms, eye.look(picture))
    return rates[L.slice].cpu()


def at(text: str, px: int) -> torch.Tensor:
    pic = screen.picture(text)
    out = torch.ones_like(pic)
    out[:, px:] = pic[:, : pic.shape[1] - px]
    return out


def cosine(a, b) -> float:
    na, nb = a.norm(), b.norm()
    return float(a @ b / (na * nb)) if na and nb else 0.0


LETTERS = "аоеиклмнвстрдпуябгжзчшщхцфыэюйьъ"


V1 = brain.areas["visual"].E
CX = brain.areas["complex"].E
NAMES = ("V1", "сложные", "буквы")


def zones(picture: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    with brain.frozen():
        brain.run(100.0)
        rates = brain.observe(cfg.fixation_ms, eye.look(picture))
    return rates[V1.slice].cpu(), rates[CX.slice].cpu(), rates[L.slice].cpu()


def recognition(label: str) -> None:
    """
    Узнавание буквы по ближайшему образцу: образец — буква на 20 px (два
    показа), проба — новый показ со сдвигом 0 / 1 / 2 / 4 px; доля верных
    среди LETTERS (наугад — 1 / len). V1 против зоны букв.
    """

    refs = {c: [sum(z) for z in zip(zones(at(c, 20)), zones(at(c, 20)))] for c in LETTERS}
    line = []

    for dx in (0, 1, 2, 4):
        hits = [0] * len(NAMES)

        for c in LETTERS:
            probe = zones(at(c, 20 + dx))

            for z in range(len(NAMES)):
                best = max(LETTERS, key=lambda d: cosine(probe[z], refs[d][z]))
                hits[z] += best == c

        line.append(f"{dx} px: " + ", ".join(f"{n} {h}" for n, h in zip(NAMES, hits)))

    print(f"{label} узнавание из {len(LETTERS)} — " + "; ".join(line))


def measure(label: str) -> None:
    codes = {c: letters(at(c, 20)) for c in LETTERS}
    moved = {c: letters(at(c, 20 + shift)) for c in LETTERS}
    same = [cosine(codes[c], moved[c]) for c in LETTERS]
    diff = [cosine(codes[a], codes[b]) for a in LETTERS for b in LETTERS if a < b]
    active = sum(int((v > 0).sum()) for v in codes.values()) / len(codes)
    word = letters(screen.picture("человек"))
    print(f"{label}: буква сдвиг {shift} px {sum(same) / len(same):.2f}, разные буквы "
          f"{sum(diff) / len(diff):.2f}; клеток на букву {active:.0f}, на слово {int((word > 0).sum())} из {L.size}")


text = "\n".join(p.read_text(encoding="utf-8") for p in sorted(Path("data/corpus/qa").glob("*.txt")))
vocab = [w for w in re.findall(r"[а-яё]+", text.lower()) if 2 <= len(w) <= int(cfg.retina_cols // 7)]
rng = random.Random(0)
measure("до чтения")
recognition("до чтения")
proj = next(p for p in brain.plastic if p.name == "complex -> letters")

for n in range(int(opts["words"])):
    rates = brain.observe(cfg.fixation_ms, eye.look(screen.picture(rng.choice(vocab))))
    brain.run(30.0)  # саккада
    brain.homeostasis(rates)
    brain.hebbian(float(opts["rate"]))
    brain.inhibitory_plasticity(rates)

    if (n + 1) % 50 == 0:
        brain.synaptic_scaling()

    if (n + 1) % 500 == 0:
        lateral = brain.projection("letters: торможение").weight
        print(f"  {n + 1} слов: вес V1 → буквы средний {float(proj.weight.mean()):.2f}, "
              f"у потолка {int((proj.weight > 0.9 * proj.cap).sum())}; торможение среднее "
              f"{float(lateral.mean()):.2f}, сильнее 1 мВ {int((lateral < -1).sum())}")

measure(f"после {opts['words']} слов")
recognition("после")

if opts["save"]:
    mind.save(opts["save"])
