"""
Клетки последовательности (зона «sequence», флаг sequence_enabled): узнаёт
ли зона фразу целиком и строит ли мозг ответ фразой.

    PYTHONPATH=. .venv/bin/python data/tools/chunk_bench.py [load=путь] [save=путь] [key=value ...]

Без load — новый мозг small (с зоной) читает train-data.md дважды, без сна.
1. Активность зоны на словах (норма ~2%).
2. Избирательность: клетки зоны в конце фразы — та же фраза дважды (общих
   должно быть много) против разных фраз (мало).
3. Ответы на вопросы — первые слова и вся фраза.
Мозг заморожен на каждый вопрос: ответы не учат его.
"""

import sys
from pathlib import Path

import torch

from brain.config import BrainConfig
from brain.mind import Mind

load = save = None
ahp = 0.0
noseq = False
overrides = []

for kv in sys.argv[1:]:
    k, v = kv.split("=", 1)

    if k == "load":
        load = v
    elif k == "save":
        save = v
    elif k == "ahp":
        ahp = float(v)
    elif k == "noseq":
        noseq = v == "1"
    else:
        overrides.append((k, v))


def apply(cfg):
    for k, v in overrides:
        old = getattr(cfg, k)
        setattr(cfg, k, (v.lower() in ("1", "true")) if isinstance(old, bool) else type(old)(v))


if load:
    mind = Mind.load(load)
    apply(mind.cfg)
else:
    cfg = BrainConfig()
    cfg.size = "small"
    cfg.save_path = "/nonexistent/brain.pt"
    cfg.sequence_enabled = True
    apply(cfg)
    mind = Mind(cfg)
    mind.calibrate()
    mind.cfg.autosave = False
    text = Path("train-data.md").read_text(encoding="utf-8")

    for _ in range(2):
        mind.language.read_text(text, nap=False)

    if save:
        mind.save(save)

mind.cfg.autosave = False
lang, brain = mind.language, mind.brain

if noseq:
    # Контроль: путь «последовательность → смысл» к врождённому.
    path = next(p for p in brain.plastic if p.name == "sequence -> association")
    path.weight.copy_(path.birth if brain.cfg.learned_from_birth else path.innate_max)

if ahp > 0:
    # Медленная адаптация смысловой коры только при ответах (стенд фразы):
    # ahp=4 — 4 мВ / 1.5 с, как в «Фраза: первый замер».
    import math

    E = brain.areas["association"].E
    brain.slow_ahp_b[E.slice] = ahp
    brain.decay_slow_ahp[E.slice] = math.exp(-brain.cfg.dt_ms / 1500.0)
seq = brain.areas.get("sequence")
print(f"зона sequence: {seq.E.size if seq else 0} клеток")

SENTENCES = [
    "Столица Франции — Париж.",
    "Париж является столицей Франции.",
    "Столица Италии — Рим.",
    "Рим является столицей Италии.",
    "Человек пьёт воду.",
    "Человек ест пищу.",
    "Земля вращается вокруг Солнца.",
    "Луна вращается вокруг Земли.",
]


def heard(sentence: str) -> tuple[torch.Tensor, list[float]]:
    """Клетки зоны, стрелявшие на последних двух словах, и активность по словам."""

    seen, active = [], []
    original = lang.perceive

    def traced(token, *a, **kw):
        rates = original(token, *a, **kw)
        cells = rates[seq.E.slice] > 0
        active.append(float(cells.float().mean()))
        seen.append(cells)
        return rates

    lang.perceive = traced

    try:
        with brain.frozen():
            lang.experience(lang.tokenize(sentence), learn=False)
            lang.rest()
    finally:
        lang.perceive = original

    return (seen[-2] | seen[-3]) if len(seen) > 2 else seen[-1], active


def overlap(a: torch.Tensor, b: torch.Tensor) -> float:
    union = int((a | b).sum())
    return int((a & b).sum()) / union if union else 0.0


if seq is not None:
    codes, rates = [], []

    for s in SENTENCES:
        first, active = heard(s)
        again, _ = heard(s)
        codes.append((s, first, again))
        rates.extend(active)

    print(f"активность зоны на слове: средняя {100 * sum(rates) / len(rates):.1f}%, макс {100 * max(rates):.1f}%")
    same = [overlap(a, b) for _, a, b in codes]
    diff = [overlap(codes[i][1], codes[j][1]) for i in range(len(codes)) for j in range(i + 1, len(codes))]
    print(f"та же фраза дважды: {100 * sum(same) / len(same):.0f}% общих клеток; "
          f"разные фразы: {100 * sum(diff) / len(diff):.0f}% (макс {100 * max(diff):.0f}%)")

    for s, a, _ in codes:
        print(f"  {s[:34]:34s} клеток {int(a.sum())}")

QUESTIONS = [
    "Столица Франции —", "Париж является", "Человек пьёт", "Человек ест", "Земля вращается вокруг",
    "Рим является", "какая столица франции?", "что пьёт человек?", "вокруг чего вращается земля?",
]

for q in QUESTIONS:
    with brain.frozen():
        reply = lang.converse(q)

    print(f"  {q[:30]:30s} → {reply.text}")
