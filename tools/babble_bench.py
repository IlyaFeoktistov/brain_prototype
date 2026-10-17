"""
Рот по буквам, шаг 1 — лепет (ROADMAP чат-минимум 7.1): рот произносит
символы, ухо слышит, слух → рот учится. Проверка: услышанный символ будит
свою артикуляцию сам (зеркало), и что рот «повторяет про себя» на словах —
по окнам 20 мс.

    PYTHONPATH=. .venv/bin/python tools/babble_bench.py [rounds=5] [cap=6] [key=value ...]

Новый мозг small без чтения: лепет — до слов, как у младенца.
"""

import sys
from pathlib import Path

import torch

from brain.config import BrainConfig
from brain.mind import Mind

rounds = 5
cap = None
overrides = []

for kv in sys.argv[1:]:
    k, v = kv.split("=", 1)

    if k == "rounds":
        rounds = int(v)
    elif k == "cap":
        cap = float(v)
    else:
        overrides.append((k, v))

cfg = BrainConfig()
cfg.size = "small"
cfg.save_path = "/nonexistent/brain.pt"
cfg.mouth_letters = True

for k, v in overrides:
    old = getattr(cfg, k)
    setattr(cfg, k, (v.lower() in ("1", "true")) if isinstance(old, bool) else type(old)(v))

mind = Mind(cfg)
mind.calibrate()
mind.cfg.autosave = False
lang, brain = mind.language, mind.brain
M = brain.areas["motor"].E
text = Path("train-data.md").read_text(encoding="utf-8")
alphabet = sorted({c.lower() for c in text if not c.isspace()})
print(f"алфавит: {len(alphabet)} символов, рот {M.size} клеток, артикуляция {cfg.articulator_cells}")
art = lang.articulators

for s in alphabet + [art.LOUD]:
    art.code(s)


def mirror(symbol: str) -> tuple[list, int]:
    with brain.frozen():
        sound = symbol * cfg.babble_repeat  # как при лепете
        rates = brain.observe(lang.ear.duration(sound), lang.ear.encode(sound))
        brain.run(cfg.gap_ms * 5)

    motor = rates[M.slice]
    return art.said(motor, 0.25), int((motor > 0).sum())


def score(tag: str) -> None:
    hits, active = 0, []

    for s in alphabet:
        said, n = mirror(s)
        active.append(n)
        hits += bool(said) and said[0][0] == s

    print(f"{tag}: зеркало {hits}/{len(alphabet)} символов первым; рот стреляет {sum(active) / len(active):.0f} клеток в среднем")


score("до лепета")
path = next(p for p in brain.plastic if p.name == "auditory -> motor")

if cap is not None:
    path.cap = cap

for r in range(rounds):
    lang.babble(alphabet, rounds=1, seed=r)
    score(f"лепет ×{r + 1}")

print(f"вес слух → рот: средний {float(path.weight.mean()):.2f}, макс {float(path.weight.max()):.2f}")

# Слово целиком: что рот повторяет про себя по окнам 20 мс.
bins: list[torch.Tensor] = []


def chunk(fired, _t):
    cells = fired[(fired >= M.start) & (fired < M.stop)] - M.start
    motor = torch.zeros(M.size)
    motor[cells] = 1.0
    bins.append(motor)


for word in ["москва", "столица", "вода", "Париж"]:
    bins.clear()
    brain.on_chunk = chunk

    with brain.frozen():
        brain.run(lang.ear.duration(word), lang.ear.encode(word))
        brain.run(40.0)

    brain.on_chunk = None
    heard = [f"{int(b.sum())}кл " + (" ".join(f"{s}{v:.2f}" for s, v in art.said(b.to(brain.device), 0.12)[:3]) or "—") for b in bins]
    print(f"  {word:8s} → " + " | ".join(heard))
