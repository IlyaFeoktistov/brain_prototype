"""
Тема между репликами и смена слова той же роли (ворота рабочей памяти).

Прочитанный мозг (load=…, по умолчанию data/measure/sy0-read.pt) проходит
цепочки реплик; каждая цепочка — с только что загруженного мозга. После
каждой реплики — ответ, «подумал» и какие слова держат цель (колонки ПФК)
в паузе перед ответом и какие категории у слов вопроса.

    PYTHONPATH=. .venv/bin/python data/tools/pbwm_bench.py [load=путь] [key=value ...]
"""

import sys

import torch

from brain.commands import deliberation_line
from brain.mind import Mind

CHAINS = [
    ["какая столица франции?", "а россии?", "какая столица италии?"],
    ["какая столица италии?", "какая столица франции?"],
    ["что пьёт человек?", "а что он ест?"],
    ["а россии?"],
]

load = "data/measure/sy0-read.pt"
overrides = []

for kv in sys.argv[1:]:
    k, v = kv.split("=", 1)

    if k == "load":
        load = v
    else:
        overrides.append((k, v))


def fresh() -> Mind:
    mind = Mind.load(load)
    mind.cfg.autosave = False

    for k, v in overrides:
        old = getattr(mind.cfg, k)
        setattr(mind.cfg, k, (v == "1") if isinstance(old, bool) else type(old)(v))

    mind.brain.recompile()
    return mind


def held_goal(lang) -> list[str]:
    """Слова, чьи колонки цели горят сейчас (по последнему шагу)."""

    if lang.last_rates is None or not lang._goal_on():
        return []

    goal = lang._area(lang.last_rates, lang.goal)
    words = []

    for word, column in lang.goal_columns.items():
        cells = slice(column * lang._goal_cells, (column + 1) * lang._goal_cells)

        if float(goal[cells].mean()) > 0:
            words.append(word)

    return words


for k, v in overrides:
    print(f"{k} = {v}")

for chain in CHAINS:
    mind = fresh()
    lang = mind.language
    print("—" * 60)

    for text in chain:
        seen: list[list[str]] = []
        spoken = lang._articulated

        def traced(rates, heard, spoken=spoken):
            seen.append(held_goal(lang))
            return spoken(rates, heard)

        lang._articulated = traced
        reply = lang.converse(text)
        lang._articulated = spoken
        print(f"я: {text}\n   мозг: {reply.text}\n   {deliberation_line(reply)}")
        print(f"   цель в паузе: {seen[0] if seen else '—'}")
