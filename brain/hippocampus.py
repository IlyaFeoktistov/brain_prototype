"""
Гиппокамп: эпизодическая память и проигрывание во сне.

У человека гиппокамп запоминает пережитое с одного раза, а во сне
многократно проигрывает эпизоды коре — и знание постепенно переходит
в кору (системная консолидация). Неожиданные и эмоционально значимые
эпизоды проигрываются чаще.

Здесь гиппокамп — модель, а не спайковая зона: он хранит фразы
(последовательности слов) и их «важность» — сколько дофамина фраза
вызвала. Кора же учится по-настоящему: проигрывание — это обычное
переживание фразы с STDP и дофамином от ошибки предсказания.
Уже выученная фраза почти не даёт дофамина, её важность падает,
и гиппокамп её со временем забывает — в коре знание остаётся.
"""

from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass
class Episode:
    tokens: list[str]
    # Сколько дофамина фраза вызвала в последний раз (неожиданность).
    salience: float
    replays: int = 0


class Hippocampus:
    def __init__(self, capacity: int, priority: float, seed: int = 0):
        self.capacity = capacity
        # Насколько проигрывание предпочитает неожиданное:
        # 0 — все фразы поровну, 1 — пропорционально дофамину.
        self.priority = priority
        self.episodes: list[Episode] = []
        self.random = random.Random(seed)

    def __len__(self) -> int:
        return len(self.episodes)

    def store(self, tokens: list[str], salience: float) -> None:
        if len(tokens) < 2:
            return

        self.episodes.append(Episode(list(tokens), salience))

        if len(self.episodes) > self.capacity:
            # Забывается самое выученное (наименее неожиданное).
            weakest = min(range(len(self.episodes)), key=lambda i: self.episodes[i].salience)
            self.episodes.pop(weakest)

    def sample(self) -> Episode | None:
        if not self.episodes:
            return None

        weights = [(e.salience + 0.1) ** self.priority for e in self.episodes]

        return self.random.choices(self.episodes, weights=weights, k=1)[0]

    def forget_consolidated(self, threshold: float) -> int:
        """Забыть эпизоды, которые уже почти не удивляют кору."""

        before = len(self.episodes)
        self.episodes = [
            e for e in self.episodes if e.replays == 0 or e.salience > threshold
        ]

        return before - len(self.episodes)

    def state_dict(self) -> dict:
        return {
            "episodes": [(e.tokens, e.salience, e.replays) for e in self.episodes],
            "random": self.random.getstate(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.episodes = [Episode(list(t), s, r) for t, s, r in state["episodes"]]
        self.random.setstate(state["random"])
