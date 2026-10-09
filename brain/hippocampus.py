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
    # Кончилась ли фраза паузой. Нет — на ней ответил мозг (реплика
    # человека): во сне конец строки после неё не проигрывается, а то
    # мозг выучит «после вопроса — тишина».
    ended: bool = True


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

    def store(self, tokens: list[str], salience: float, ended: bool = True) -> None:
        if len(tokens) < 2:
            return

        self.episodes.append(Episode(list(tokens), salience, ended=ended))

        if len(self.episodes) > self.capacity:
            # Забывается самое выученное (наименее неожиданное).
            weakest = min(range(len(self.episodes)), key=lambda i: self.episodes[i].salience)
            self.episodes.pop(weakest)

    def extend(self, tokens: list[str], more: list[str], salience: float, ended: bool) -> None:
        """
        Эпизод не кончился, а продолжился: последний эпизод с этими словами
        дописывается (временной контекст тот же). Нет такого — новый.
        """

        for episode in reversed(self.episodes[-10:]):
            if episode.tokens == tokens:
                episode.tokens = tokens + list(more)
                episode.salience = max(episode.salience, salience)
                episode.ended = ended
                return

        self.store(tokens + list(more), salience, ended)

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
            "episodes": [(e.tokens, e.salience, e.replays, e.ended) for e in self.episodes],
            "random": self.random.getstate(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.episodes = [Episode(list(e[0]), e[1], e[2], *(e[3:] or [True])) for e in state["episodes"]]
        self.random.setstate(state["random"])
