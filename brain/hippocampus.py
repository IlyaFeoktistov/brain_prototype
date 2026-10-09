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

    def recall(self, cue: list[str], key=lambda t: t) -> tuple[list[str], float]:
        """
        Достраивание по части (CA3): эпизод, куда пришло больше всего
        активации от слов подсказки, — и что в нём шло после последнего из
        них. Активация слова делится между всеми эпизодами, где оно есть
        (эффект веера, Anderson 1974): «?» или «что» встречались в сотнях
        эпизодов и почти не различают их, «франции» — в нескольких, и её
        голос сильный. Сила — доля от наибольшей возможной активации (эпизод
        со всеми словами подсказки); слово, которого нет ни в одном эпизоде,
        связей в CA3 не имеет и никуда активацию не шлёт — в долю не входит.
        При равенстве побеждает свежий. Нечего достраивать — ([], 0).
        """

        words = {key(t) for t in cue}

        if not words:
            return [], 0.0

        seen = [[key(t) for t in episode.tokens] for episode in self.episodes]
        fan = dict.fromkeys(words, 0)

        for keys in seen:
            for word in words & set(keys):
                fan[word] += 1

        weight = {word: 1.0 / n for word, n in fan.items() if n}
        total = sum(weight.values())
        best, score = None, 0.0

        if not total:
            return [], 0.0

        for episode, keys in zip(reversed(self.episodes), reversed(seen)):
            share = sum(weight[w] for w in weight.keys() & set(keys)) / total

            if share > score:
                last = max(i for i, t in enumerate(keys) if t in words)

                # Достраивать нечего — эпизод кончается на подсказке.
                if last + 1 < len(keys):
                    best, score = episode.tokens[last + 1:], share

        return (list(best), score) if best else ([], 0.0)

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
