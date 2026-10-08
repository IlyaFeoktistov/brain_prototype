"""
Категории (зона-хаб, как височный полюс): обобщение по ролям слов.

Слово A вступает в категорию, которая уже предсказывает то, что пришло
после A; категории слова A заодно начинают предсказывать новое. Так
«кошка, собака, лошадь» (после них «спит», «ест») становятся одной
категорией, и услышав «корова спит», мозг сам ждёт «корова ест». Никто
категорий не задаёт — они возникают из опыта; без правил языка.

Пока модель (как гиппокамп), не нейроны: членство и предсказания —
веса 0..1, растут к 1 с каждой встречей (rate × (1 − w)).
"""

from __future__ import annotations


class Categories:
    def __init__(self, rate: float):
        self.rate = rate
        # слово -> {категория: вес членства}
        self.member: dict[str, dict[int, float]] = {}
        # категория -> {слово: вес предсказания}
        self.predicts: dict[int, dict[str, float]] = {}

    def __len__(self) -> int:
        return len(self.predicts)

    def learn(self, a: str, b: str) -> None:
        """Переход A → B."""

        rate = self.rate
        mine = self.member.setdefault(a, {})

        # Категории A теперь предсказывают и B.
        for unit in mine:
            weights = self.predicts[unit]
            weights[b] = weights.get(b, 0.0) + rate * (1.0 - weights.get(b, 0.0))

        # A вступает в категорию, которая сильнее всех уже предсказывает B.
        best = max(
            (unit for unit, weights in self.predicts.items() if b in weights and unit not in mine),
            key=lambda unit: self.predicts[unit][b],
            default=None,
        )

        if best is not None:
            mine[best] = mine.get(best, 0.0) + rate * (1.0 - mine.get(best, 0.0))
        elif not mine:
            unit = len(self.predicts)
            self.predicts[unit] = {b: rate}
            mine[unit] = rate

    def expect(self, a: str) -> dict[str, float]:
        """Что предсказывают категории слова A: слово -> сила (сумма = 1)."""

        scores: dict[str, float] = {}

        for unit, membership in self.member.get(a, {}).items():
            for word, weight in self.predicts[unit].items():
                scores[word] = scores.get(word, 0.0) + membership * weight

        total = sum(scores.values())

        return {word: score / total for word, score in scores.items()} if total else {}

    def state_dict(self) -> dict:
        return {"member": self.member, "predicts": self.predicts}

    def load_state_dict(self, state: dict) -> None:
        self.member = {word: dict(units) for word, units in state["member"].items()}
        self.predicts = {int(unit): dict(words) for unit, words in state["predicts"].items()}
