"""
Понятия: у каждого слова своё место в смысловой коре.

Как «клетки понятий» в мозге (Quiroga 2005: нейрон отвечает на одного
человека — на фото, имя, голос), каждому слову при первой встрече
выделяется своя небольшая группа клеток ассоциативной коры — из
наименее занятых. Слово будит ровно эту группу, а не случайный разброс
клеток. Связи между словами
учатся между группами, и посторонние клетки в предсказание не лезут.

Без правил какого-либо языка: место получает слово, как оно звучит.
Что «франция» и «франции» (или «France») — одно, мозг должен выучить
сам по общим контекстам. [замер] место на основу (русский стеммер)
сливало контексты форм: после «франции» ждал «находится» (из
«Франция находится…») — ранги фактов 3–6 вместо 1–3.

Почему: со случайной проводкой «слух → смысл» слово будило сотни
посторонних клеток; выученные на новом тексте связи от них забивали
предсказание старого (см. DESIGN «Интерференция»).
"""

from __future__ import annotations

import torch


class Concepts:
    """Слово -> клетки смысловой коры (локальные индексы E)."""

    def __init__(self, size: int, cells: int, seed: int, device):
        self.size = size
        self.cells = min(cells, size)
        self.device = device
        self.generator = torch.Generator().manual_seed(seed)
        self.places: dict[str, torch.Tensor] = {}
        # Сколько понятий уже живёт в каждой клетке.
        self.usage = torch.zeros(size)

    def __contains__(self, token: str) -> bool:
        return token in self.places

    def __len__(self) -> int:
        return len(self.places)

    def get(self, token: str, allocate: bool) -> torch.Tensor | None:
        place = self.places.get(token)

        if place is None and allocate:
            place = self._allocate()
            self.places[token] = place

        return place

    def reset(self) -> None:
        self.places.clear()
        self.usage.zero_()

    def _allocate(self) -> torch.Tensor:
        """Наименее занятые клетки; среди равных — случайные."""

        jitter = torch.rand(self.size, generator=self.generator) * 0.5
        chosen = torch.topk(-(self.usage + jitter), self.cells).indices
        self.usage[chosen] += 1

        return chosen.to(self.device)

    def state_dict(self) -> dict:
        return {
            "places": {key: place.cpu() for key, place in self.places.items()},
            "usage": self.usage,
        }

    def load_state_dict(self, state: dict) -> None:
        self.places = {key: place.to(self.device) for key, place in state["places"].items()}
        self.usage = state["usage"].clone()
