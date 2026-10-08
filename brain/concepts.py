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

import math

import torch


class Concepts:
    """
    Слово -> клетки смысловой коры (локальные индексы E).

    Зона — карта: клетки лежат сеткой cols × rows, место слова —
    компактное пятно. Новое слово ставится вокруг места слова, после
    которого прозвучало (ближайшие свободные клетки), — слова,
    встречающиеся вместе, собираются по соседству.
    """

    def __init__(self, size: int, cells: int, seed: int, device):
        self.size = size
        self.cells = min(cells, size)
        self.device = device
        self.generator = torch.Generator().manual_seed(seed)
        self.places: dict[str, torch.Tensor] = {}
        # Сколько понятий уже живёт в каждой клетке.
        self.usage = torch.zeros(size)
        self.cols = max(1, math.ceil(math.sqrt(size)))
        index = torch.arange(size)
        self.xy = torch.stack([index % self.cols, index // self.cols], 1).float()

    def __contains__(self, token: str) -> bool:
        return token in self.places

    def __len__(self) -> int:
        return len(self.places)

    def get(self, token: str, allocate: bool, near: torch.Tensor | None = None) -> torch.Tensor | None:
        """near — место слова, после которого это прозвучало (новое встанет рядом)."""

        place = self.places.get(token)

        if place is None and allocate:
            place = self._allocate(near)
            self.places[token] = place

        return place

    def reset(self) -> None:
        self.places.clear()
        self.usage.zero_()

    def _allocate(self, near: torch.Tensor | None) -> torch.Tensor:
        """
        Ближайшие к якорю наименее занятые клетки: сначала свободные
        (занятость весит больше любого расстояния), среди них — ближние.
        Якорь — центр места соседнего слова или случайная свободная клетка.
        """

        if near is not None and len(near):
            anchor = self.xy[near.cpu()].mean(0)
        else:
            free = (self.usage == self.usage.min()).nonzero().flatten()
            pick = free[torch.randint(len(free), (1,), generator=self.generator)]
            anchor = self.xy[pick[0]]

        distance = (self.xy - anchor).norm(dim=1)
        jitter = torch.rand(self.size, generator=self.generator) * 0.5
        score = self.usage * (2 * self.cols) + distance + jitter
        chosen = torch.topk(-score, self.cells).indices
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
