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

    def __init__(self, size: int, cells: int, seed: int, device, reuse: bool = False, keep: frozenset = frozenset()):
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
        # Места кончились — новое слово занимает место слова, слышанного
        # лишь однажды и давнее всех (забывается шлак: опечатки, случайные
        # имена), а не делит клетки с живыми словами: деление мест зашумляло
        # все предсказания. Повторённое не забывается: если шлака нет,
        # места делятся, как раньше, и full — сигнал, что мозгу мало места.
        # keep — слова, которые не забываются (служебные: конец мысли).
        self.reuse = reuse
        self.keep = keep
        # Сколько раз слово слышано; слышанные однажды — по порядку
        # встречи (первое — давнее всех, его и забыть).
        self.heard: dict[str, int] = {}
        self.once: dict[str, None] = {}
        # Забытые с прошлого раза: (слово, его клетки) — хозяин чистит
        # синапсы, словари и прочее, что знал о слове.
        self.evicted: list[tuple[str, torch.Tensor]] = []
        self.full = False
        self.forgotten = 0
        # Как слово было написано: место -> {форма: сколько раз}. Рот
        # говорит самую частую (долг: должен учиться рот, а не счётчик).
        self.forms: dict[str, dict[str, int]] = {}

    def __contains__(self, token: str) -> bool:
        return token in self.places

    def __len__(self) -> int:
        return len(self.places)

    def get(self, token: str, allocate: bool, near: torch.Tensor | None = None) -> torch.Tensor | None:
        """near — место слова, после которого это прозвучало (новое встанет рядом)."""

        place = self.places.get(token)

        if place is None and allocate:
            if self.reuse and len(self.places) >= self.size // self.cells:
                self._evict(near)
            place = self._allocate(near)
            self.places[token] = place

        # Встреча (allocate — слово слышно в опыте, а не просто справка).
        if place is not None and allocate:
            times = self.heard.get(token, 0) + 1
            self.heard[token] = times
            if times == 1:
                self.once[token] = None
            else:
                self.once.pop(token, None)

        return place

    def note_form(self, token: str, form: str) -> None:
        if token in self.places:
            seen = self.forms.setdefault(token, {})
            seen[form] = seen.get(form, 0) + 1

    def form(self, token: str) -> str:
        seen = self.forms.get(token)

        return max(seen, key=seen.get) if seen else token

    def _evict(self, near: torch.Tensor | None) -> None:
        """
        Забыть давнее из слов, слышанных однажды: его клетки свободны.
        Слово, рядом с которым ставится новое, не забывается.
        """

        anchor = next((t for t, p in self.places.items() if near is not None and p is near), None)
        victim = next((t for t in self.once if t not in self.keep and t != anchor), None)

        if victim is None:
            self.full = True
            return

        cells = self.places.pop(victim)
        self.heard.pop(victim, None)
        self.once.pop(victim, None)
        self.forms.pop(victim, None)
        self.usage[cells.cpu()] -= 1
        self._table = None
        self.evicted.append((victim, cells))
        self.forgotten += 1

    def table(self) -> tuple[list[str], torch.Tensor]:
        """Все места разом: слова и клетки [слов, cells] (для чтения спайков)."""

        if getattr(self, "_table", None) is None or len(self._table[0]) != len(self.places):
            tokens = list(self.places)
            cells = (
                torch.stack([self.places[t] for t in tokens])
                if tokens else torch.zeros(0, self.cells, dtype=torch.long, device=self.device)
            )
            self._table = (tokens, cells)

        return self._table

    def reset(self) -> None:
        self.heard.clear()
        self.once.clear()
        self.forms.clear()
        self.evicted.clear()
        self.places.clear()
        self._table = None
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
            "heard": self.heard,
            "once": list(self.once),
            "forgotten": self.forgotten,
            "forms": self.forms,
        }

    def load_state_dict(self, state: dict) -> None:
        self._table = None
        self.places = {key: place.to(self.device) for key, place in state["places"].items()}
        self.usage = state["usage"].clone()
        self.heard = dict(state.get("heard", {}))
        self.once = dict.fromkeys(state.get("once", []))
        self.forgotten = state.get("forgotten", 0)
        self.forms = {key: dict(seen) for key, seen in state.get("forms", {}).items()}
