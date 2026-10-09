"""
Стенд отладки («ЭЭГ»): показатели здоровья мозга после каждой фразы.

Это наблюдение, а не поведение: мозг по ним ничего не решает. Нормы —
чтобы ловить поломки сразу, а не по обвалу экзамена (так нашёлся
эпилептический очаг частых слов: кора стреляла в тишине на 240 Гц).

Что считается (по зонам, отдельно E и I):
  - доля активных клеток на слове и в паузе после фразы;
  - средняя частота (Гц) на слове и в паузе;
  - вход клетки по пластичным путям относительно врождённого;
  - потерянные спайки (переполнение таблицы событий).

На слове и в паузе — одна операция на видеокарте, сверка с нормами —
одна синхронизация на фразу.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import torch


# Нормы. Не подбирались под результат — это границы «явно сломано».
# Кора в паузе после фразы (150 мс тишины) почти молчит: шум — одиночные
# спайки (средняя частота зоны < 1 Гц), очаг частых слов — 546 клеток на
# 240 Гц, ~12 Гц на всю смысловую кору 20 тыс. нейронов.
PAUSE_RATE_MAX = 3.0
# На слове активна доля около target_activity зоны; взрыв — в разы больше.
WORD_ACTIVE_FACTOR = 5.0
# Вход клетки по связям внутри зоны (E→E — там замыкаются очаги) —
# бюджет масштабирования 1.5× врождённого; очаг давал 55×. Пути между
# зонами только показываются: у «ПФК → смысл» врождённые веса почти
# нулевые, выученный контекст по замыслу вырастает из них в разы.
INPUT_RATIO_MAX = 10.0
# Зоны, которые в тишине должны молчать (рабочая память держит след и
# имеет право звучать; ядра и чувства — свои правила).
QUIET_KINDS = ("cortex",)
QUIET_SKIP = ("limbic", "vta", "amygdala", "category", "lc")


@dataclass
class Snapshot:
    """Показатели за одну фразу."""

    sentence: str
    words: int
    # зона -> (доля активных E, частота E, доля активных I, частота I)
    word: dict[str, tuple[float, float, float, float]] = field(default_factory=dict)
    pause: dict[str, tuple[float, float, float, float]] = field(default_factory=dict)
    # путь -> (макс, среднее) входа клетки к врождённому
    inputs: dict[str, tuple[float, float]] = field(default_factory=dict)
    dropped: float = 0.0
    problems: list[str] = field(default_factory=list)


class Health:
    def __init__(self, brain, cfg, log_size: int = 50):
        self.brain = brain
        self.cfg = cfg
        self.device = brain.device
        self.names = list(brain.areas)
        self.targets = {spec.name: spec.target_activity for spec in cfg.areas}
        self.kinds = {name: area.kind for name, area in brain.areas.items()}
        # Каждому нейрону — номер группы: зона × {E, I}.
        group = torch.full((brain.n,), -1, dtype=torch.long, device=self.device)

        for i, area in enumerate(brain.areas.values()):
            group[area.E.slice] = 2 * i
            group[area.I.slice] = 2 * i + 1

        self.group = group.clamp(min=0)
        self.valid = (group >= 0).float()
        self.sizes = torch.zeros(2 * len(self.names), device=self.device).index_add_(0, self.group, self.valid)
        self.sizes.clamp_(min=1.0)
        self._reset()
        self._dropped = float(brain.dropped)
        self.last: Snapshot | None = None
        # Фразы с нарушениями (последние log_size).
        self.log: deque[Snapshot] = deque(maxlen=log_size)
        self.sentences = 0

    def _reset(self) -> None:
        groups = 2 * len(self.names)
        self._word_active = torch.zeros(groups, device=self.device)
        self._word_rate = torch.zeros(groups, device=self.device)
        self._pause_active = torch.zeros(groups, device=self.device)
        self._pause_rate = torch.zeros(groups, device=self.device)
        self._words = 0

    def _by_group(self, values: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(self.sizes).index_add_(0, self.group, values * self.valid) / self.sizes

    def word(self, rates: torch.Tensor) -> None:
        self._word_active += self._by_group((rates > 0).float())
        self._word_rate += self._by_group(rates)
        self._words += 1

    def pause(self, rates: torch.Tensor) -> None:
        self._pause_active = self._by_group((rates > 0).float())
        self._pause_rate = self._by_group(rates)

    def _inputs(self) -> torch.Tensor:
        """Вход клетки к врождённому по каждому пластичному пути: макс, среднее."""

        values = []

        for proj in self.brain.plastic:
            birth = proj.birth.float().sum(1).clamp(min=1e-6)
            ratio = proj.weight.sum(1) / birth
            values.append(torch.stack([ratio.max(), ratio.mean()]))

        return torch.stack(values) if values else torch.zeros(0, 2, device=self.device)

    def sentence(self, text: str) -> Snapshot:
        """Фраза кончилась: сверить с нормами (одна синхронизация)."""

        words = max(1, self._words)
        packed = torch.cat([
            self._word_active / words, self._word_rate / words,
            self._pause_active, self._pause_rate,
            self._inputs().flatten(), self.brain.dropped,
        ]).tolist()
        g = 2 * len(self.names)
        word_active, word_rate = packed[:g], packed[g:2 * g]
        pause_active, pause_rate = packed[2 * g:3 * g], packed[3 * g:4 * g]
        inputs = packed[4 * g:-1]
        dropped = packed[-1]

        snap = Snapshot(text, self._words)

        for i, name in enumerate(self.names):
            e, inh = 2 * i, 2 * i + 1
            snap.word[name] = (word_active[e], word_rate[e], word_active[inh], word_rate[inh])
            snap.pause[name] = (pause_active[e], pause_rate[e], pause_active[inh], pause_rate[inh])

        for j, proj in enumerate(self.brain.plastic):
            snap.inputs[proj.name] = (inputs[2 * j], inputs[2 * j + 1])

        snap.dropped = dropped - self._dropped
        self._dropped = dropped
        snap.problems = self._problems(snap)
        self.last = snap
        self.sentences += 1

        if snap.problems:
            self.log.append(snap)

        self._reset()

        return snap

    def _problems(self, snap: Snapshot) -> list[str]:
        problems = []

        for name, (active, rate, _, _) in snap.pause.items():
            if self.kinds[name] in QUIET_KINDS and name not in QUIET_SKIP and rate > PAUSE_RATE_MAX:
                problems.append(f"{name}: в тишине активно {active:.1%} клеток ({rate:.1f} Гц в среднем) — очаг?")

        for name, (active, _, _, _) in snap.word.items():
            target = self.targets.get(name, 0.0)
            if target > 0 and active > WORD_ACTIVE_FACTOR * target:
                problems.append(f"{name}: на слове активно {active:.1%} клеток при норме {target:.0%} — взрыв?")

        for name, (top, _) in snap.inputs.items():
            if name.endswith("E->E") and top > INPUT_RATIO_MAX:
                problems.append(f"{name}: вход клетки {top:.0f}× врождённого — связи разогнаны")

        if snap.dropped > 0:
            problems.append(f"потеряно спайков: {snap.dropped:.0f} (таблица событий переполнена)")

        return problems

    def report(self) -> str:
        """Последняя фраза по зонам + журнал нарушений."""

        if self.last is None:
            return "Показателей ещё нет: мозг не прожил ни одной фразы."

        snap = self.last
        lines = [
            f"Последняя фраза ({snap.words} слов): {snap.sentence[:80]}",
            "  зона           на слове: E активно / Гц, I Гц   в паузе: E активно / Гц",
        ]

        for name in self.names:
            ea, er, _, ir = snap.word[name]
            pa, pr, _, _ = snap.pause[name]
            lines.append(f"  {name:14s} {ea:7.2%} / {er:6.1f}   {ir:6.1f}       {pa:7.2%} / {pr:6.1f}")

        if snap.inputs:
            lines.append("  вход клетки к врождённому (макс / среднее):")
            for name, (top, mean) in snap.inputs.items():
                lines.append(f"    {name:28s} {top:6.1f}× / {mean:4.2f}×")

        bad = sum(1 for s in self.log)
        lines.append(
            f"Нарушений: {bad} фраз из {self.sentences}"
            + ("" if not bad else " — последние:")
        )

        for s in list(self.log)[-5:]:
            lines.append(f"  «{s.sentence[:60]}»")
            lines.extend(f"    {p}" for p in s.problems)

        return "\n".join(lines)
