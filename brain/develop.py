"""
Развитие мозга после сборки: гомеостаз возбудимости (cfg.homeostasis,
engine.homeostasis_update) на тишине. Мозг живёт окнами win мс, после
каждого окна регулируемые клетки сдвигают фон к частоте покоя своего типа;
в конце гомеостаз выключается — сдвиг заморожен в фоне клеток (bias,
сохраняется с мозгом).

Зачем при создании: сдвиги, посчитанные раньше и сохранённые по номеру
клетки, после любой правки анатомии (выросла зона, добавлено ядро) ложатся
на чужие клетки. Развитие считает их заново под ту анатомию, что собрана
сейчас — тот же механизм, что у tools/develop.py.

Тишина: с глазом — пустой экран (Session._blank, ямка видит ровную
страницу), без глаза — порт слов молчит.
"""

from __future__ import annotations

from typing import Callable

import torch

from brain.mind import Mind

# Прогрев до первого окна: переходный процесс после сборки (заряд мембран,
# STP, среднее поле) — в средние частоты гомеостаза не идёт. 2.5 с — как в
# опытах tools/develop.py, на которых посчитан пресет.
WARM_MS = 2500.0
# Окно гомеостаза: частота окна усредняется по homeostasis_avg_windows окнам.
WINDOW_MS = 1000.0
# Бюджет ускоренного гомеостаза на мозг, с модельного времени (cfg.develop_done_s): 30 с ≈
# сутки настоящей подстройки (server.SLOW_HOMEO); после — только медленный гомеостаз.
BUDGET_S = 30.0


def blank(mind: Mind) -> Callable[[float], torch.Tensor]:
    """Растр тишины на ms: пустой экран у мозга с глазом, молчащий порт слов без глаза."""

    if "eye" in mind.cfg.organs:
        from brain.session import Session

        return Session(mind)._blank
    from brain.body import Words

    port = Words.of(mind.brain)
    return lambda ms: port.hear(None, ms)


def develop(mind: Mind, seconds: float, win_ms: float = WINDOW_MS, warm_ms: float = WARM_MS,
            on_window: Callable[[int, int, torch.Tensor], None] | None = None,
            window: Callable[[float], torch.Tensor] | None = None,
            quiet: Callable[[float], torch.Tensor] | None = None) -> torch.Tensor:
    """
    Прожить warm_ms + seconds с гомеостазом; вернуть сдвиги фона [мВ, клетки].
    on_window(номер окна с 0, всего окон, частоты окна) — после каждого окна
    (прогресс, отчёт). window(ms) — частоты окна вместо b.observe на тишине
    (отчёт со снимками веток в tools/develop.py). quiet(ms) — растр тишины,
    если у вызывающего он уже есть (по умолчанию blank(mind)).
    """

    b = mind.brain
    quiet = quiet or blank(mind)
    plastic, homeo = b.plasticity, mind.cfg.homeostasis
    b.plasticity = False
    mind.cfg.homeostasis = True
    try:
        if warm_ms > 0:
            b.run(warm_ms, quiet(warm_ms))
        n = int(seconds * 1000.0 / win_ms)
        for w in range(n):
            rates = window(win_ms) if window else b.observe(win_ms, quiet(win_ms))
            b.homeostasis_update(rates)
            if on_window:
                on_window(w, n, rates)
    finally:
        b.plasticity = plastic
        mind.cfg.homeostasis = homeo
    return b.homeo_shift if getattr(b, "homeo_shift", None) is not None else torch.zeros(b.n, device=b.device)
