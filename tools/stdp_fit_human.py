"""
Окно STDP человека по измерениям: гиппокамп (anatomy.HUMAN_HIPPOCAMPUS_STDP,
Testa-Silva et al. 2010, https://doi.org/10.3389/fnsyn.2010.00012 — точки
по клеткам) и кора (anatomy.HUMAN_CORTEX_STDP, Verhoog et al. 2013,
https://doi.org/10.1523/JNEUROSCI.3158-13.2013 — средние ± SEM).

Форма окна — та, что можно собрать из следов спайков (все пары со всеми):
каждая сторона — одна экспонента или разность двух,
    Δt > 0 (пре раньше поста):  a1·exp(−Δt/τ1) − a2·exp(−Δt/τ2),
    Δt < 0 (пост раньше пре):   b1·exp(Δt/σ1) − b2·exp(Δt/σ2).
Протокол опыта воспроизводится: N пар, в паре 1 или 2 спайка поста через 10
мс (пары через 7 с не перекрываются), изменение — N · A+ · сумма окна.
Амплитуды — наименьшие квадраты (вес — 1/SEM², у точек по клеткам — 1);
постоянные времени — перебор в [2, 300] мс. Печатает лучшую подгонку,
диапазон постоянных, при которых ошибка не хуже лучшей больше чем на 10%
(что данные определяют, а что нет), предупреждение у края перебора, окно
между точками и за пределами проверенного, интеграл окна.

Это подгонка формы к данным, не параметры симуляции: в движок они идут
отдельно и только после сверки.

    PYTHONPATH=. .venv/bin/python tools/stdp_fit_human.py [data=hippocampus|cortex]
"""

import math
import sys

import torch

from brain import anatomy
from brain.config import BrainConfig

opts = {"data": "hippocampus"}

for kv in sys.argv[1:]:
    k, v = kv.split("=", 1)
    opts[k] = v

torch.set_default_dtype(torch.float64)
A = BrainConfig().stdp_a_plus
TAUS = [2.0, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100, 130, 160, 200, 250, 300]

if opts["data"] == "hippocampus":
    N = anatomy.HUMAN_HIPPOCAMPUS_STDP_PAIRINGS
    points = [(dt, k, y, 1.0) for dt, k, y in anatomy.HUMAN_HIPPOCAMPUS_STDP]
    tested = (-130.0, 10.0)
else:
    N = anatomy.HUMAN_CORTEX_STDP_PAIRINGS
    points = [(dt, 1, y, sem) for dt, y, sem in anatomy.HUMAN_CORTEX_STDP]
    tested = (-130.0, 60.0)


def times(dt, k):
    return [dt + 10.0 * i for i in range(k)]


def side(sign):
    """Точки одной стороны: спайки поста с этой стороны от пре."""

    rows = []

    for dt, k, y, sem in points:
        ts = [t for t in times(dt, k) if (t > 0) == (sign > 0) and t != 0]

        if ts and len(ts) == k:
            rows.append((ts, y, sem))

    return rows


def fit(rows, taus):
    """Амплитуды при заданных τ (у второй — со знаком «−»), взвешенная ошибка."""

    M = torch.tensor([[N * A * sum(math.exp(-abs(t) / tau) for t in ts) * (1 if j == 0 else -1)
                       for j, tau in enumerate(taus)] for ts, _, _ in rows])
    y = torch.tensor([y for _, y, _ in rows])
    w = torch.tensor([1.0 / sem for *_, sem in rows])
    amp = torch.linalg.lstsq(M * w[:, None], (y * w)[:, None]).solution[:, 0]
    err = float((((M @ amp - y) * w) ** 2).sum())
    return amp.tolist(), err


def search(rows, two):
    found = []

    for t1 in TAUS:
        for t2 in ([t for t in TAUS if t > t1] if two else [None]):
            taus = [t1] + ([t2] if two else [])

            if len(rows) < len(taus) + 1 and two:
                continue
            amp, err = fit(rows, taus)

            if two and (amp[0] <= 0 or amp[1] <= 0):
                continue
            found.append((err, taus, amp))

    found.sort(key=lambda f: f[0])
    return found


def window(fits, t):
    taus, amp = fits[t > 0]
    return sum(a * (1 if j == 0 else -1) * math.exp(-abs(t) / tau) for j, (a, tau) in enumerate(zip(amp, taus)))


print(f"{opts['data']}: {len(points)} точек, {N} пар, A+ {A}; проверено {tested[0]:+.0f}…{tested[1]:+.0f} мс")
chosen = {}

for name, sign in (("пост раньше пре (Δt < 0)", -1), ("пре раньше поста (Δt > 0)", +1)):
    rows = side(sign)
    print(f"\n{name}: {len(rows)} точек")

    if not rows:
        continue

    for two in (False, True):
        found = search(rows, two)

        if not found:
            print(f"  {'две' if two else 'одна'} экспоненты: не подходит (мало точек или амплитуда < 0)")
            continue

        err, taus, amp = found[0]
        ok = [f for f in found if f[0] <= err * 1.1 + 1e-9]
        spans = [(min(f[1][j] for f in ok), max(f[1][j] for f in ok)) for j in range(len(taus))]
        edge = any(t in (TAUS[0], TAUS[-1]) for t in taus)
        parts = ", ".join(f"{'+' if j == 0 else '−'}{a:.2f}·τ {t:g} мс (от {lo:g} до {hi:g})"
                          for j, (a, t, (lo, hi)) in enumerate(zip(amp, taus, spans)))
        print(f"  {'две' if two else 'одна'} экспоненты: ошибка {err:.2f}; {parts}"
              + ("  ⚠ τ у края перебора — данные её не определяют" if edge else ""))
        # Берётся форма с меньшей ошибкой (лишняя экспонента ошибку не
        # увеличивает — сравнивать формы стоит по диапазонам выше).
        if (sign > 0) not in chosen or err < chosen[sign > 0][0]:
            chosen[sign > 0] = (err, taus, amp, two)

fits = {s: (c[1], c[2]) for s, c in chosen.items()}

if len(fits) == 2:
    print("\nвзята форма с меньшей ошибкой:")

    for s, c in chosen.items():
        print(f"  {'Δt > 0' if s else 'Δt < 0'}: {'две' if c[3] else 'одна'} экспоненты")
    print("\nточки — опыт / модель (протокол опыта, сумма пар); † — спайки поста по обе стороны от пре, в подгонку не входили:")

    for dt, k, y, sem in points:
        ts = times(dt, k)
        model = N * A * sum(window(fits, t) for t in ts)
        mixed = " †" if len({t > 0 for t in ts}) > 1 else ""
        print(f"  {dt:+7.1f} мс ×{k}: {y:+.2f}{f' ± {sem:.2f}' if sem != 1.0 else ''} / {model:+.2f}{mixed}")

    print("\nокно (протокол из одного спайка поста), * — вне проверенного:")

    for dt in (-250, -200, -150, -130, -100, -80, -60, -40, -20, -10, -5, -2, 2, 5, 8, 10, 15, 20, 40, 60, 100):
        mark = "*" if not tested[0] <= dt <= tested[1] else " "
        print(f"  {mark}{dt:+5d} мс: {N * A * window(fits, float(dt)):+.3f}")

    integral = sum(a * (1 if j == 0 else -1) * tau for s in fits for j, (a, tau) in enumerate(zip(fits[s][1], fits[s][0])))
    print(f"\nинтеграл окна (мс, в долях A+): {integral:+.1f} — > 0: независимые спайки растят вес, < 0 — ослабляют;"
          " зависит от хвостов вне проверенного диапазона")
