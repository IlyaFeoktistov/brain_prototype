"""
Депрессия синапса таламус → кора (cfg.nucleus_stp, путь lgn → visual) —
замер одного синапса, мозг не меняется. Счёт ресурса — тот же, что в
движке (engine._step, шаг dt): в конце шага x += (1 − x)·dt/τ, затем
x −= U·x·спайк; доставка спайка — на следующем шаге с весом
thalamic_psp_mv × x (brain/csrc/events.cu, `w *= stp_x[pre]`), то есть
ресурс уже после выброса. Рядом — сила по определению модели (ресурс до
спайка, как у синапсов колонок: x / (1 − u), и как в блоке Chariker).

Печатает (доля от ВПСП отдохнувшего синапса thalamic_psp_mv):
  1. отдохнувший синапс: первый спайк после долгой паузы;
  2. серии 8 / 20 / 50 / 100 Гц из покоя и с фона 8 Гц (пуассоновский),
     спайки 1–5 и устоявшийся — относительно первого серии;
  3. восстановление: после серии 20 Гц × 10 тестовый спайк через
     50 / 100 / 250 / 500 / 750 / 1000 мс;
  4. передача (частота × сила) на пуассоновском фоне 8 / 20 / 50 / 100 Гц.
LGN → PV — тот же ресурс клетки-источника (×thalamic_pv_ratio к весу),
депрессия у E и PV в модели одинакова.

    PYTHONPATH=. .venv/bin/python tools/tc_depression.py [trials=2000] [seed=1]
"""

from __future__ import annotations

import math
import sys

import torch

from brain.config import BrainConfig


def params(cfg: BrainConfig) -> tuple[float, float]:
    for src, dst, U, tau in cfg.nucleus_stp:
        if (src, dst) == ("lgn", "visual"):
            return U, tau
    raise SystemExit("нет lgn → visual в cfg.nucleus_stp")


def run(spikes: torch.Tensor, U: float, tau: float, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    """spikes [испытаний, шагов] (0/1). Возвращает силу каждого шага по
    движку (ресурс после выброса) и по определению (до выброса)."""
    n, steps = spikes.shape
    x = torch.ones(n, dtype=torch.float64)
    after = torch.zeros(n, steps, dtype=torch.float64)
    before = torch.zeros(n, steps, dtype=torch.float64)
    rec = dt / tau
    for t in range(steps):
        x = x + (1.0 - x) * rec
        before[:, t] = x
        x = x - U * x * spikes[:, t]
        after[:, t] = x
    return after, before


def train(n: int, steps: int, dt: float, bg_hz: float, start: int, hz: float, count: int,
          gen: torch.Generator) -> torch.Tensor:
    s = (torch.rand(n, steps, generator=gen, dtype=torch.float64) < bg_hz * dt / 1000.0).double()
    if hz > 0:
        s[:, start:] = 0.0
        step = round(1000.0 / hz / dt)
        for k in range(count):
            s[:, start + k * step] = 1.0
    return s


def main(opts: dict) -> None:
    cfg = BrainConfig()
    U, tau = params(cfg)
    dt = cfg.dt_ms
    n = int(opts.get("trials", 2000))
    gen = torch.Generator().manual_seed(int(opts.get("seed", 1)))
    psp = cfg.thalamic_psp_mv
    print(f"lgn → visual: U {U}, τ восстановления {tau:.0f} мс, шаг {dt} мс, ВПСП отдохнувшего {psp} мВ "
          f"(PV ×{cfg.thalamic_pv_ratio})")

    # 1. Отдохнувший синапс.
    s = torch.zeros(1, 10)
    s[0, 5] = 1.0
    after, before = run(s, U, tau, dt)
    print("\n1. первый спайк после паузы: по движку "
          f"{float(after[0, 5]):.2f} ({float(after[0, 5]) * psp:.2f} мВ), по определению {float(before[0, 5]):.2f}")

    # 2. Серии.
    print("\n2. серия из 10 спайков, сила спайков 2–5 и 10-го относительно первого (одинаково по движку и по "
          "определению);\n   абсолютная сила первого — доля ВПСП отдохнувшего (движок / определение)")
    for bg in (0.0, 8.0):
        warm = round(3000.0 / dt)
        for hz in (8.0, 20.0, 50.0, 100.0):
            step = round(1000.0 / hz / dt)
            steps = warm + 10 * step + 1
            sp = train(n if bg else 1, steps, dt, bg, warm, hz, 10, gen)
            after, before = run(sp, U, tau, dt)
            idx = [warm + k * step for k in range(10)]
            b = before[:, idx].mean(0)
            a = after[:, idx].mean(0)
            rel = (b / b[0]).tolist()
            print(f"   фон {bg:.0f} Гц, {hz:>3.0f} Гц: 1-й {float(a[0]):.2f} / {float(b[0]):.2f}; "
                  f"2–5: {' '.join(f'{r:.2f}' for r in rel[1:5])}; 10-й {rel[9]:.2f}")

    # 3. Восстановление после серии 20 Гц × 10 (без фона и с фоном 8 Гц).
    print("\n3. восстановление после серии 20 Гц × 10: тестовый спайк / первый серии")
    for bg in (0.0, 8.0):
        row = []
        for gap in (50.0, 100.0, 250.0, 500.0, 750.0, 1000.0):
            warm = round(3000.0 / dt)
            step = round(50.0 / dt)
            last = warm + 9 * step
            test = last + round(gap / dt)
            sp = train(n if bg else 1, test + 2, dt, bg, warm, 20.0, 10, gen)
            sp[:, last + 1:test] = (torch.rand(sp.shape[0], test - last - 1, generator=gen, dtype=torch.float64)
                                    < bg * dt / 1000.0).double()
            sp[:, test] = 1.0
            _, before = run(sp, U, tau, dt)
            row.append(float(before[:, test].mean() / before[:, warm].mean()))
        print(f"   фон {bg:.0f} Гц: " + "  ".join(f"{g:.0f} мс {r:.2f}" for g, r in
                                               zip((50, 100, 250, 500, 750, 1000), row)))

    # 4. Передача на пуассоновском фоне.
    print("\n4. пуассоновский вход: средняя сила спайка и передача (частота × сила, в ВПСП отдохнувшего за с)")
    base = None
    for hz in (1.0, 8.0, 20.0, 50.0, 100.0):
        steps = round(20000.0 / dt)
        sp = (torch.rand(200, steps, generator=gen, dtype=torch.float64) < hz * dt / 1000.0).double()
        after, before = run(sp, U, tau, dt)
        cut = round(2000.0 / dt)
        m = sp[:, cut:] > 0
        a, b = float(after[:, cut:][m].mean()), float(before[:, cut:][m].mean())
        if hz == 8.0:
            base = hz * a
        theory = 1.0 / (1.0 + U * hz * tau / 1000.0)
        print(f"   {hz:>3.0f} Гц: сила по движку {a:.3f}, по определению {b:.3f} (1/(1+Urτ) {theory:.3f}); "
              f"передача {hz * a:.2f} / {hz * b:.2f}" + (f", к 8 Гц ×{hz * a / base:.2f}" if base else ""))
    print(f"\n   поток при бесконечной частоте → 1/(U·τ) = {1000.0 / (U * tau):.2f} ВПСП/с (по определению), "
          f"по движку × (1 − U) = {1000.0 * (1 - U) / (U * tau):.2f}")


if __name__ == "__main__":
    main(dict(a.split("=", 1) for a in sys.argv[1:]))
