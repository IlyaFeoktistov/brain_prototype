"""
Анатомия залпов зоны-колонки (CHAT_PLAN шаг 3, цикл 11 — залпы ПФК): мозг как
в опыте B (organs=words, разгон 2.5 с, адаптация с гомеостазом), запись без
гомеостаза окнами 5 мс.

Залп — окно, где спайков главных клеток слоя-опоры (anchor, по умолчанию
L23.E) больше 5× их среднего. По залпам: доля клеток каждой популяции,
выстреливших в окне залпа ±10 мс; кто раньше — средний сдвиг пика каждой
популяции и входных зон относительно окна залпа (−100…+100 мс); средняя
частота каждой популяции вне залпов. Сравнение — та же доля в случайных
окнах (суррогат «без залпа»).

    PYTHONPATH=. python tools/area_bursts.py [area=prefrontal] [anchor=L23.E] [seed=42] [adapt=3] [record=6]
        [inputs=md,association.L5.E,association.L23.E,amygdala]
"""

from __future__ import annotations

import sys

import torch

from brain.config import BrainConfig
from brain.mind import Mind
from tools import develop


def main(opts: dict) -> None:
    cfg = BrainConfig()
    cfg.size = "small"
    cfg.seed = int(opts.get("seed", 42))
    cfg.autosave = False
    cfg.homeostasis = True
    cfg.event_overflow = "warn"
    cfg.organs = ("words",)
    develop.override_external(cfg, opts.get("extpop", ""))
    develop.copy_external(cfg, opts.get("extcopy", ""))
    b = Mind(cfg).brain
    b.plasticity = False
    s = develop.Silence(b)
    name = opts.get("area", "prefrontal")
    area = b.areas[name]
    pops = {f"{layer}.{kind}": p for layer, group in area.layers.items() for kind, p in group.items()}
    anchor = opts.get("anchor", "L23.E")
    inputs = {}
    for spec in opts.get("inputs", "md,association.L5.E,association.L23.E,amygdala").split(","):
        parts = spec.split(".")
        a = b.areas[parts[0]]
        inputs[spec] = a.E if len(parts) == 1 else a.layers[parts[1]][parts[2]]

    b.run(2500.0, s._blank(2500.0))
    for _ in range(int(opts.get("adapt", 3))):
        b.homeostasis_update(b.observe(1000.0, s._blank(1000.0)))
    b.cfg.homeostasis = False

    bins = int(float(opts.get("record", 6)) * 1000 / 5)
    series = {k: torch.zeros(bins, device=b.device) for k in [*pops, *inputs]}
    # Кто выстрелил в каком окне — по клеткам (для доли участников залпа).
    fired = {k: torch.zeros(bins, p.size, dtype=torch.bool, device=b.device) for k, p in pops.items()}
    for t in range(bins):
        r = b.observe(5.0, s._blank(5.0))
        for k, p in [*pops.items(), *inputs.items()]:
            x = r[p.slice] * 0.005
            series[k][t] = x.sum()
            if k in fired:
                fired[k][t] = x > 0

    a = series[anchor]
    mean = float(a.mean())
    events = (a > 5 * mean).nonzero().flatten().tolist() if mean > 0 else []
    # Начало залпа — первое окно серии подряд.
    starts = [t for t in events if t - 1 not in events and 20 <= t < bins - 20]
    print(f"зерно {cfg.seed}, {name}: опора {anchor} {mean / pops[anchor].size / 0.005:.2f} Гц; "
          f"залпов {len(starts)} за {bins * 0.005:.0f} с", flush=True)
    if not starts:
        return

    gen = torch.Generator().manual_seed(1)
    random = torch.randint(20, bins - 20, (200,), generator=gen).tolist()

    def share(k: str, ts: list[int]) -> float:
        f = fired[k]
        return float(torch.stack([f[t - 2: t + 3].any(0).float().mean() for t in ts]).mean())

    def peak(k: str) -> tuple[float, float]:
        y = series[k]
        lags = []
        for t in starts:
            w = y[t - 20: t + 21]
            lags.append((int(w.argmax()) - 20) * 5)
        base = float(y.mean())
        rise = float(torch.stack([y[t - 2: t + 3].mean() for t in starts]).mean()) / base if base > 0 else float("nan")
        return sum(lags) / len(lags), rise

    print("внешний фон по популяциям (среднее / разброс, мВ):",
          "  ".join(f"{k} {float(b.noise_mean[p.slice].mean()):.1f}/{float(b.noise_std[p.slice].mean()):.1f}" for k, p in pops.items()))
    print("популяция        частота   в залпе ±10 мс / в случайном окне   пик относительно начала, мс   × средней")
    for k, p in pops.items():
        hz = float(series[k].mean()) / p.size / 0.005
        lag, rise = peak(k)
        print(f"   {k:12s} {hz:7.2f} Гц   {share(k, starts):6.1%} / {share(k, random):6.1%}"
              f"                 {lag:+6.1f}                 {rise:6.1f}")
    print("входы:")
    for k, p in inputs.items():
        hz = float(series[k].mean()) / p.size / 0.005
        lag, rise = peak(k)
        print(f"   {k:22s} {hz:7.2f} Гц   пик {lag:+6.1f} мс   × средней {rise:5.2f}")


if __name__ == "__main__":
    main(dict(a.split("=", 1) for a in sys.argv[1:]))
