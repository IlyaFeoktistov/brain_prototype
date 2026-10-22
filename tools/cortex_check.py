"""
Режим коры после подбора фона, не только средние частоты: по зонам и
слоям — частоты E и PV, доля клеток без спайков, средний потенциал E и
отношение тормозного тока к возбуждающему (построенные синапсы); затем
вызванный ответ: пустой экран, потом слово — LGN и V1 L4 по окнам 10 мс,
латентность (первое окно выше фона на 3 SD) и величина ответа.

    PYTHONPATH=. python tools/cortex_check.py [seeds=42,7,1] [column=20000] [ms=2000]
"""

from __future__ import annotations

import sys

import torch

from brain.config import BrainConfig
from brain.mind import Mind
from brain.session import Session


def run(seed: int, column: int, ms: float, text: str) -> None:
    cfg = BrainConfig()
    cfg.size = "small"
    cfg.min_column_neurons = column
    cfg.seed = seed
    cfg.autosave = False
    mind = Mind(cfg)
    b = mind.brain
    b.plasticity = False
    b.run(2500.0)

    steps = int(ms / 10.0)
    counts = torch.zeros(b.n, device=b.device)
    v = torch.zeros(b.n, device=b.device)
    exc = torch.zeros(b.n, device=b.device)
    inh = torch.zeros(b.n, device=b.device)

    for _ in range(steps):
        counts += b.observe(10.0) * 10.0 / 1000.0
        v += b.v[: b.n] / steps
        exc += b.i_syn[: b.n] / steps
        inh += b.i_inh[: b.n].abs() / steps

    print(f"\nзерно {seed}, колонка {column}: {b.n} нейронов")
    print(f"   {'зона':12s} {'слой':4s} {'E Гц':>6s} {'PV Гц':>6s} {'E молчат':>9s} {'PV молчат':>9s} {'V E, мВ':>8s} {'Iторм/Iвозб E':>13s}")

    for name, area in b.areas.items():
        if area.layers is None:
            continue

        for layer, group in area.layers.items():
            e, pv = group["E"], group["PV"]
            hz = lambda p: float(counts[p.slice].mean()) / (ms / 1000.0)
            silent = lambda p: float((counts[p.slice] == 0).float().mean())
            ratio = float(inh[e.slice].mean() / exc[e.slice].mean().clamp(min=1e-6))
            print(f"   {name:12s} {layer:4s} {hz(e):6.1f} {hz(pv):6.1f} {silent(e):9.0%} {silent(pv):9.0%} {float(v[e.slice].mean()):8.1f} {ratio:13.2f}")

    # Вызванный ответ: 500 мс пустого экрана, затем слово.
    s = Session(mind)
    s.line, s.gaze = "", 0.0
    lgn, l4 = b.areas["lgn"].E, b.areas["visual"].layers["L4"]["E"]
    trace = {"lgn": [], "v1": []}

    for phase, line, n in (("пусто", "", 50), ("слово", text + " ", 30)):
        s.line = line

        for _ in range(n):
            r = b.observe(10.0, s._view(10.0))
            trace["lgn"].append(float(r[lgn.slice].mean()))
            trace["v1"].append(float(r[l4.slice].mean()))

    for key, values in trace.items():
        base = torch.tensor(values[:50])
        mean, sd = float(base.mean()), float(base.std())
        after = values[50:]
        onset = next((i * 10 for i, x in enumerate(after) if x > mean + 3 * sd), None)
        peak = max(after)
        late = sum(after[10:]) / len(after[10:])
        print(f"   ответ {key:3s}: фон {mean:5.1f} ± {sd:4.1f} Гц, пик {peak:5.1f}, после 100 мс {late:5.1f} Гц, "
              f"латентность {'—' if onset is None else f'{onset}–{onset + 10} мс'}")


if __name__ == "__main__":
    opts = dict(kv.split("=", 1) for kv in sys.argv[1:])

    for seed in (int(x) for x in opts.get("seeds", "42,7,1").split(",")):
        run(seed, int(opts.get("column", 20000)), float(opts.get("ms", 2000)), opts.get("text", "мама мыла раму"))
