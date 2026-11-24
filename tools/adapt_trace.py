"""
Ход адаптации по окнам (CHAT_PLAN шаг 3, цикл 37): когда и после каких
сдвигов гомеостаза ПФК переходит во вспышки. Мозг — как в опыте B
(organs=words): разгон 2.5 с, затем окна по 1 с с гомеостазом; каждое окно
пишется по 5 мс.

По окну: частота и популяционный CV (окна 5 мс; у независимых клеток ≈
1/√(N·r·0.005)) — ПФК L2/3 E, L2/3 PV, L5 E, MD, TRN MD, EC II/III,
смысловая L2/3 E; наибольшая доля CA3 PC в окне 5 мс; средний сдвиг
гомеостаза (мВ) ПФК L2/3 E / PV / L5 E, MD, TRN MD, EC и доля клеток у предела.

freeze=N — гомеостаз только N окон, дальше без него (контроль: вспышки —
от сдвигов или от времени).

    PYTHONPATH=. python tools/adapt_trace.py [seed=42] [windows=30] [freeze=]
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
    b = Mind(cfg).brain
    b.plasticity = False
    s = develop.Silence(b)
    A = b.areas
    pfc = A["prefrontal"].layers
    pops = {
        "ПФК23E": pfc["L23"]["E"].slice, "ПФК23PV": pfc["L23"]["PV"].slice, "ПФК5E": pfc["L5"]["E"].slice,
        "MD": A["md"].E.slice, "TRNmd": A["trn_md"].E.slice, "EC": A["entorhinal"].E.slice,
        "смысл23E": A["association"].layers["L23"]["E"].slice,
    }
    shifts = ("ПФК23E", "ПФК23PV", "ПФК5E", "MD", "TRNmd", "EC")
    ca3 = A["ca3"].E.slice
    freeze = int(opts["freeze"]) if opts.get("freeze") else None
    lim = cfg.homeostasis_max_mv

    b.run(2500.0, s._blank(2500.0))
    print("окно | " + " | ".join(f"{k} Гц/CV" for k in pops) + " | CA3 max% | сдвиги мВ: " + " ".join(shifts), flush=True)

    for w in range(int(opts.get("windows", 30))):
        series = {k: torch.zeros(200, device=b.device) for k in pops}
        ca3_max = 0.0
        total = None
        for t in range(200):
            r = b.observe(5.0, s._blank(5.0))
            total = r.clone() if total is None else total + r
            for k, sl in pops.items():
                series[k][t] = r[sl].mean()
            ca3_max = max(ca3_max, float((r[ca3] > 0).float().mean()))
        rates = total / 200
        if freeze is None or w < freeze:
            b.homeostasis_update(rates)
        cells = []
        for k, y in series.items():
            m = float(y.mean())
            cells.append(f"{m:5.2f}/{float(y.std() / y.mean()) if m > 0 else float('nan'):4.2f}")
        sh = []
        for k in shifts:
            x = b.homeo_shift[pops[k]]
            edge = float((x.abs() >= lim - 1e-3).float().mean())
            sh.append(f"{float(x.mean()):+5.1f}" + (f"({edge:.0%})" if edge > 0.05 else ""))
        print(f"{w + 1:4d} | " + " | ".join(cells) + f" | {ca3_max:6.1%} | " + " ".join(sh), flush=True)


if __name__ == "__main__":
    main(dict(a.split("=", 1) for a in sys.argv[1:]))
