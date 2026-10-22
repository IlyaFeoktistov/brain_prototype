"""
Что гонит популяцию: частота популяции в покое, когда каждый её вход по
очереди выключен (веса пути → 0; средний ток недостающих соседей того же
пути остаётся — он видна отдельной строкой у карточки клетки).

    PYTHONPATH=. python tools/input_trace.py pop=motor.L4.E,hand.L4.E [ms=1000] [warm=2500]

Пути, выключение которых снижает частоту сильнее всех, — главные источники;
если частота падает от выключения пути из своей же колонки — раскачивает
рекуррентная петля, а не внешний вход.
"""

from __future__ import annotations

import sys

import torch

from brain.config import BrainConfig
from brain.mind import Mind


def main(pops: list[str], ms: float, warm: float) -> None:
    cfg = BrainConfig()
    cfg.size = "small"
    cfg.autosave = False
    b = Mind(cfg).brain
    b.plasticity = False
    targets = {}

    for name in pops:
        area, layer, kind = name.split(".")
        targets[name] = b.areas[area].layers[layer][kind]

    b.run(warm)

    def rates() -> dict[str, float]:
        with b.frozen():
            r = b.observe(ms)
        return {name: float(r[pop.slice].mean()) for name, pop in targets.items()}

    base = rates()
    print("покой: " + ", ".join(f"{n} {v:.1f} Гц" for n, v in base.items()))
    # Входы целевых популяций (кроме процедурных путей — их веса не хранятся).
    paths: dict[str, list] = {}

    for p in b.projections:
        if any(p.post is pop for pop in targets.values()) and p.weight.numel():
            paths.setdefault(p.name, []).append(p)

    rows = []

    for name, ps in paths.items():
        saved = [p.weight.clone() for p in ps]

        for p in ps:
            p.weight.zero_()

        r = rates()

        for p, w in zip(ps, saved):
            p.weight.copy_(w)

        synapses = sum(p.weight.numel() // p.post.size for p in ps)
        rows.append((name, synapses, r))

    for pop in targets:
        print(f"\n{pop}: без пути (входов на клетку) → частота, Гц (покой {base[pop]:.1f})")

        for name, k, r in sorted(rows, key=lambda x: x[2][pop]):
            if any(p.post is targets[pop] for p in paths[name]):
                print(f"   {name:40s} {k:5d}  → {r[pop]:6.1f}")


if __name__ == "__main__":
    opts = dict(kv.split("=", 1) for kv in sys.argv[1:])
    main(opts.get("pop", "motor.L4.E,hand.L4.E").split(","), float(opts.get("ms", 1000)), float(opts.get("warm", 2500)))
