"""
Явные входы зоны-колонки из других зон по популяциям-получателям (CHAT_PLAN
шаг 3, цикл 14): путь, входов на клетку, Σw на клетку (мВ движка), средний
ток при частоте источника в покое (Σw × частота × τm, anatomy.RATES_HZ; у
колонок-источников — цель слоя) и поле недостающих входов из других зон.

    PYTHONPATH=. python tools/area_inputs.py [areas=association,prefrontal] [seed=42]
"""

from __future__ import annotations

import sys

import torch

from brain import anatomy
from brain.config import BrainConfig
from brain.mind import Mind


def main(opts: dict) -> None:
    cfg = BrainConfig()
    cfg.size = "small"
    cfg.seed = int(opts.get("seed", 42))
    cfg.autosave = False
    cfg.organs = ("words",)
    b = Mind(cfg).brain
    owner = {}
    for a in b.areas.values():
        owner[a.name] = (a.cells.start, a.cells.stop)

    def area_of(i):
        return next(n for n, (lo, hi) in owner.items() if lo <= i < hi)

    for name in opts.get("areas", "association,prefrontal").split(","):
        area = b.areas[name]
        pops = {p.name: p for _, p in area.populations}
        print(f"\n== {name}: входы из других зон (путь; входов на клетку; Σw; ток при покое, мВ)")
        rows = []
        for p in b.projections:
            if getattr(p, "source", None) != "net" or not torch.is_tensor(getattr(p, "pre", None)) or p.weight.dim() != 2:
                continue
            post = next((n for n, q in pops.items() if q.start <= p.post.start < q.stop), None)
            if post is None:
                continue
            src = area_of(int(p.pre.flatten()[0]))
            if src == name:
                continue
            hz = anatomy.RATES_HZ.get(src, 2.5)
            w = p.weight.float()
            sw = float(w.sum(1).mean())
            rows.append((post, p.name, w.shape[1], sw, sw * hz / 1000.0 * float(b.tau_m[p.post.start])))
        for post, path, k, sw, mv in sorted(rows):
            print(f"   → {post:24s} {path:34s} {k:5d}  {sw:8.2f}  {mv:7.2f}")
        for post, pre, coeff, rate, *_ in b.mean_field:
            if post.name in pops and pre.name.split(".")[0] != name:
                print(f"   → {post.name:24s} поле от {pre.name:24s}        {coeff * rate:7.2f}")


if __name__ == "__main__":
    main(dict(a.split("=", 1) for a in sys.argv[1:]))
