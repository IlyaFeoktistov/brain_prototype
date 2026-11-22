"""
Построенные связи внутри колонки (CHAT_PLAN шаг 3, цикл 13): по паре
популяций источник → получатель — входов на клетку, средний вес и Σw на
клетку (мВ движка, вес = ВПСП / пик); сравнение двух зон. Только пути,
у которых и источник, и получатель в зоне.

    PYTHONPATH=. python tools/column_wiring.py [areas=prefrontal,association] [seed=42]
"""

from __future__ import annotations

import sys

import torch

from brain.config import BrainConfig
from brain.mind import Mind


def wiring(b, name: str) -> dict:
    area = b.areas[name]
    pops = {p.name: p for _, p in area.populations}
    owner = lambda i: next((n for n, p in pops.items() if p.start <= i < p.stop), None)
    out = {}
    for p in b.projections:
        if getattr(p, "source", None) != "net" or not hasattr(p, "weight") or p.weight.dim() != 2:
            continue
        post = owner(p.post.start)
        if post is None:
            continue
        pre_idx = p.pre if torch.is_tensor(p.pre) else None
        if pre_idx is None or pre_idx.numel() == 0:
            continue
        first = int(pre_idx.flatten()[0])
        pre = owner(first)
        if pre is None:
            continue
        w = p.weight.float()
        key = (pre.split(".", 1)[1], post.split(".", 1)[1])
        k, mean, total = w.shape[1], float(w.mean()), float(w.sum(1).mean())
        a = out.setdefault(key, [0, 0.0, 0.0])
        a[0] += k
        a[1] = mean if a[1] == 0 else (a[1] + mean) / 2
        a[2] += total
    return out


def main(opts: dict) -> None:
    cfg = BrainConfig()
    cfg.size = "small"
    cfg.seed = int(opts.get("seed", 42))
    cfg.autosave = False
    cfg.organs = ("words",)
    b = Mind(cfg).brain
    names = opts.get("areas", "prefrontal,association").split(",")
    tables = {n: wiring(b, n) for n in names}
    keys = sorted(set().union(*tables.values()))
    print("связь".ljust(18) + "".join(f"{n[:12]:>36s}" for n in names))
    print(" " * 18 + "".join(f"{'входов   вес     Σw':>36s}" for _ in names))
    for key in keys:
        row = f"{key[0]:>7s}→{key[1]:<9s} "
        for n in names:
            v = tables[n].get(key)
            row += f"{'—':>36s}" if v is None else f"{v[0]:>18d} {v[1]:7.3f} {v[2]:8.2f}"
        print(row)


if __name__ == "__main__":
    main(dict(a.split("=", 1) for a in sys.argv[1:]))
