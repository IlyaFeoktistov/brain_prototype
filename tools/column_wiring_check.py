"""
Проверка проводки колонки при cfg.column_built_shares (ISSUES 25) до
подбора фона и гомеостаза: два построения мозга — флаг выключен / включён,
остальное то же. По местным путям колонки с тормозными (I → E, E → I,
I → I) каждой зоны — входов на клетку против ожидаемого по Potjans:
вероятность Potjans (cfg.column_probability) × клеток-источников зоны /
контактов на связь (cfg.column_contacts_*), множители связей 1. Плюс: какие
пути (имя, входов, средний вес) различаются между построениями — должны
только тормозные пути колонки (и, через общий генератор, их размещение).

    PYTHONPATH=. .venv/bin/python tools/column_wiring_check.py [areas=hand,visual] [seed=1]
"""

from __future__ import annotations

import gc
import sys

import torch

from tools.reading_bench import build


def snapshot(opts: dict) -> tuple[dict, dict, dict]:
    mind, b = build(opts)
    cfg = b.cfg
    stats = {}
    for p in b.projections:
        if hasattr(p, "pre") and torch.is_tensor(getattr(p, "pre", None)):
            stats[p.name] = (tuple(p.pre.shape), round(float(p.weight.float().mean()), 6))
        else:
            stats[p.name] = ((p.post.size, p.k_in), round(float(p.value), 6))
    sizes = {}
    for nm, ar in b.areas.items():
        if getattr(ar, "layers", None):
            sizes[nm] = {(l, k): P.size for l, g in ar.layers.items() for k, P in g.items()}
    conf = {"prob": cfg.column_probability, "order": [l for l, _, _ in cfg.column_cells],
            "ie": dict(cfg.column_contacts_ie), "ie_d": cfg.column_contacts_ie_default,
            "ei": cfg.column_contacts_ei, "ii": cfg.column_contacts_ii, "on": cfg.column_contacts_on}
    del mind, b
    gc.collect()
    torch.cuda.empty_cache()
    return stats, sizes, conf


def main(opts: dict) -> None:
    seed = int(opts.get("seed", 1))
    areas = opts.get("areas", "hand,visual").split(",")
    base = {"fit": "lgn-a", "homeo": f"lgn-d-pop-tcind-s{seed}", "relay_v_t_mv": "-59", "trn_scale": "0.1",
            "map_ee_sheet": "hand", "seed": str(seed)}
    old, sizes, c = snapshot({**base, "built_shares": "0"})
    new, sizes2, _ = snapshot({**base, "built_shares": "1"})
    assert sizes == sizes2, "размеры популяций различаются"
    order = c["order"]
    label = lambda l, k: ({"E": "E", "PV": "I"}.get(k, k)) if l == "L23" else f"{l}.{k}"  # noqa: E731
    for area in areas:
        print(f"\n== {area}: входов на клетку (было → стало; Potjans; стало / Potjans)")
        S = sizes[area]
        for (dl, dk) in S:
            for (sl, sk) in S:
                if sk == dk == "E":
                    continue
                name = f"{area}: {label(sl, sk)}->{label(dl, dk)}"
                if name not in old and name not in new:
                    continue
                row = 2 * order.index(dl) + (dk != "E")
                col = 2 * order.index(sl) + (sk != "E")
                p = c["prob"][row][col]
                if c["on"]:
                    if sk == "E":
                        p /= c["ei"]
                    elif dk == "E":
                        p /= c["ie"].get(sk, c["ie_d"])
                    else:
                        p /= c["ii"]
                expect = p * S[(sl, sk)]
                ko = old.get(name, ((0, 0), 0))[0][1]
                kn = new.get(name, ((0, 0), 0))[0][1]
                print(f"  {name:28s} {ko:5d} → {kn:5d}; Potjans {expect:7.1f}; ×{kn / expect if expect else float('nan'):.2f}"
                      f"  (вес {old.get(name, (0, 0))[1]:+.3f} → {new.get(name, (0, 0))[1]:+.3f})")
    diff = [n for n in sorted(set(old) | set(new)) if old.get(n) != new.get(n)]
    col_names = [n for n in diff if any(n.startswith(f"{a}: ") for a in sizes) and ("I" in n.split(": ")[1] or "PV" in n)]
    other = [n for n in diff if n not in col_names]
    print(f"\nпутей всего {len(old)}; изменились {len(diff)}: тормозные пути колонок {len(col_names)}, прочие {len(other)}")
    for n in other[:40]:
        print(f"  прочий: {n}: {old.get(n)} → {new.get(n)}")


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
