"""
Карта полей зрительного пути (DESIGN 2026-11-13 «Ступени по Dehaene»): у
каждого пути между ступенями — откуда клетка-получатель берёт входы по
месту на строке (field_cols источника, px; строка 56 px ≈ 8 букв): размах
мест входов, σ мест (и σ, взвешенная весами), разброс весов (CV) и где
центры клеток-получателей. Без прогона — только проводка свежего мозга.

    PYTHONPATH=. .venv/bin/python tools/rf_map.py [seed=1]
"""

import sys

import torch

from tools.reading_bench import build


def main(opts: dict) -> None:
    seed = opts.get("seed", "1")
    mind, b = build({"fit": "lgn-a", "homeo": f"lgn-d-pop-tcind-s{seed}", "relay_v_t_mv": "-59", "trn_scale": "0.1", "seed": seed})
    A, fc = b.areas, b.field_cols
    pops = {p.name: p for a in ("visual", "letters", "association") for g in A[a].layers.values() for p in g.values()}

    def locate(i: int):
        return next((p for p in pops.values() if p.start <= i < p.stop), None)

    names = ("visual: L4.E->E", "visual -> letters", "letters: L4.E->E", "letters -> association")
    for p in b.projections:
        if p.name not in names:
            continue
        pre = p.pre.long()
        src = locate(int(pre[0, 0]))
        cols = fc[src.name].cpu()[(pre - src.start).cpu()].float()
        w = p.weight.float().cpu()
        span = cols.max(1).values - cols.min(1).values
        wc = (w * cols).sum(1) / w.sum(1).clamp_min(1e-9)
        ws = ((w * (cols - wc[:, None]) ** 2).sum(1) / w.sum(1).clamp_min(1e-9)).sqrt()
        cv = (w.std(1) / w.mean(1).clamp_min(1e-9)).median()
        print(f"{p.name}: {src.name} → {p.post.name}, входов {pre.shape[1]}; размах мест входов {span.median():.1f} px; "
              f"σ мест {cols.std(1).median():.2f} px (по весам {ws.median():.2f}); вес CV {float(cv):.2f}; "
              f"центры получателей {float(wc.quantile(0.05)):.1f}…{float(wc.quantile(0.95)):.1f} px")


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
