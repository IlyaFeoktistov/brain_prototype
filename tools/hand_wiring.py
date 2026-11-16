"""
Пространственный профиль связей L2/3 руки из проводки (без динамики):
P(связь | расстояние по карте) для E → E, PV → E, E → PV (DESIGN 2026-11-16
«Баланс E/I в участке L2/3 руки»). Карта руки — линия 0…1 без физического
масштаба (engine._place_maps), местные входы — «k ближайших» (_nearest_pre).
Для перевода в мкм — эквивалент на листе: площадь зоны руки = клеток зоны /
клеток колонки 1 мм² (Potjans, как engine._sheet_place) и радиус круга, где
у клетки на листе лежат столько же клеток источника, сколько в полосе карты.

    PYTHONPATH=. .venv/bin/python tools/hand_wiring.py [seed=1]
"""

from __future__ import annotations

import math
import sys

import torch

from tools.reading_bench import build


def main(opts: dict) -> None:
    seed = int(opts.get("seed", 1))
    mind, b = build({"fit": "lgn-a", "homeo": f"lgn-d-pop-tcind-s{seed}", "relay_v_t_mv": "-59", "trn_scale": "0.1", **opts})
    A = b.areas["hand"]
    E, PV = A.layers["L23"]["E"], A.layers["L23"]["PV"]
    cells = sum(p.size for _, p in A.populations)
    per_mm2 = sum(e + i for _, e, i in b._column_layers("hand"))
    area_mm2 = cells / per_mm2
    print(f"зона руки: клеток {cells}, колонка 1 мм² — {per_mm2}; площадь на листе {area_mm2:.3f} мм² (сторона {math.sqrt(area_mm2):.2f} мм); "
          f"L2/3 E {E.size}, PV {PV.size}", flush=True)
    dens = {P.name: P.size / area_mm2 for P in (E, PV)}
    for p in b.projections:
        if p.source != "net" or not p.name.startswith("hand:") or p.post not in (E, PV):
            continue
        pre_pop = next((P for P in (E, PV) if P.start <= int(p.pre.min()) < P.start + P.size), None)
        if pre_pop is None:
            continue
        pre = p.pre.long().cpu() - pre_pop.start
        cpost = b.field_cols[p.post.name].float().cpu()
        cpre = b.field_cols[pre_pop.name].float().cpu()
        d = (cpre[pre] - cpost[:, None]).abs()
        d = torch.minimum(d, 1 - d)
        k = p.pre.shape[1]
        edges = torch.tensor([0, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.5])
        cnt = torch.histc(d.flatten(), bins=1, min=0, max=1)  # noqa: F841
        row = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            got = float(((d >= lo) & (d < hi)).sum()) / p.post.size
            avail = 2 * float(hi - lo) * pre_pop.size
            row.append(got / avail)
        # эквивалент на листе: клеток источника в полосе ±x по линии = 2x·N; круг с тем же числом на листе
        r_um = lambda x: 1000 * math.sqrt(2 * x * pre_pop.size / (math.pi * dens[pre_pop.name]))  # noqa: E731
        q50 = float(d.median())
        print(f"\n{p.name}: {pre_pop.name} → {p.post.name}, входов {k}, PSP пары {float(p.weight.float().mean()) * b._psp_peak(p.post, inhibitory=float(p.weight.mean()) < 0):+.3f} мВ; "
              f"медиана |Δ| {q50:.4f} ≈ радиус {r_um(q50):.0f} мкм на листе", flush=True)
        print("   |Δ| карты:   " + " ".join(f"{float(lo):.3f}–{float(hi):.3f}" for lo, hi in zip(edges[:-1], edges[1:])))
        print("   ≈ мкм листа: " + " ".join(f"{r_um(float(lo)):5.0f}–{r_um(float(hi)):<5.0f}" for lo, hi in zip(edges[:-1], edges[1:])))
        print("   P(связь):    " + " ".join(f"{x:11.3f}" for x in row), flush=True)


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
