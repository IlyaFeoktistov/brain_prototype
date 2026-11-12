"""
Усиление петли «сеть V1 → фон → сеть» (стенд «фон L2/3», DESIGN 2026-11-12;
мозг не меняется). Пустой экран; pulses раз: пауза rest мс, затем всем E
клеткам V1 добавлено delta мВ к фону bias на pulse мс. Прибавка частоты E
(L2/3, L4) на импульсе против паузы перед ним. Сравнение с режимом без петли
(bg=independent): если петля умножает прибавку в k раз, усиление петли
G = 1 − 1/k (G ≥ 1 — неустойчиво). Импульс короче медленной опорной
частоты режима "network*" (2 с) — петля на этом окне ещё действует.

    PYTHONPATH=. .venv/bin/python tools/loop_gain.py [bg=independent] [delta=1] [pulse=500] [rest=1500] [pulses=10] [seed=1] [homeo=…]
"""

from __future__ import annotations

import sys

import torch

from brain.session import Session
from tools.reading_bench import build


def main(opts: dict) -> None:
    seed = int(opts.get("seed", 1))
    opts = {"fit": "lgn-a", "homeo": f"lgn-d-pop-s{seed}", "relay_v_t_mv": "-59", "trn_scale": "0.1", **opts}
    mind, b = build(opts)
    s = Session(mind, window_ms=10.0)
    vis = b.areas["visual"].layers
    pops = {"L2/3": vis["L23"]["E"], "L4": vis["L4"]["E"]}
    e_slices = [g["E"].slice for g in vis.values() if "E" in g]
    delta, pulse, rest = float(opts.get("delta", 1)), float(opts.get("pulse", 500)), float(opts.get("rest", 1500))
    s.line, s.gaze = "", 0.0
    b.run(2500.0, s._blank(2500.0))
    base = {k: [] for k in pops}
    kick = {k: [] for k in pops}
    for _ in range(int(opts.get("pulses", 10))):
        b.run(rest - 500.0, s._blank(rest - 500.0))
        r0 = b.observe(500.0, s._blank(500.0))
        for sl in e_slices:
            b.bias[sl] += delta
        r1 = b.observe(pulse, s._blank(pulse))
        for sl in e_slices:
            b.bias[sl] -= delta
        for k, p in pops.items():
            base[k].append(float(r0[p.slice].mean()))
            kick[k].append(float(r1[p.slice].mean()))
    print(f"режим {b.cfg.background_mode}, импульс +{delta} мВ на {pulse:.0f} мс всем E V1, {len(base['L4'])} раз:")
    for k in pops:
        b0, b1 = torch.tensor(base[k]), torch.tensor(kick[k])
        d = b1 - b0
        print(f"  V1 {k} E: пауза {float(b0.mean()):.2f} Гц, импульс {float(b1.mean()):.2f}, прибавка "
              f"{float(d.mean()):+.2f} ± {float(d.std() / len(d) ** 0.5):.2f} Гц")


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
