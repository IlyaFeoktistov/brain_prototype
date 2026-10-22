"""
Из чего складывается потенциал клетки реле LGN: средний V клеток реле со
всеми входами и без каждого по очереди (разница — вклад источника, мВ;
клетки реле почти всё время ниже порога, так что разность V — честная
мера). Источники: сетчатка (водитель), триада, местные тормозные, TRN,
кора (V1 L6), случайный фон ядра, постоянный фон (nucleus_bias_mv).

Сверка: клетка реле наяву ~−63 мВ (мышь, тихое бодрствование — Molnár et
al. 2021; кошка −63 ± 7); торможение от TRN при стимуляции 50 Гц ~4 мВ
(мышь, срез — Campbell et al. 2020).

    PYTHONPATH=. python tools/relay_balance.py [lgn_bias=мВ] [ms=2000] [text=…]
"""

from __future__ import annotations

import sys

import torch

from brain.config import BrainConfig
from brain.mind import Mind
from brain.session import Session


def main(lgn_bias: float | None, ms: float, text: str) -> None:
    cfg = BrainConfig()
    cfg.size = "small"
    cfg.autosave = False
    mind = Mind(cfg)
    b = mind.brain
    b.plasticity = False
    lgn = b.areas["lgn"].E
    s = Session(mind)
    s.line, s.gaze = text + " " if text else "", 0.0

    if lgn_bias is not None:
        b.bias[lgn.slice] = lgn_bias

    for _ in range(int(2500 / s.window_ms)):
        b.observe(s.window_ms, s._view(s.window_ms))

    def measure() -> tuple[float, float]:
        with b.frozen():
            v, hz, n = 0.0, 0.0, int(ms / 5.0)
            for _ in range(n):
                r = b.observe(5.0, s._view(5.0))
                v += float(b.v[lgn.slice].mean()) / n
                hz += float(r[lgn.slice].mean()) / n
        return v, hz

    base_v, base_hz = measure()
    print(f"фон LGN {float(b.bias[lgn.start]):+.1f} мВ, экран: «{text}» — V клеток реле {base_v:.1f} мВ, {base_hz:.1f} Гц"
          f"   [наяву ~−63 мВ; порог {cfg.v_threshold:.0f}, покой {cfg.v_rest:.0f}]")
    groups = {
        "сетчатка (водитель)": lambda p: p.post is lgn and p.name.startswith("eye ->") and "триада" not in p.name,
        "триада": lambda p: p.post is lgn and "триада" in p.name,
        "местные тормозные": lambda p: p.post is lgn and p.name == "lgn: I->E",
        "TRN": lambda p: p.post is lgn and p.name == "trn -> lgn",
        "кора (V1 L6)": lambda p: p.post is lgn and p.name == "visual -> lgn",
    }

    for label, pick in groups.items():
        ps = [p for p in b.projections if pick(p)]
        saved = [p.weight.clone() for p in ps]
        for p in ps:
            p.weight.zero_()
        v, hz = measure()
        for p, w in zip(ps, saved):
            p.weight.copy_(w)
        print(f"   {label:22s} вклад {base_v - v:+6.1f} мВ (без него V {v:6.1f}, {hz:5.1f} Гц)")

    for label, name in (("случайный фон ядра", "noise_gain"), ("постоянный фон", "bias")):
        t = getattr(b, name)
        saved = t[lgn.slice].clone()
        t[lgn.slice] = 0.0
        v, hz = measure()
        t[lgn.slice] = saved
        print(f"   {label:22s} вклад {base_v - v:+6.1f} мВ (без него V {v:6.1f}, {hz:5.1f} Гц)")

    print(f"   частоты источников: сетчатка {s.cfg.retina_rate_hz:.0f} Гц на штрих / 20 спонтанно, "
          f"TRN {float(b.observe(200.0)[b.areas['trn'].E.slice].mean()):.1f} Гц, местные I {float(b.observe(200.0)[b.areas['lgn'].I.slice].mean()):.1f} Гц")


if __name__ == "__main__":
    opts = dict(kv.split("=", 1) for kv in sys.argv[1:])
    main(float(opts["lgn_bias"]) if "lgn_bias" in opts else None, float(opts.get("ms", 2000)), opts.get("text", "мама мыла раму"))
