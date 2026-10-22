"""
Клетка реле LGN отдельно от сети (все входы, кроме своей сетчатки, выключены:
TRN, местные тормозные, кора, случайный фон; потенциал держит фон):

    1. порог: потенциал начала спайка при медленно растущем токе (dV/dt > 10
       мВ/мс) против заданного V_T;
    2. одиночный ретинальный ВПСП при −63 мВ — с триадой и без;
    3. кривая: начальный потенциал × интервал двух ВПСП → доля клеток со
       спайком (разброс клеток — их разброс возбудимости).

Сверка: макака — спайк реле обычно от суммы ≥ 2 ВПСП за 40 мс (Sincich et
al. 2007); около половины ретинальных спайков проходят (Carandini, Horton &
Sincich 2007).

    PYTHONPATH=. python tools/relay_threshold.py [relay_v_t=мВ]

relay_v_t — V_T клетки реле (гипотеза, BrainConfig.relay_v_t_mv).
"""

from __future__ import annotations

import torch

from brain.config import BrainConfig
from brain.mind import Mind


def main(relay_v_t: float | None = None) -> None:
    cfg = BrainConfig()
    cfg.relay_v_t_mv = relay_v_t
    cfg.size = "small"
    cfg.autosave = False
    b = Mind(cfg).brain
    b.plasticity = False
    lgn = b.areas["lgn"].E
    eye = [x.name for x in b.sensors].index("eye")
    off, size = b.sensor_offsets[eye], b.sensors[eye].size
    b.spontaneous[off: off + size] = 0.0
    triad = [p for p in b.projections if p.post is lgn and "триада" in p.name]

    for p in b.projections:
        if p.post is lgn and not p.name.startswith("eye"):
            p.weight.zero_()
    b.noise_gain[lgn.slice] = 0.0
    spread = b.bias[lgn.slice] - b.bias[lgn.slice].mean()
    b.CHUNK, b._graphs = 1, {}
    dt = cfg.dt_ms
    quiet = torch.zeros(1, b.n_receptors, device=b.device)
    cells = torch.arange(0, size, 4, device=b.device)
    kick = quiet.clone()
    kick[0, off + cells] = 1.0

    def hold(v: float, ms: float = 150.0) -> None:
        b.bias[lgn.slice] = (v - cfg.v_rest) + spread
        for _ in range(int(ms / dt)):
            b.run(dt, quiet)

    # 1. Порог: ток растёт на 0.02 мВ за шаг от −63.
    hold(-63.0)
    v_prev = b.v[lgn.slice].clone()
    onset = torch.full((lgn.size,), float("nan"), device=b.device)

    for k in range(int(400 / dt)):
        b.bias[lgn.slice] += 0.02
        b.run(dt, quiet)
        v = b.v[lgn.slice]
        steep = ((v - v_prev) / dt > 10.0) & onset.isnan()
        onset[steep] = v_prev[steep]
        v_prev = v.clone()

    ok = ~onset.isnan()
    print(f"1. порог: заданный V_T {float(b.v_t[lgn.start]):.1f} мВ (ΔT {float(b.delta_t[lgn.start]):.1f}); "
          f"начало спайка (dV/dt > 10 мВ/мс) {float(onset[ok].median()):.1f} мВ (медиана, {int(ok.sum())} клеток)")

    # 2. Одиночный ВПСП при −63.
    for label, on in (("с триадой", True), ("без триады", False)):
        saved = [p.weight.clone() for p in triad]
        if not on:
            for p in triad:
                p.weight.zero_()
        hold(-63.0)
        v0 = b.v[lgn.start + cells].clone()
        peak = torch.zeros(len(cells), device=b.device)
        for k in range(int(30 / dt)):
            b.run(dt, kick if k == 0 else quiet)
            peak = torch.maximum(peak, b.v[lgn.start + cells] - v0)
        print(f"2. ВПСП при −63 мВ {label}: {float(peak.mean()):.2f} мВ")
        for p, w in zip(triad, saved):
            p.weight.copy_(w)

    # 3. Кривая: начальный V × интервал → доля со спайком (с триадой).
    gaps = (5.0, 10.0, 20.0, 40.0)
    print("3. доля клеток со спайком (с триадой); строки — начальный V, столбцы — один ВПСП и пара через Δ мс")
    print("      V    один  " + "  ".join(f"{g:>4.0f}" for g in gaps))
    for v_start in (-67.0, -65.0, -63.0, -61.0, -59.0, -57.0, -55.0):
        row = []
        for gap in (None, *gaps):
            hold(v_start)
            spiked = torch.zeros(len(cells), dtype=torch.bool, device=b.device)
            span = int(((gap or 0) + 15) / dt)
            for k in range(span):
                second = gap is not None and k == int(gap / dt)
                b.run(dt, kick if k == 0 or second else quiet)
                spiked |= b.spikes[lgn.start + cells] > 0
            row.append(float(spiked.float().mean()))
        print(f"   {v_start:5.0f} " + "  ".join(f"{x:4.0%}" for x in row))


if __name__ == "__main__":
    import sys

    opts = dict(kv.split("=", 1) for kv in sys.argv[1:])
    main(float(opts["relay_v_t"]) if "relay_v_t" in opts else None)
