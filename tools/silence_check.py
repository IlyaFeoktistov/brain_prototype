"""
Почему клетки V1 почти не молчат на пустом экране (замер, мозг не меняется;
мозг как в letters_dc, гомеостаз lgn-d-pop заморожен). У бодрствующей макаки
на пустом сером экране молчат 35% клеток L2 и 72% L3 (Gur & Snodderly 2008,
https://doi.org/10.1113/jphysiol.2008.151795), у мозга — 1–2% за 20 с.

По каждой клетке популяции (pops) за seconds с пустого экрана, шаг мозга
(CHUNK = 1): средний потенциал мембраны вне рефрактерности, его размах (СКО),
расстояние до порога экспоненты v_t в мВ и в СКО (z), частота. Клетки — по
z: z < 1 — постоянный ток держит у порога; z ≥ 3 — порог переходят только
редкие выбросы. Плюс постоянные для всей популяции части входа, мВ
(установившийся вклад): внешний фон колонки (шум движка, среднее и СКО
одного шага), постоянный ток недостающих соседей (dc_mv), bias со сдвигом
гомеостаза — и их разброс по клеткам (у одинаковых для всех клеток частей
разброса нет — молчащих они не создают).

    PYTHONPATH=. .venv/bin/python tools/silence_check.py [pops=visual.L4E,visual.L23E] [seconds=3] [seed=1]
"""

from __future__ import annotations

import sys
import time

import torch

from brain.engine import SpikingBrain
from brain.session import Session

SpikingBrain.CHUNK = 1

from tools.reading_bench import build  # noqa: E402


def main(opts: dict) -> None:
    seed = int(opts.get("seed", 1))
    opts = {"fit": "lgn-a", "homeo": f"lgn-d-pop-s{seed}", "relay_v_t_mv": "-59", "trn_scale": "0.1", **opts}
    mind, b = build(opts)
    s = Session(mind, window_ms=10.0)
    cfg = b.cfg
    dt = cfg.dt_ms
    pops = {}
    for name in opts.get("pops", "visual.L4E,visual.L23E").split(","):
        area, rest = name.split(".")
        pops[name] = b.areas[area].layers[rest[:-1] if rest.endswith("E") else rest[:-2]][rest[-1] if rest.endswith("E") else rest[-2:]]
    s.line, s.gaze = "", 0.0
    b.run(2500.0, s._blank(2500.0))
    steps = int(float(opts.get("seconds", 3)) * 1000 / dt)
    acc = {k: [torch.zeros(p.size, device=b.device) for _ in range(3)] for k, p in pops.items()}
    spikes = {k: torch.zeros(p.size, device=b.device) for k, p in pops.items()}
    view = s._blank(steps * dt)
    t0 = time.time()
    for t in range(steps):
        b.run(dt, view[t:t + 1])
        for k, p in pops.items():
            v = b.v[p.slice]
            free = (b.refractory[p.slice] <= 0).float()
            a = acc[k]
            a[0] += free
            a[1] += v * free
            a[2] += v * v * free
            spikes[k] += b.spikes[p.slice]
    print(f"пустой экран {steps * dt / 1000:.1f} с, {time.time() - t0:.0f} с", flush=True)
    for k, p in pops.items():
        n, s1, s2 = acc[k]
        mean = s1 / n.clamp_min(1)
        sd = (s2 / n.clamp_min(1) - mean ** 2).clamp_min(1e-9).sqrt()
        vt = b.v_t[p.slice]
        gap = vt - mean
        z = gap / sd
        hz = spikes[k] / (steps * dt / 1000)
        q = lambda x: " / ".join(f"{float(v):.2f}" for v in torch.quantile(x.float(), torch.tensor([0.1, 0.5, 0.9], device=x.device)))
        tau = float(b.tau_m[p.start])
        ext_mean = b.noise_mean[p.slice] / dt * b.noise_gain[p.slice] * tau
        ext_sd = b.noise_std[p.slice] * b.noise_gain[p.slice]
        print(f"\n{k} ({p.size} клеток): частота 10/50/90% {q(hz)} Гц, молчат {float((hz == 0).float().mean()):.1%}")
        print(f"  мембрана, мВ: средняя {q(mean)}, СКО {q(sd)}, до порога v_t {q(gap)} мВ = {q(z)} СКО")
        for lo, hi in ((-99, 1), (1, 2), (2, 3), (3, 99)):
            m = (z >= lo) & (z < hi)
            if m.any():
                print(f"  z {lo if lo > -99 else '−∞'}…{hi if hi < 99 else '∞'}: клеток {float(m.float().mean()):.0%}, "
                      f"частота медиана {float(hz[m].median()):.2f} Гц, молчат {float((hz[m] == 0).float().mean()):.0%}")
        r = torch.corrcoef(torch.stack([hz, -z]))[0, 1]
        print(f"  корреляция частоты с близостью к порогу (−z): {float(r):+.2f}")
        print(f"  постоянные части, мВ (медиана; разброс 10–90%): внешний фон {q(ext_mean)} (СКО шага {float(ext_sd.median()):.2f}), "
              f"dc соседей {q(b.dc_mv[p.slice])}, bias+гомеостаз {q(b.bias[p.slice])}, v_t {q(vt)}")


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
