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

Сверка: у бодрствующей макаки на пустом экране (фиксация, whole-cell, 31
клетка V1) медиана расстояния мембраны до порога спайка 13.9 мВ, медиана
асимметрии распределения потенциала +0.72 — хвост в деполяризацию, спайки от
редких крупных коррелированных событий (Tan et al. 2014,
https://doi.org/10.1038/nn.3684, PMC4067243). Поэтому ещё: асимметрия
потенциала каждой клетки; для traced клеток у ямки — полная запись мембраны:
корреляция потенциала и счёта спайков (окна 50 мс) между соседями, крупные
деполяризации (выше среднего клетки на 2 СКО: доля времени, событий в
секунду). Порог в мозге — v_t экспоненты AdEx, в данных — начало спайка
(сопоставимо с точностью до нескольких мВ).

    PYTHONPATH=. .venv/bin/python tools/silence_check.py [pops=visual.L4E,visual.L23E] [seconds=3] [traced=300] [seed=1]
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
    acc = {k: [torch.zeros(p.size, device=b.device) for _ in range(4)] for k, p in pops.items()}
    n_tr = int(opts.get("traced", 300))
    traced = {k: (b.field_cols[p.name] - cfg.fovea_col).abs().argsort()[:n_tr].to(b.device) if p.name in b.field_cols
              else torch.arange(n_tr, device=b.device) for k, p in pops.items()}
    traces = {k: torch.zeros(steps, n_tr, device=b.device) for k in pops}
    fired = {k: torch.zeros(steps, n_tr, device=b.device) for k in pops}
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
            a[3] += v * v * v * free
            spikes[k] += b.spikes[p.slice]
            traces[k][t] = v[traced[k]]
            fired[k][t] = b.spikes[p.slice][traced[k]]
    print(f"пустой экран {steps * dt / 1000:.1f} с, {time.time() - t0:.0f} с", flush=True)
    for k, p in pops.items():
        n, s1, s2, s3 = acc[k]
        mean = s1 / n.clamp_min(1)
        sd = (s2 / n.clamp_min(1) - mean ** 2).clamp_min(1e-9).sqrt()
        skew = (s3 / n.clamp_min(1) - 3 * mean * sd ** 2 - mean ** 3) / sd ** 3
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
        print(f"  асимметрия потенциала {q(skew)} (макака на пустом экране: медиана +0.72)")
        # Соседи у ямки: корреляция потенциала (без шагов спайка/рефрактерности — как есть) и счёта за 50 мс.
        tr, fr = traces[k], fired[k]
        zt = (tr - tr.mean(0)) / tr.std(0).clamp_min(1e-6)
        cv = (zt.T @ zt) / len(zt)
        off = cv[~torch.eye(n_tr, dtype=torch.bool, device=cv.device)]
        w = int(50 / dt)
        cnt = fr[: len(fr) // w * w].view(-1, w, n_tr).sum(1)
        ok = cnt.std(0) > 0
        cc = torch.corrcoef(cnt[:, ok].T)
        offc = cc[~torch.eye(int(ok.sum()), dtype=torch.bool, device=cc.device)]
        high = tr > (tr.mean(0) + 2 * tr.std(0))
        starts = (high[1:] & ~high[:-1]).float().sum(0) / (steps * dt / 1000)
        print(f"  {n_tr} клеток у ямки: корреляция потенциала соседей медиана {float(off.median()):+.3f}, "
              f"счёта за 50 мс {float(offc.nanmedian()):+.3f}; выше среднего + 2 СКО {float(high.float().mean()):.1%} времени, "
              f"событий {float(starts.median()):.1f} в с")
        r = torch.corrcoef(torch.stack([hz, -z]))[0, 1]
        print(f"  корреляция частоты с близостью к порогу (−z): {float(r):+.2f}")
        print(f"  постоянные части, мВ (медиана; разброс 10–90%): внешний фон {q(ext_mean)} (СКО шага {float(ext_sd.median()):.2f}), "
              f"dc соседей {q(b.dc_mv[p.slice])}, bias+гомеостаз {q(b.bias[p.slice])}, v_t {q(vt)}")


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
