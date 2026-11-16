"""
Что ведёт залп руки при молчащем VA (DESIGN 2026-11-16 «Ворота базальных
ганглиев»: без подобранного тока VA 0 Гц, а рука жмёт 9–12/с, VLp растёт
до 23–26 Гц). Кандидаты: мозжечковый таламус VLp или собственная динамика
карты руки. Без правки мозга, кроме флага va_bias (по умолчанию −76 —
снять подобранный ток VA; одна величина).

Пустой экран, окна 10 мс, sec секунд. Карта руки одномерная, место клетки —
b.field_cols (0…1), клетки по порядку. Событие — начало нажатия (условие
Hand.pressed в окне t, не в t−1), блок клавиши k. Средний ход за 0…lag×10 мс до
события (лаг в окнах по 10 мс), в долях от среднего за весь прогон:
    VLp, VA — частота ядра;
    L2/3 у клавиши (|место − место блока| ≤ near) и L2/3 вдали (тот же
        размер участка, место через полкарты) — частота;
    у клавиши: поставка VLp → L2/3, VA → L2/3, премоторная → L2/3,
        L2/3 → L2/3 руки (Σ PSP × частота источника, без депрессии — мера);
    L5 блока (доля выстреливших из 8) и L5 у клавиши — частота.
Порядок по времени: что растёт раньше — VLp (вход) или L2/3 / L5 участка.

    PYTHONPATH=. .venv/bin/python tools/hand_onset.py [sec=20] [seed=1] [va_bias=-76] [near=0.005] [lag=10]
"""

from __future__ import annotations

import sys
import time

import torch

from brain.session import Session
from tools.reading_bench import build

WIN = 10.0


def main(opts: dict) -> None:
    seed = int(opts.get("seed", 1))
    opts = {"fit": "lgn-a", "homeo": f"lgn-d-pop-tcind-s{seed}", "relay_v_t_mv": "-59", "trn_scale": "0.1", **opts}
    mind, b = build(opts)
    s = Session(mind, window_ms=WIN)
    sec, va_bias, near = float(opts.get("sec", 20)), float(opts.get("va_bias", -76)), float(opts.get("near", 0.005))
    n_lag = int(opts.get("lag", 10))
    torch.manual_seed(seed)
    A = b.areas
    if va_bias:
        b.bias[A["thalamus"].E.slice] += va_bias
    L23, L5 = A["hand"].layers["L23"]["E"], s.H
    c23 = b.field_cols[L23.name].float().cpu()
    c5 = b.field_cols[L5.name].float().cpu()
    keys, cells = len(s.keyboard), s.hand.cells
    groups = {"VLp → L2/3": ("vl_cb -> hand",), "VA → L2/3": ("thalamus -> hand",), "премоторная → L2/3": ("premotor -> hand",),
              "L2/3 → L2/3": ("hand: E->E",)}
    into = [p for p in b.projections if p.post is L23 and p.source == "net"]
    gp = {g: [p for p in into if any(p.name.startswith(x) for x in pre)] for g, pre in groups.items()}
    print(f"va_bias {va_bias:+.0f} мВ; L2/3 руки {L23.size}, L5 {L5.size}; пути в L2/3 по группам: "
          + "; ".join(f"{g}: {', '.join(p.name for p in ps)}" for g, ps in gp.items()), flush=True)

    s.line, s.gaze = "", 0.0
    b.run(2500.0, s._blank(2500.0))
    n_win = int(sec * 1000 / WIN)
    blank = s._blank(WIN)
    nuc, r23, r5, sup = [], [], [], {g: [] for g in gp}
    t0 = time.time()
    for t in range(n_win):
        r = b.observe(WIN, blank)
        nuc.append(torch.stack([r[A["vl_cb"].E.slice].mean(), r[A["thalamus"].E.slice].mean()]).cpu())
        r23.append(r[L23.slice].half().cpu())
        r5.append(r[L5.slice].half().cpu())
        for g, ps in gp.items():
            v = sum((p.weight.float() * r[p.pre.long()]).sum(1) for p in ps)
            sup[g].append(v.half().cpu())
        if t == 49:
            print(f"  первые 0.5 с: {time.time() - t0:.0f} с", flush=True)
    nuc = torch.stack(nuc)
    r23, r5 = torch.stack(r23).float(), torch.stack(r5).float()
    sup = {g: torch.stack(v).float() for g, v in sup.items()}
    print(f"окон {n_win}, {time.time() - t0:.0f} с", flush=True)

    blk = (r5[:, : keys * cells] > 0).float().view(n_win, keys, cells).mean(2)
    best = blk.argmax(1)
    cond = blk.gather(1, best[:, None]).squeeze(1) >= 0.5
    onset = [(t, int(best[t])) for t in range(n_lag + 1, n_win) if cond[t] and not cond[t - 1]]
    print(f"начал нажатия {len(onset)} ({len(onset) / sec:.1f} в с)", flush=True)
    lags = list(range(n_lag, -1, -1))
    feats: dict[str, list] = {k: [] for k in ("VLp", "VA", "L2/3 у клавиши", "L2/3 вдали", *gp, "L5 блока (доля)", "L5 у клавиши")}
    for t, k in onset:
        pos = float(c5[k * cells: (k + 1) * cells].mean())
        m23 = (c23 - pos).abs() <= near
        far = ((c23 - ((pos + 0.5) % 1.0)).abs() <= near)
        m5 = (c5 - pos).abs() <= near
        ts = torch.tensor([t - d for d in lags])
        feats["VLp"].append(nuc[ts, 0])
        feats["VA"].append(nuc[ts, 1])
        feats["L2/3 у клавиши"].append(r23[ts][:, m23].mean(1))
        feats["L2/3 вдали"].append(r23[ts][:, far].mean(1))
        for g in gp:
            feats[g].append(sup[g][ts][:, m23].mean(1))
        feats["L5 блока (доля)"].append(blk[ts, k])
        feats["L5 у клавиши"].append(r5[ts][:, m5].mean(1))
    base = {"VLp": nuc[:, 0].mean(), "VA": nuc[:, 1].mean(), "L2/3 у клавиши": r23.mean(), "L2/3 вдали": r23.mean(),
            **{g: sup[g].mean() for g in gp}, "L5 блока (доля)": blk.mean(), "L5 у клавиши": r5.mean()}
    print(f"\nсредний ход до начала нажатия, × к среднему (в скобках — среднее): лаг, мс " + " ".join(f"{-d * WIN:>6.0f}" for d in lags))
    for k, v in feats.items():
        if not v:
            continue
        m = torch.stack(v).mean(0)
        b0 = float(base[k])
        row = " ".join(f"{float(x) / b0:6.2f}" if abs(b0) > 1e-9 else "     —" for x in m)
        print(f"  {k:22s} ({b0:8.2f}) {row}", flush=True)
    # корреляция по времени: VLp против L2/3 и L5 всей руки
    def xc(a: torch.Tensor, c: torch.Tensor, lag: int) -> float:
        a, c = a[: len(a) - lag] - a.mean(), c[lag:] - c.mean()
        return float((a * c).sum() / (a.norm() * c.norm()).clamp_min(1e-12))
    g23, g5 = r23.mean(1), r5.mean(1)
    print("\nкорреляция всей руки: X(t) → Y(t + лаг), лаг 0/10/20/50/100 мс")
    for nm, a, c in (("VLp → L2/3", nuc[:, 0], g23), ("VLp → L5", nuc[:, 0], g5), ("L2/3 → L5", g23, g5), ("L5 → L2/3", g5, g23),
                     ("L2/3 → VLp", g23, nuc[:, 0])):
        print(f"  {nm:12s} " + " ".join(f"{xc(a, c, l):+.2f}" for l in (0, 1, 2, 5, 10)), flush=True)
    print(f"всего {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
