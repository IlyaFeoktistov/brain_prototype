"""
Аудит веток дендрита L2/3 E (DESIGN 2026-11-17 «Что гонит моторные зоны»:
ток веток через L1 23–56 мВ при пороге плато 10 мВ). Мозг не меняется.

Устройство (engine, events.cu): у клетки cfg.dendrite_branches веток; синапс
пути с признаком «на дендрит» (моторный таламус → L1, обратные пути, SOM → E)
кладёт свой вес в ток ветки net_branch (случайная ветка, свой генератор) —
до истощения, без депрессии синапса; ток ветки спадает с dendrite_tau_ms.
Плато ветки = сигмоида((ток − порог) / крутизна), клетка берёт самую горячую
ветку: + dendrite_gain × плато × возб. ток сомы + dendrite_soma_mv × плато.

Что меряется по популяциям (pops), в двух состояниях VA (va=0,-76):
1. дендритные пути: входов на клетку, вес (ед. движка), входов на ветку
   (среднее, 10–90% по клеткам-веткам);
2. снимок i_dend в конце каждого окна WIN мс: ток самой горячей ветки
   (медиана, 10–90%), доля клеток-окон с плато > 0.5 (самая горячая ветка),
   доля веток выше порога;
3. ток ветки по источникам — равновесная оценка Σ вес × частота входа ×
   dendrite_tau / 1000 по каждой ветке (как движок: без депрессии): среднее
   по веткам и доля веток, которые один источник сам доводит до порога,
   против доли, которые доводит только сумма источников;
4. сверка оценки (сумма источников) со снимком движка (среднее по веткам);
5. для сравнения — та же оценка с депрессией (ресурс источника, как у
   синапса на сому).

    PYTHONPATH=. .venv/bin/python tools/dend_audit.py [seed=1] [va=0,-76] [seconds=3]
        [pops=motor.L23E,hand.L23E,premotor.L23E,visual.L23E,letters.L23E]
"""

from __future__ import annotations

import sys
import time

import torch

from brain.session import Session
from tools.reading_bench import build

WIN = 10.0


def q(x: torch.Tensor, p: float) -> float:
    return float(x.float().quantile(p)) if x.numel() < 16_000_000 else float(x.float()[:: max(1, x.numel() // 1_000_000)].quantile(p))


def run(opts: dict, va: float) -> None:
    seed = int(opts.get("seed", 1))
    o = {"fit": "lgn-a", "homeo": f"lgn-d-pop-tcind-s{seed}", "relay_v_t_mv": "-59", "trn_scale": "0.1", **opts}
    mind, b = build(o)
    cfg = b.cfg
    s = Session(mind, window_ms=WIN)
    nb = max(1, cfg.dendrite_branches)
    thr, slope, tau = cfg.dendrite_threshold_mv, cfg.dendrite_slope_mv, cfg.dendrite_tau_ms
    if va:
        b.bias[b.areas["thalamus"].E.slice] += va
    print(f"\n######## va_bias {va:+.0f}; веток {nb}, порог {thr} мВ, крутизна {slope}, τ {tau} мс, "
          f"усиление {cfg.dendrite_gain}, в сому {cfg.dendrite_soma_mv} мВ", flush=True)

    targets = []
    for key in opts.get("pops", "motor.L23E,hand.L23E,premotor.L23E,visual.L23E,letters.L23E").split(","):
        area, rest = key.split(".")
        pop = b.areas[area].layers[rest[:3]][rest[3:]]
        paths = []
        for p in b.projections:
            if p.post.start != pop.start or p.post.size != pop.size or p.source != "net" or not torch.is_tensor(getattr(p, "pre", None)):
                continue
            pre = p.pre.long()
            if getattr(p, "dendrite", False) or bool((b.dendrite_pre[pre] & b.dendrite_post[pop.start]).any()):
                br = p.branch.long()
                paths.append((p, pre, p.weight.float(), br))
        targets.append((key, pop, paths))
        print(f"\n== {key}: дендритные пути", flush=True)
        for p, pre, w, br in paths:
            cnt = torch.stack([(br == j).sum(1) for j in range(nb)], 1).float()
            print(f"  {p.name:34s} входов {pre.shape[1]:4d}, вес {float(w.mean()):+.3f}; на ветку среднее {float(cnt.mean()):.1f}, "
                  f"10–90% {q(cnt, 0.1):.0f}–{q(cnt, 0.9):.0f}", flush=True)

    s.line, s.gaze = "", 0.0
    b.run(2500.0, s._blank(2500.0))
    blank = s._blank(WIN)
    n_win = int(float(opts.get("seconds", 3)) * 1000 / WIN)
    acc = {key: {"hot": [], "pl": [], "above": [], "src": {}, "srcd": {}, "alone": {}, "sum": [], "snap": [], "rate": []} for key, _, _ in targets}
    t0 = time.time()
    for t in range(n_win):
        r = b.observe(WIN, blank)
        r = torch.cat([r, torch.zeros(b.n_slots - b.n, device=r.device)])
        for key, pop, paths in targets:
            A = acc[key]
            br_now = b.i_dend.view(b.n_slots, nb)[pop.slice]
            hot = br_now.amax(1)
            A["hot"].append(hot.half().cpu())
            A["pl"].append(float((torch.sigmoid((hot - thr) / slope) > 0.5).float().mean()))
            A["above"].append(float((br_now > thr).float().mean()))
            A["snap"].append(float(br_now.mean()))
            A["rate"].append(float(r[pop.slice].mean()))
            if t % 10:
                continue
            total = torch.zeros(pop.size, nb, device=b.device)
            alone = torch.zeros(pop.size, nb, dtype=torch.bool, device=b.device)
            for p, pre, w, br in paths:
                sup = w * r[pre] * tau / 1000.0
                per = torch.zeros(pop.size, nb, device=b.device).scatter_add_(1, br, sup)
                x = b.stp_x[pre] / (1.0 - b.stp_u_cell[pre]).clamp_min(1e-3)
                perd = torch.zeros(pop.size, nb, device=b.device).scatter_add_(1, br, sup * torch.where(b.stp_u_cell[pre] > 0, x, 1.0))
                total += per
                alone |= per > thr
                A["src"][p.name] = A["src"].get(p.name, 0.0) + float(per.mean())
                A["srcd"][p.name] = A["srcd"].get(p.name, 0.0) + float(perd.mean())
            A["sum"].append((float(total.mean()), float((total > thr).float().mean()), float(alone.float().mean())))
        if t == 0:
            print(f"\n  первое окно: {time.time() - t0:.1f} с", flush=True)
    print(f"  окон {n_win}, {time.time() - t0:.0f} с", flush=True)
    for key, pop, paths in targets:
        A = acc[key]
        hot = torch.cat(A["hot"]).float()
        ns = len(A["sum"])
        est, est_above, est_alone = (sum(x[i] for x in A["sum"]) / ns for i in range(3))
        print(f"\n== {key}: {sum(A['rate']) / n_win:.2f} Гц; самая горячая ветка медиана {q(hot, 0.5):.1f} мВ (10–90% "
              f"{q(hot, 0.1):.1f}–{q(hot, 0.9):.1f}); плато > 0.5: {sum(A['pl']) / n_win:.0%} клеток-окон; веток выше порога "
              f"{sum(A['above']) / n_win:.0%}", flush=True)
        print(f"  сверка: ток ветки среднее — движок {sum(A['snap']) / n_win:.2f}, оценка по источникам {est:.2f} мВ; веток выше "
              f"порога по оценке {est_above:.0%}, из них один источник сам — {est_alone:.0%} веток", flush=True)
        for name in sorted(A["src"], key=lambda n: -A["src"][n]):
            print(f"    {name:34s} на ветку {A['src'][name] / ns:6.2f} мВ (с депрессией было бы {A['srcd'][name] / ns:6.2f})", flush=True)
    del mind, b
    torch.cuda.empty_cache()


def main(opts: dict) -> None:
    for va in (float(x) for x in opts.get("va", "0,-76").split(",")):
        run(opts, va)


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
