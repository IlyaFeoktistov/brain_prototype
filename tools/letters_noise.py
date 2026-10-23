"""
Помогает ли шум различению в зоне букв (стохастический резонанс — Douglass
et al. 1993, https://doi.org/10.1038/365337a0: шум помогает подпороговому
сигналу, пока собственного шума меньше оптимума).

Мозг как в reading_bench, без пластичности. Шум — у каждой клетки букв L4 E
свой сдвиг покоя (bias), новый каждые WIN мс, среднее 0, sd по уровням
levels (мВ). Уровни перемешаны по показам в одном мозге (каждый показ —
своя реализация шума); на уровень — n показов каждой фразы, чтением.
Меры по уровню: точность A/B (ближайший центр, выброс одного), разность
A−B к разбросу повторов (среднее по клеткам), частота, доля клеток без
спайков за все показы уровня — для букв L4 и L2/3 E.

    PYTHONPATH=. python tools/letters_noise.py [levels=0,1,3] [n=20] [seed=1]
"""

from __future__ import annotations

import sys

import torch

from brain.session import Session
from tools.letters_drive import loo
from tools.reading_bench import build

WIN = 10.0


def main(opts: dict) -> None:
    opts = {"fit": "lgn-a", "homeo": f"lgn-a-s{opts.get('seed', 1)}", "relay_v_t_mv": "-59",
            "relay_rest_mv": "-48", "trn_scale": "0.5", **opts}
    mind, b = build(opts)
    cfg = b.cfg
    s = Session(mind, window_ms=WIN)
    levels = [float(x) for x in opts.get("levels", "0,1,3").split(",")]
    n_rep, fix, step = int(opts.get("n", 20)), float(opts.get("fix", 225)), float(opts.get("step", 5))
    texts = (opts.get("a", "мама мыла раму"), opts.get("b", "папа пила чай"))
    gen = torch.Generator().manual_seed(int(opts.get("seed", 1)))
    noise_gen = torch.Generator(device=b.device).manual_seed(int(opts.get("seed", 1)) + 100)
    let = b.areas["letters"].layers
    L4 = let["L4"]["E"]
    stages = {"буквы L4": L4, "буквы L2/3": let["L23"]["E"]}
    base = b.bias[L4.slice].clone()

    s.line, s.gaze = "", 0.0
    b.run(2500.0)

    def gazes(line: str) -> list[float]:
        width = s.screen.width(line) / max(1, len(line))
        n = max(1, int((len(line) + step - 1) // step))
        return [i * step * width for i in range(n)][:3]

    def show(lab: int, sd: float) -> dict[str, torch.Tensor]:
        s.line = ""
        b.run(500.0, s._blank(500.0))
        line = texts[lab] + " "
        rows = {k: [] for k in stages}
        for i, g in enumerate(gazes(line)):
            if i:
                s.line, s.gaze = line, g
                b.run(cfg.saccade_ms, s._blank(cfg.saccade_ms))
            s.line, s.gaze = line, g
            acc = {k: 0 for k in stages}
            for _ in range(int(fix // WIN)):
                if sd > 0:
                    b.bias[L4.slice] = base + sd * torch.randn(L4.size, device=b.device, generator=noise_gen)
                r = b.observe(WIN, s._view(WIN)) * (WIN / 1000.0)
                for k, pop in stages.items():
                    acc[k] = acc[k] + r[pop.slice].cpu()
            b.bias[L4.slice] = base
            for k in stages:
                rows[k].append(acc[k])
        return {k: torch.cat(v) for k, v in rows.items()}

    plan = [(lev, lab) for lev in range(len(levels)) for lab in (0, 1) for _ in range(n_rep)]
    plan = [plan[i] for i in torch.randperm(len(plan), generator=gen)]
    got = {lev: ([], []) for lev in range(len(levels))}
    for j, (lev, lab) in enumerate(plan):
        got[lev][0].append(show(lab, levels[lev]))
        got[lev][1].append(lab)
        if (j + 1) % 30 == 0:
            print(f"  показов {j + 1}/{len(plan)}", flush=True)

    print(f"\n=== Шум на буквах L4 E: «{texts[0]}» против «{texts[1]}», по {n_rep} показов на фразу и уровень")
    print(f"  {'sd, мВ':>6s} {'ступень':12s} {'сп/кл/фикс':>10s} {'молчат':>7s} {'A−B/шум':>8s} {'точность':>9s} {'случайно':>9s}")
    for lev, sd in enumerate(levels):
        feats, labels = got[lev]
        y = torch.tensor(labels)
        for k in stages:
            x = torch.stack([f[k] for f in feats])
            ma, mb = x[y == 0].mean(0), x[y == 1].mean(0)
            noise = torch.cat([x[y == 0] - ma, x[y == 1] - mb]).std(0)
            live = noise > 0
            snr = float(((ma - mb).abs()[live] / noise[live]).mean())
            cells = x.view(len(x), 3, -1).sum((0, 1))
            silent = float((cells == 0).float().mean())
            ch = float(torch.tensor([loo(x, y[torch.randperm(len(y), generator=gen)]) for _ in range(20)]).mean())
            print(f"  {sd:6.1f} {k:12s} {float(x.mean()):10.3f} {silent:7.2f} {snr:8.3f} {loo(x, y):9.2f} {ch:9.2f}",
                  flush=True)


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
