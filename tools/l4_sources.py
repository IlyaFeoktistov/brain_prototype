"""
Из чего возбуждение поля V1 L4 E в покое, в первые 50 мс фиксации и после
(DESIGN «LGN → L4 за фиксацию…»: вход таламуса к ~100 мс падает к фону).

Мозг как в reading_bench, без пластичности, неподвижный взгляд, A и B по n
показов. Клетки поля — больше половины входов LGN от зрячих клеток
фиксации. По каждому возбуждающему пути в L4 E — Σ вес × спайки источника
(у пути LGN × ресурс источника, как отдаёт engine; у путей коры без их
кратковременной пластичности — мера поставки), доля от суммы × ток
возбуждения i_syn → мВ по путям; среднее поле AMPA (одно на популяцию) и
фон (bias). Окна: пустой экран 200 мс перед показом, 0–50 и 50–fix мс.
Отдельно — частоты источников (LGN зрячие, L4 поле, L6, L2/3).

    PYTHONPATH=. python tools/l4_sources.py [n=10] [seed=1]
"""

from __future__ import annotations

import sys

import torch

from brain.session import Session
from tools.reading_bench import build

WIN = 10.0


def main(opts: dict) -> None:
    opts = {"fit": "lgn-a", "homeo": f"lgn-b-s{opts.get('seed', 1)}", "relay_v_t_mv": "-59",
            "trn_scale": "0.1", **opts}
    mind, b = build(opts)
    cfg = b.cfg
    s = Session(mind, window_ms=WIN)
    n_rep, fix, step = int(opts.get("n", 10)), float(opts.get("fix", 225)), float(opts.get("step", 5))
    texts = (opts.get("a", "мама мыла раму"), opts.get("b", "папа пила чай"))
    lgn = b.areas["lgn"].E
    vis = b.areas["visual"].layers
    E = vis["L4"]["E"]
    eye_i = [x.name for x in b.sensors].index("eye")
    driver = next(p for p in b.projections if p.name == "eye -> " + lgn.name).pre[:, 0].long() - b.sensor_offsets[eye_i]
    tc = next(p for p in b.projections if p.name == f"lgn -> {E.name}")
    tc_pre = tc.pre.long() - lgn.start
    paths = [p for p in b.projections if p.post is E and p.source == "net" and float(p.weight.mean()) > 0]
    q = int(b.mf_post_id[E.start])
    n_win, early = int(fix // WIN), int(50 // WIN)

    s.line, s.gaze = "", 0.0
    b.run(2500.0, s._blank(2500.0))

    def gazes(line: str) -> list[float]:
        width = s.screen.width(line) / max(1, len(line))
        n = max(1, int((len(line) + step - 1) // step))
        return [i * step * width for i in range(n)][:3]

    def seeing(line: str, gaze: float) -> torch.Tensor:
        s.line, s.gaze = line, gaze
        lit = s.eye.rates(s._picture(s.screen, cfg.fovea_col)) >= 5.0
        return lit.to(driver.device)[driver]

    keys = [p.name for p in paths] + ["среднее поле AMPA", "ток i_syn", "фон bias", "LGN зрячие Гц", "L4 поле Гц",
                                      "L6 E Гц", "L2/3 E Гц"]

    def sample(r: torch.Tensor, see: torch.Tensor, field: torch.Tensor) -> torch.Tensor:
        drives = []
        for p in paths:
            src = r[p.pre.long()]
            if p is tc:
                src = src * b.stp_x[p.pre.long()]
            drives.append((p.weight.float() * src).sum(1)[field].mean())
        drives = torch.stack(drives)
        cur = b.i_syn[E.slice][field].float().mean()
        mf = float((b.mf_exc[q] * b.mf_rate).sum())
        rates = [r[lgn.slice][see].mean() if see.any() else r.new_tensor(0.0), r[E.slice][field].mean(),
                 r[vis["L6"]["E"].slice].mean(), r[vis["L23"]["E"].slice].mean()]
        return torch.cat([drives / drives.sum().clamp_min(1e-9) * cur, cur.new_tensor([mf]), cur[None],
                          b.bias[E.slice][field].mean()[None], torch.stack(rates)]).cpu()

    acc = {"покой": 0, "0–50 мс": 0, f"50–{int(fix)} мс": 0}
    counts = {k: 0 for k in acc}
    for _ in range(n_rep):
        for lab in (0, 1):
            line = texts[lab] + " "
            points = gazes(line)
            see0 = seeing(line, points[0])
            field0 = see0[tc_pre].float().mean(1) > 0.5
            s.line = ""
            b.run(300.0, s._blank(300.0))
            for _ in range(20):
                r = b.observe(WIN, s._blank(WIN))
                acc["покой"] = acc["покой"] + sample(r, see0, field0)
                counts["покой"] += 1
            for i, g in enumerate(points):
                if i:
                    s.line, s.gaze = line, g
                    b.run(cfg.saccade_ms, s._blank(cfg.saccade_ms))
                see = seeing(line, g)
                field = see[tc_pre].float().mean(1) > 0.5
                s.line, s.gaze = line, g
                for t in range(n_win):
                    r = b.observe(WIN, s._view(WIN))
                    k = "0–50 мс" if t < early else f"50–{int(fix)} мс"
                    acc[k] = acc[k] + sample(r, see, field)
                    counts[k] += 1

    print(f"\n=== Возбуждение поля V1 L4 E по источникам, мВ (доля пути × i_syn); неподвижно, по {n_rep} показов A и B")
    print(f"  {'источник':34s} " + " ".join(f"{k:>12s}" for k in acc))
    table = {k: acc[k] / counts[k] for k in acc}
    for j, name in enumerate(keys):
        print(f"  {name:34s} " + " ".join(f"{float(table[k][j]):12.2f}" for k in acc), flush=True)


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
