"""
Где за фиксацию затухает сигнал LGN → V1 L4 (DESIGN «Движение глаза при
фиксации…»: даже при дрейфе поздняя частота L4 ≈ фону).

Мозг как в reading_bench, без пластичности, параметры не меняются. Условия
перемешаны по показам: неподвижно / дрейф + микросаккада (траектории
fixation_motion — одни на все повторы). По окнам 10 мс фиксации, у клеток,
видящих штрих в этом окне (при дрейфе — пересчёт на каждом окне):
LGN — частота и ресурс синапсов (stp_x); поле L4 (больше половины входов LGN
от зрячих в начале фиксации) — эффективный вход таламуса (Σ вес × спайк
источника × его ресурс; так engine отдаёт синапс ядра с депрессией),
ток возбуждения (i_syn), мембрана, частота. Фон — те же клетки на пустом
экране перед показом (200 мс). Надёжность A−B поздних 50–fix мс по
половинам показов: эффективный вход, ток, спайки поля L4.

    PYTHONPATH=. python tools/lgn_l4_timeline.py [n=15] [drift=15] [micro=4] [ms_at=110] [seed=1]
"""

from __future__ import annotations

import sys

import torch

from brain.session import Session
from tools.reading_bench import build

WIN = 10.0


def corr(u: torch.Tensor, v: torch.Tensor) -> float:
    u, v = u - u.mean(), v - v.mean()
    return float((u * v).sum() / (u.norm() * v.norm()).clamp_min(1e-12))


def main(opts: dict) -> None:
    opts = {"fit": "lgn-a", "homeo": f"lgn-a-s{opts.get('seed', 1)}", "relay_v_t_mv": "-59",
            "relay_rest_mv": "-48", "trn_scale": "0.5", **opts}
    mind, b = build(opts)
    cfg = b.cfg
    s = Session(mind, window_ms=WIN)
    n_rep, fix, step = int(opts.get("n", 15)), float(opts.get("fix", 225)), float(opts.get("step", 5))
    drift, micro = float(opts.get("drift", 15)), float(opts.get("micro", 4))
    ms_at, turn = float(opts.get("ms_at", 110)), float(opts.get("turn", 0.1))
    texts = (opts.get("a", "мама мыла раму"), opts.get("b", "папа пила чай"))
    gen = torch.Generator().manual_seed(int(opts.get("seed", 1)))
    lgn = b.areas["lgn"].E
    L4 = b.areas["visual"].layers["L4"]["E"]
    eye_i = [x.name for x in b.sensors].index("eye")
    driver = next(p for p in b.projections if p.name == "eye -> " + lgn.name).pre[:, 0].long() - b.sensor_offsets[eye_i]
    tc = next(p for p in b.projections if p.name == f"lgn -> {L4.name}")
    tc_pre = tc.pre.long() - lgn.start
    n_win = int(fix // WIN)
    early = int(50 // WIN)

    path_gen = torch.Generator().manual_seed(1000 + int(opts.get("seed", 1)))
    trajectories = []
    for _ in range(3):
        x, sign, row = 0.0, 1.0, []
        for _ in range(n_win):
            if float(torch.rand(1, generator=path_gen)) < turn:
                sign = -sign
            x += sign * drift * WIN / 1000.0
            row.append(x)
        trajectories.append(row)
    conditions = {
        "неподвижно": lambda f, t: 0.0,
        "дрейф + микросаккада": lambda f, t: trajectories[f][t] + (micro if t * WIN >= ms_at else 0.0),
    }

    s.line, s.gaze = "", 0.0
    b.run(2500.0)

    def seeing(line: str, gaze: float) -> torch.Tensor:
        s.line, s.gaze = line, gaze
        lit = s.eye.rates(s._picture(s.screen, cfg.fovea_col)) >= 5.0
        return lit.to(driver.device)[driver]

    def gazes(line: str) -> list[float]:
        width = s.screen.width(line) / max(1, len(line))
        n = max(1, int((len(line) + step - 1) // step))
        return [i * step * width for i in range(n)][:3]

    names = ("LGN зрячие, Гц", "ресурс зрячих", "эфф. вход L4", "ток L4", "V L4, мВ", "L4 поле, Гц")

    def measure(r: torch.Tensor, see: torch.Tensor, field: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Покадровые меры (средние по клеткам) и векторы по клеткам поля."""

        eff = (tc.weight.float() * r[lgn.slice][tc_pre] * b.stp_x[lgn.slice][tc_pre]).sum(1)
        cur = b.i_syn[L4.slice].float()
        spk = r[L4.slice]
        pick = lambda v, m: v[m].mean() if m.any() else v.new_tensor(0.0)
        row = torch.stack([pick(r[lgn.slice], see), pick(b.stp_x[lgn.slice], see), pick(eff, field),
                           pick(cur, field), pick(b.v[L4.slice], field), pick(spk, field)]).cpu()
        return row, eff.cpu(), cur.cpu(), spk.cpu()

    def show(lab: int, move) -> dict:
        s.line = ""
        b.run(300.0, s._blank(300.0))
        line = texts[lab] + " "
        first = seeing(line, gazes(line)[0])
        field0 = first[tc_pre].float().mean(1) > 0.5
        base = 0
        for _ in range(20):
            s.line = ""
            r = b.observe(WIN, s._blank(WIN))
            base = base + measure(r, first, field0)[0] / 20
        trace, late = [], {"eff": [], "cur": [], "spk": []}
        for i, g in enumerate(gazes(line)):
            if i:
                s.line, s.gaze = line, g
                b.run(cfg.saccade_ms, s._blank(cfg.saccade_ms))
            field = seeing(line, g)[tc_pre].float().mean(1) > 0.5
            acc = {k: 0 for k in late}
            rows = []
            for t in range(n_win):
                gaze = g + move(i, t)
                see = seeing(line, gaze)
                s.line, s.gaze = line, gaze
                r = b.observe(WIN, s._view(WIN))
                row, eff, cur, spk = measure(r, see, field)
                rows.append(row)
                if t >= early:
                    acc["eff"] = acc["eff"] + eff / (n_win - early)
                    acc["cur"] = acc["cur"] + cur / (n_win - early)
                    acc["spk"] = acc["spk"] + spk / (n_win - early)
            trace.append(torch.stack(rows))
            for k in late:
                late[k].append(acc[k])
        return {"base": base, "trace": torch.stack(trace).mean(0), "late": {k: torch.cat(v) for k, v in late.items()}}

    plan = [(c, lab) for c in conditions for lab in (0, 1) for _ in range(n_rep)]
    plan = [plan[i] for i in torch.randperm(len(plan), generator=gen)]
    got = {c: ([], []) for c in conditions}
    for j, (c, lab) in enumerate(plan):
        got[c][0].append(show(lab, conditions[c]))
        got[c][1].append(lab)
        if (j + 1) % 20 == 0:
            print(f"  показов {j + 1}/{len(plan)}", flush=True)

    print(f"\n=== LGN → V1 L4 за фиксацию ({fix:g} мс, окна {WIN:g} мс; среднее по 3 фиксациям и показам); "
          f"по {n_rep} показов фразы на условие")
    for c in conditions:
        shows, labels = got[c]
        y = torch.tensor(labels)
        base = torch.stack([sh["base"] for sh in shows]).mean(0)
        tr = torch.stack([sh["trace"] for sh in shows]).mean(0)
        print(f"\n  {c}")
        for j, name in enumerate(names):
            fmt = "{:6.2f}" if name.startswith("ресурс") else "{:6.1f}"
            print(f"    {name:16s} фон {fmt.format(float(base[j]))} | " + " ".join(fmt.format(float(v)) for v in tr[:, j]))
        half = len(y) // 2
        parts = (slice(0, half), slice(half, None))
        rel = []
        for k, label in (("eff", "эфф. вход"), ("cur", "ток"), ("spk", "спайки")):
            x = torch.stack([sh["late"][k] for sh in shows])
            d = [x[p][y[p] == 0].mean(0) - x[p][y[p] == 1].mean(0) for p in parts]
            rel.append(f"{label} {corr(d[0], d[1]):+.3f}")
        print(f"    надёжность A−B у всех L4 E, 50–{int(fix)} мс: " + ", ".join(rel), flush=True)


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
