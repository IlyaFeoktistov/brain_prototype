"""
Где в V1 теряется вызванный ответ на текст: L4 или переход L4 → L2/3
(DESIGN «Возбуждение букв — на 87% вход V1, но он почти одинаков для A и B»).

Мозг как в reading_bench, без пластичности, ничего не меняется. A и B
вперемешку по n показов чтением; перед каждым показом пустой экран 500 мс,
его последние 200 мс — спонтанная частота клетки. У V1 L4 E и L2/3 E:
- вызванная минус спонтанная (Гц) у клеток «на штрихе» (L4 — больше
  половины входов LGN от зрячих клеток этой фиксации; L2/3 — столбец поля в
  столбце с чернилами этой фиксации) и у прочих, ранние 50 мс и вся
  фиксация, и отношение вызванной к спонтанной;
- доля клеток с воспроизводимым ответом на текст: t (показ − свой фон) >
  t_sel на первой половине показов и тот же знак на второй;
- надёжность различия A/B — корреляция по клеткам разности A−B первой и
  второй половины показов (спайки по фиксациям);
- переход: у L2/3 E вход от L4 (Σ вес × спайки L4 E, путь «visual:
  L4.E->E») — надёжность его A−B и корреляция по клеткам A−B входа с A−B
  спайков клетки (сколько различия входа проходит в выход).

    PYTHONPATH=. python tools/v1_layers.py [n=30] [t_sel=2.6] [seed=1]
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
    n_rep, t_sel = int(opts.get("n", 30)), float(opts.get("t_sel", 2.6))
    fix, step = float(opts.get("fix", 225)), float(opts.get("step", 5))
    texts = (opts.get("a", "мама мыла раму"), opts.get("b", "папа пила чай"))
    gen = torch.Generator().manual_seed(int(opts.get("seed", 1)))
    lgn = b.areas["lgn"].E
    vis = b.areas["visual"].layers
    L4, L23 = vis["L4"]["E"], vis["L23"]["E"]
    eye_i = [x.name for x in b.sensors].index("eye")
    driver = next(p for p in b.projections if p.name == "eye -> " + lgn.name).pre[:, 0].long() - b.sensor_offsets[eye_i]
    tc_pre = next(p for p in b.projections if p.name == f"lgn -> {L4.name}").pre.long() - lgn.start
    ff = next(p for p in b.projections if p.name == "visual: L4.E->E" and p.post is L23)
    ff_pre = ff.pre.long() - L4.start
    cols23 = b.field_cols[L23.name].cpu()
    R, C = cfg.retina_rows, cfg.retina_cols

    s.line, s.gaze = "", 0.0
    b.run(2500.0)

    def seen(line: str, gaze: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Клетки «на штрихе»: L4 по входам LGN, L2/3 по столбцу поля."""

        s.line, s.gaze = line, gaze
        rates = s.eye.rates(s._picture(s.screen, cfg.fovea_col))
        lit = (rates >= 5.0).to(driver.device)
        relay = lit[driver]
        on4 = relay[tc_pre].float().mean(1) > 0.5
        ink_cols = lit.view(2, R, C).any(0).any(0).cpu()
        on23 = ink_cols[cols23.round().long().clamp(0, C - 1)]
        return on4.cpu(), on23

    def gazes(line: str) -> list[float]:
        width = s.screen.width(line) / max(1, len(line))
        n = max(1, int((len(line) + step - 1) // step))
        return [i * step * width for i in range(n)][:3]

    def show(lab: int) -> dict:
        s.line = ""
        b.run(300.0, s._blank(300.0))
        spont4, spont23 = 0, 0
        for _ in range(20):
            r = b.observe(WIN, s._blank(WIN))
            spont4 = spont4 + r[L4.slice].cpu() / 20
            spont23 = spont23 + r[L23.slice].cpu() / 20
        line = texts[lab] + " "
        out = {"spont4": spont4, "spont23": spont23, "early4": [], "full4": [], "early23": [], "full23": [],
               "in23": [], "on4": [], "on23": []}
        for i, g in enumerate(gazes(line)):
            if i:
                s.line, s.gaze = line, g
                b.run(cfg.saccade_ms, s._blank(cfg.saccade_ms))
            on4, on23 = seen(line, g)
            s.line, s.gaze = line, g
            acc = {k: 0 for k in ("early4", "full4", "early23", "full23", "in23")}
            n_win = int(fix // WIN)
            for t in range(n_win):
                r = b.observe(WIN, s._view(WIN))
                r4, r23 = r[L4.slice], r[L23.slice]
                acc["full4"] = acc["full4"] + r4.cpu() / n_win
                acc["full23"] = acc["full23"] + r23.cpu() / n_win
                acc["in23"] = acc["in23"] + (ff.weight.float() * r4[ff_pre]).sum(1).cpu() / n_win
                if t < 50 // WIN:
                    acc["early4"] = acc["early4"] + r4.cpu() / (50 // WIN)
                    acc["early23"] = acc["early23"] + r23.cpu() / (50 // WIN)
            for k, v in acc.items():
                out[k].append(v)
            out["on4"].append(on4)
            out["on23"].append(on23)
        return out

    order = [0, 1] * n_rep
    order = [order[i] for i in torch.randperm(len(order), generator=gen)]
    shows = []
    for j, lab in enumerate(order):
        shows.append(show(lab))
        if (j + 1) % 20 == 0:
            print(f"  показов {j + 1}/{len(order)}", flush=True)
    y = torch.tensor(order)
    half = len(order) // 2
    parts = (slice(0, half), slice(half, None))

    print(f"\n=== V1: вызванный ответ на текст и различие A/B; «{texts[0]}» против «{texts[1]}», по {n_rep} показов")
    for layer, size in (("4", L4.size), ("23", L23.size)):
        name = "L4 E" if layer == "4" else "L2/3 E"
        spont = torch.stack([sh[f"spont{layer}"] for sh in shows])            # [показ, клетка]
        on = torch.stack([torch.stack(sh[f"on{layer}"]) for sh in shows])     # [показ, фикс, клетка]
        full = torch.stack([torch.stack(sh[f"full{layer}"]) for sh in shows])
        early = torch.stack([torch.stack(sh[f"early{layer}"]) for sh in shows])
        sp = spont[:, None, :].expand_as(full)
        line = []
        for tag, x in (("ранние 50 мс", early), ("вся фиксация", full)):
            ev_on, sp_on = x[on].mean(), sp[on].mean()
            ev_off, sp_off = x[~on].mean(), sp[~on].mean()
            line.append(f"{tag}: на штрихе {float(sp_on):.2f} → {float(ev_on):.2f} Гц (×{float(ev_on / sp_on.clamp_min(1e-9)):.2f}), "
                        f"прочие {float(sp_off):.2f} → {float(ev_off):.2f}")
        frac_on = float(on.float().mean())
        # Воспроизводимый ответ на текст: показ (среднее по фиксациям) против своего фона.
        resp = full.mean(1) - spont
        def tval(z: torch.Tensor) -> torch.Tensor:
            return z.mean(0) / (z.std(0) / len(z) ** 0.5).clamp_min(1e-9)
        t1, t2 = tval(resp[parts[0]]), tval(resp[parts[1]])
        picked = t1.abs() > t_sel
        rep = picked & (torch.sign(t2) == torch.sign(t1))
        up = rep & (t1 > 0)
        # Надёжность A−B по фиксациям.
        x = full.reshape(len(shows), -1)
        d = [x[p][y[p] == 0].mean(0) - x[p][y[p] == 1].mean(0) for p in parts]
        print(f"\n  V1 {name} ({size} клеток; «на штрихе» — {frac_on * 100:.0f}% клеток-фиксаций)")
        for l in line:
            print(f"    {l}")
        print(f"    ответ на текст воспроизводим у {float(rep.float().mean()) * 100:.1f}% клеток "
              f"(рост {float(up.float().mean()) * 100:.1f}%, спад {float((rep & (t1 < 0)).float().mean()) * 100:.1f}%; "
              f"отобрано на 1-й половине {float(picked.float().mean()) * 100:.1f}%)")
        print(f"    надёжность половин A−B: {corr(d[0], d[1]):+.3f}", flush=True)
        if layer == "23":
            inp = torch.stack([torch.stack(sh["in23"]) for sh in shows]).reshape(len(shows), -1)
            di = [inp[p][y[p] == 0].mean(0) - inp[p][y[p] == 1].mean(0) for p in parts]
            da = inp[y == 0].mean(0) - inp[y == 1].mean(0)
            do = x[y == 0].mean(0) - x[y == 1].mean(0)
            print(f"    вход от L4 (Σ вес × спайки L4): среднее {float(inp.mean()):.2f}, надёжность половин A−B "
                  f"{corr(di[0], di[1]):+.3f}; корр. A−B входа с A−B спайков L2/3 {corr(da, do):+.3f}", flush=True)


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
