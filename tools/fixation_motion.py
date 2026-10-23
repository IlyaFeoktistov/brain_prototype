"""
Держит ли движение глаза при фиксации ответ V1 после первых 50 мс
(DESIGN «Гомеостаз популяции…»: у V1 L2/3 за фиксацию ×1.01 — только
всплеск первых 50 мс).

Данные: дрейф макаки — пиковая скорость ~33–45 угл. мин/с (Malevich,
Buonocore & Hafed 2020, https://doi.org/10.7554/eLife.57595); фиксационные
саккады ≤100 угл. мин, типично ~10 (Snodderly, Kagan & Gur 2001,
https://doi.org/10.1017/S0952523801182118); при дрейфе ~2/3 клеток V1
держат разряд, пока стимул в поле (клетки «положения/дрейфа» 38% — 80 ± 24%
ответа после саккады, смешанные 37% — 38 ± 18%, саккадные 25% — 4 ± 11%;
обзор Gur/Kagan/Snodderly, PMC4475509).

Перенос: буква у нас ~7 px, при чтении ~0.25–0.3° → 1 px ≈ 2.5 угл. мин
[оценка]: дрейф ~15 px/с (≈ 3 px за фиксацию), микросаккада ~4 px. Сдвиг
картинки — только по горизонтали и на целый пиксель (Session._picture):
дрейф ступенчатый. Траектории одни на все повторы (не новые на показ).

Условия (перемешаны по показам в одном мозге, по n показов фразы на
условие): неподвижно / дрейф (случайное блуждание с постоянной скоростью
drift px/с, направление меняется с вероятностью turn за 10 мс) / дрейф +
микросаккада (скачок micro px на ms_at мс фиксации).
Меры — у V1 L4 E, V1 L2/3 E, буквы L4 E, L2/3 E: фон (200 мс пустого перед
показом), вызванная ранние 0–50 мс и поздние 50–fix мс (среднее по клеткам,
Гц), надёжность половин A−B поздних (корреляция по клеткам A−B первой и
второй половины показов), точность популяции поздних (центры по первой
половине → вторая).

    PYTHONPATH=. python tools/fixation_motion.py [n=20] [drift=15] [micro=4] [ms_at=110] [turn=0.1] [seed=1]
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
    n_rep, fix, step = int(opts.get("n", 20)), float(opts.get("fix", 225)), float(opts.get("step", 5))
    drift, micro = float(opts.get("drift", 15)), float(opts.get("micro", 4))
    ms_at, turn = float(opts.get("ms_at", 110)), float(opts.get("turn", 0.1))
    texts = (opts.get("a", "мама мыла раму"), opts.get("b", "папа пила чай"))
    gen = torch.Generator().manual_seed(int(opts.get("seed", 1)))
    vis, let = b.areas["visual"].layers, b.areas["letters"].layers
    stages = {"V1 L4": vis["L4"]["E"], "V1 L2/3": vis["L23"]["E"], "буквы L4": let["L4"]["E"],
              "буквы L2/3": let["L23"]["E"]}
    n_win = int(fix // WIN)
    early = int(50 // WIN)

    # Траектории смещения взгляда за фиксацию (px, по окнам) — одни на все повторы.
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
        "дрейф": lambda f, t: trajectories[f][t],
        "дрейф + микросаккада": lambda f, t: trajectories[f][t] + (micro if t * WIN >= ms_at else 0.0),
    }

    s.line, s.gaze = "", 0.0
    b.run(2500.0)

    def gazes(line: str) -> list[float]:
        width = s.screen.width(line) / max(1, len(line))
        n = max(1, int((len(line) + step - 1) // step))
        return [i * step * width for i in range(n)][:3]

    def show(lab: int, move) -> dict[str, dict[str, torch.Tensor]]:
        s.line = ""
        b.run(300.0, s._blank(300.0))
        spont = {k: 0 for k in stages}
        for _ in range(20):
            r = b.observe(WIN, s._blank(WIN))
            for k, pop in stages.items():
                spont[k] = spont[k] + r[pop.slice].cpu() / 20
        line = texts[lab] + " "
        out = {k: {"spont": spont[k], "early": [], "late": []} for k in stages}
        for i, g in enumerate(gazes(line)):
            if i:
                s.line, s.gaze = line, g
                b.run(cfg.saccade_ms, s._blank(cfg.saccade_ms))
            acc_e = {k: 0 for k in stages}
            acc_l = {k: 0 for k in stages}
            for t in range(n_win):
                s.line, s.gaze = line, g + move(i, t)
                r = b.observe(WIN, s._view(WIN))
                for k, pop in stages.items():
                    if t < early:
                        acc_e[k] = acc_e[k] + r[pop.slice].cpu() / early
                    else:
                        acc_l[k] = acc_l[k] + r[pop.slice].cpu() / (n_win - early)
            for k in stages:
                out[k]["early"].append(acc_e[k])
                out[k]["late"].append(acc_l[k])
        return out

    plan = [(c, lab) for c in conditions for lab in (0, 1) for _ in range(n_rep)]
    plan = [plan[i] for i in torch.randperm(len(plan), generator=gen)]
    got = {c: ([], []) for c in conditions}
    for j, (c, lab) in enumerate(plan):
        got[c][0].append(show(lab, conditions[c]))
        got[c][1].append(lab)
        if (j + 1) % 30 == 0:
            print(f"  показов {j + 1}/{len(plan)}", flush=True)

    print(f"\n=== Движение глаза при фиксации: «{texts[0]}» против «{texts[1]}», по {n_rep} показов фразы на условие; "
          f"дрейф {drift:g} px/с, микросаккада {micro:g} px на {ms_at:g} мс")
    print(f"  {'условие':22s} {'ступень':11s} {'фон':>5s} {'0–50':>6s} {'50–' + str(int(fix)):>7s} "
          f"{'поздн./фон':>10s} {'надёжн. A−B':>12s} {'точность':>9s}")
    for c in conditions:
        shows, labels = got[c]
        y = torch.tensor(labels)
        half = len(y) // 2
        parts = (slice(0, half), slice(half, None))
        for k in stages:
            spont = torch.stack([sh[k]["spont"] for sh in shows])
            e = torch.stack([torch.stack(sh[k]["early"]) for sh in shows])
            l = torch.stack([torch.stack(sh[k]["late"]) for sh in shows])
            x = l.reshape(len(shows), -1)
            d = [x[p][y[p] == 0].mean(0) - x[p][y[p] == 1].mean(0) for p in parts]
            xs, ys, xh, yh = x[parts[0]], y[parts[0]], x[parts[1]], y[parts[1]]
            cents = torch.stack([xs[ys == 0].mean(0), xs[ys == 1].mean(0)])
            def cc(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                u = u - u.mean(-1, keepdim=True)
                v = v - v.mean(-1, keepdim=True)
                return (u * v).sum(-1) / (u.norm(dim=-1) * v.norm(dim=-1)).clamp_min(1e-9)
            guess = torch.stack([cc(xh, cents[0]), cc(xh, cents[1])], 1).argmax(1)
            acc = float((guess == yh).float().mean())
            sp, ev_e, ev_l = float(spont.mean()), float(e.mean()), float(l.mean())
            print(f"  {c:22s} {k:11s} {sp:5.2f} {ev_e:6.2f} {ev_l:7.2f} {ev_l / max(sp, 1e-9):10.2f} "
                  f"{corr(d[0], d[1]):+12.3f} {acc:9.2f}", flush=True)


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
