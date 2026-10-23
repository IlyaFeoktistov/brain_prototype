"""
Есть ли в зоне букв исходная избирательность к фразам (до любой учёбы).

Li & DiCarlo 2008 (https://doi.org/10.1126/science.1160028): обучение по
смежности через саккады меняло избирательность клеток IT, у которых она
была (P ~20, N ~14 спайк/с), а клетки без исходной избирательности (P−N < 1
спайк/с) не менялись. Прежде чем пробовать такое правило — есть ли здесь
что перестраивать.

Мозг как в reading_bench, без пластичности. A и B вперемешку по n показов
чтением. Показы делятся пополам (по порядку): на отборочной половине у
каждой клетки t Уэлча A против B (спайки за показ, 3 фиксации), отбор
|t| > t_sel; на отложенной — сохраняется ли знак предпочтения у отобранных,
их разность и отношение к разбросу повторов. Контроль — тот же отбор с
перемешанными метками отборочной половины (сколько «избирательных» даёт
шум). Ещё: частота лучших 1% клеток (по ответу на отборочной), точность
популяции (ближайший центр по отборочной половине → отложенная).
Ступени: V1 L2/3 (сравнение), буквы L4, буквы L2/3.

    PYTHONPATH=. python tools/letters_select.py [n=30] [t_sel=2.6] [seed=1]
"""

from __future__ import annotations

import sys

import torch

from brain.session import Session
from tools.reading_bench import build

WIN = 10.0


def welch(x: torch.Tensor, y: torch.Tensor, lab: torch.Tensor) -> torch.Tensor:
    a, b = x[lab == 0], x[lab == 1]
    se = (a.var(0) / len(a) + b.var(0) / len(b)).sqrt().clamp_min(1e-9)
    return (a.mean(0) - b.mean(0)) / se


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
    vis, let = b.areas["visual"].layers, b.areas["letters"].layers
    stages = {"V1 L2/3": vis["L23"]["E"], "буквы L4": let["L4"]["E"], "буквы L2/3": let["L23"]["E"]}

    s.line, s.gaze = "", 0.0
    b.run(2500.0)

    def gazes(line: str) -> list[float]:
        width = s.screen.width(line) / max(1, len(line))
        n = max(1, int((len(line) + step - 1) // step))
        return [i * step * width for i in range(n)][:3]

    def show(lab: int) -> dict[str, torch.Tensor]:
        s.line = ""
        b.run(500.0, s._blank(500.0))
        line = texts[lab] + " "
        acc = {k: 0 for k in stages}
        for i, g in enumerate(gazes(line)):
            if i:
                s.line, s.gaze = line, g
                b.run(cfg.saccade_ms, s._blank(cfg.saccade_ms))
            s.line, s.gaze = line, g
            for _ in range(int(fix // WIN)):
                r = b.observe(WIN, s._view(WIN)) * (WIN / 1000.0)
                for k, pop in stages.items():
                    acc[k] = acc[k] + r[pop.slice].cpu()
        return acc

    order = [0, 1] * n_rep
    order = [order[i] for i in torch.randperm(len(order), generator=gen)]
    feats = []
    for j, lab in enumerate(order):
        feats.append(show(lab))
        if (j + 1) % 20 == 0:
            print(f"  показов {j + 1}/{len(order)}", flush=True)
    y = torch.tensor(order)
    half = len(order) // 2
    sel, held = slice(0, half), slice(half, None)
    seconds = 3 * fix / 1000.0

    print(f"\n=== Исходная избирательность: «{texts[0]}» против «{texts[1]}», по {n_rep} показов, "
          f"отбор на {half} показах (|t| > {t_sel:g}), проверка на {len(order) - half}")
    for k in stages:
        x = torch.stack([f[k] for f in feats]) / seconds          # Гц за показ
        ys, yh = y[sel], y[held]
        xs, xh = x[sel], x[held]
        t = welch(xs, None, ys)
        picked = t.abs() > t_sel
        shuf_t = welch(xs, None, ys[torch.randperm(len(ys), generator=gen)])
        n_shuf = int((shuf_t.abs() > t_sel).sum())
        ha, hb = xh[yh == 0].mean(0), xh[yh == 1].mean(0)
        sd = torch.cat([xh[yh == 0] - ha, xh[yh == 1] - hb]).std(0)
        keep = (torch.sign(ha - hb) == torch.sign(t)) & picked
        diff_h = ((ha - hb) * torch.sign(t))[picked]
        snr_h = (diff_h / sd[picked].clamp_min(1e-9)) if picked.any() else torch.zeros(1)
        # Лучшие 1% по ответу (среднее A и B) на отборочной половине.
        resp = xs.mean(0)
        top = resp.topk(max(1, len(resp) // 100)).indices
        pa, pb = xh[yh == 0][:, top].mean(0), xh[yh == 1][:, top].mean(0)
        pref, non = torch.maximum(pa, pb).mean(), torch.minimum(pa, pb).mean()
        # Популяция: центры классов по отборочной половине → отложенная.
        def corr(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            u = u - u.mean(-1, keepdim=True)
            v = v - v.mean(-1, keepdim=True)
            return (u * v).sum(-1) / (u.norm(dim=-1) * v.norm(dim=-1)).clamp_min(1e-9)
        cents = torch.stack([xs[ys == 0].mean(0), xs[ys == 1].mean(0)])
        guess = torch.stack([corr(xh, cents[0]), corr(xh, cents[1])], 1).argmax(1)
        acc = float((guess == yh).float().mean())
        print(f"\n  {k} ({x.shape[1]} клеток): частота A {float(x[y == 0].mean()):.2f} / B {float(x[y == 1].mean()):.2f} Гц, "
              f"точность популяции на отложенных {acc:.2f}")
        print(f"    отобрано {int(picked.sum())} ({float(picked.float().mean()) * 100:.1f}%), "
              f"по шуму (перемешанные метки) {n_shuf} ({n_shuf / x.shape[1] * 100:.1f}%)")
        if picked.any():
            print(f"    у отобранных знак сохранился на отложенных: {float(keep.sum() / picked.sum()):.2f} "
                  f"(случайно 0.50); разность пред − непред {float(diff_h.mean()):+.2f} Гц, "
                  f"к разбросу повторов {float(snr_h.mean()):+.2f}; "
                  f"воспроизводимо избирательных {int(keep.sum())} ({float(keep.float().mean()) * 100:.2f}%)")
        print(f"    лучшие 1% ({len(top)} клеток) на отложенных: предпочитаемая {float(pref):.1f} Гц, "
              f"другая {float(non):.1f} Гц", flush=True)


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
