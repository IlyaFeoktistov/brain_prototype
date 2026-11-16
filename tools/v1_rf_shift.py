"""
Поля клеток V1 у ямки и их терпимость к сдвигу буквы (DESIGN 2026-11-16,
после letters_similarity: на ±3 px вся популяция V1 теряет букву — r той же
буквы = r разных). Вопрос: нет ли среди клеток V1 подгруппы «сложных»,
которые при сдвиге сохраняют избирательность к букве (средняя корреляция
популяции может её скрывать). Без учёбы и без правки анатомии.

Ориентиры (~24.5 px/°, cfg 0.75° ≈ 18.4 px; сравнение, не норматив):
радиус классического поля V1 макаки на 3–7° — 0.36 ± 0.13° (0.11–0.82°,
79 клеток), V2 — 0.74° (Shushruth et al. 2009, https://doi.org/10.1152/jn.00512.2009;
парафовеа, диаметр ~0.72° ≈ 18 px, у ямки меньше); у сложных клеток поля
больше, чем у простых, ответ на стимул в разных местах поля (Hubel &
Wiesel 1962, кошка, https://doi.org/10.1113/jphysiol.1962.sp006837).
Не поле клетки: 0.2° у ямки (Tehovnik & Slocum 2007) — корковое
представительство участка поля; pRF человека (Dumoulin & Wandell 2008,
https://doi.org/10.1016/j.neuroimage.2007.09.034) — σ поля популяции вокселя.

1. Поле: черта «|» по местам −bar..+bar px от ямки (шаг 1 px, n_bar
   показов); клетки V1 L4 E и L2/3 E с местом в ±core px от ямки. Профиль
   ответа (минус частота на паузе) против (место черты − место клетки),
   средний по отзывчивым клеткам (выравнивание по анатомическому месту —
   без смещения к шумовому пику); ширина на полувысоте профиля и по
   клеткам (только с надёжным пиком). Ширина черты со штрихом и сетчаткой
   (центр σ 0.7 px) входит в измерение.
2. Сдвиг: буквы letters= на сдвигах 0, ±1, ±2, ±3 px (без дрожи), n
   показов. По клетке: избирательность у ямки (F по буквам; порог — 95%
   F при перемешанных метках), r настройки по буквам ямка ↔ сдвиг d
   (потолок — r двух половин показов у ямки), сохранённая частота.
3. Подгруппа: клетки отбираются по первой половине показов (избирательна
   у ямки и r ямка ↔ ±3 px > sel_r), проверка на второй — перенос по
   центрам ямки на ±3 px; рядом — вся популяция и случайная подгруппа того
   же размера.

    PYTHONPATH=. .venv/bin/python tools/v1_rf_shift.py [letters=бгжк] [n=6] [n_bar=3] [bar=10] [core=4] [sel_r=0.5] [seed=1]
"""

from __future__ import annotations

import sys
import time

import torch

from brain.session import Session
from tools.reading_bench import build
from tools.word_read import LETTERS  # noqa: F401  (тот же набор букв, что в опытах)


def corr_rows(x: torch.Tensor) -> torch.Tensor:
    x = x - x.mean(1, keepdim=True)
    x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-9)
    return x @ x.T


def rowwise_r(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """r по последней оси для каждой строки a, b [cells, k]; nan, если нет разброса."""

    a = a - a.mean(-1, keepdim=True)
    b = b - b.mean(-1, keepdim=True)
    den = a.norm(dim=-1) * b.norm(dim=-1)
    return torch.where(den > 0, (a * b).sum(-1) / den.clamp_min(1e-12), torch.nan)


def f_stat(x: torch.Tensor, y: torch.Tensor, L: int) -> torch.Tensor:
    """F по буквам для каждой клетки: x [trials, cells]."""

    mu = x.mean(0)
    cm = torch.stack([x[y == q].mean(0) for q in range(L)])
    nq = torch.tensor([int((y == q).sum()) for q in range(L)], dtype=x.dtype)
    between = (nq[:, None] * (cm - mu) ** 2).sum(0) / (L - 1)
    within = sum(((x[y == q] - cm[q]) ** 2).sum(0) for q in range(L)) / (len(y) - L)
    return between / within.clamp_min(1e-9)


def main(opts: dict) -> None:
    seed = int(opts.get("seed", 1))
    opts = {"fit": "lgn-a", "homeo": f"lgn-d-pop-tcind-s{seed}", "relay_v_t_mv": "-59", "trn_scale": "0.1", **opts}
    mind, b = build(opts)
    s = Session(mind, window_ms=10.0)
    shown = opts.get("letters", "бгжк")
    L = len(shown)
    n, n_bar, bar = int(opts.get("n", 6)), int(opts.get("n_bar", 3)), int(opts.get("bar", 10))
    core, sel_r = float(opts.get("core", 4)), float(opts.get("sel_r", 0.5))
    show_ms, rest_ms = 350.0, 150.0
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    fov = b.cfg.fovea_col
    pops = {}
    for name, key in (("V1 L4", "L4"), ("V1 L2/3", "L23")):
        pop = b.areas["visual"].layers[key]["E"]
        cols = b.field_cols[pop.name].float().cpu()
        sel = ((cols - fov).abs() <= core).nonzero().flatten()
        pops[name] = (sel + pop.start, cols[sel])
    print(f"клетки с местом в ±{core:.0f} px от ямки: " + ", ".join(f"{k} {len(v[0])}" for k, v in pops.items())
          + f"; черта ±{bar} px × {n_bar}, буквы «{shown}» 0/±1/±2/±3 px × {n}", flush=True)

    s.line, s.gaze = "", 0.0
    b.run(2500.0, s._blank(2500.0))
    t0 = time.time()
    rest_acc: list[torch.Tensor] = []

    def show(text: str, place_off: float) -> torch.Tensor:
        """text центром на месте fov + place_off px; частоты всех клеток, Гц."""

        s.line = text
        s.gaze = s.screen.width(text) / 2 - place_off
        r = b.observe(show_ms, s._view(show_ms)).float().cpu()
        s.line = ""
        rest_acc.append(b.observe(rest_ms, s._blank(rest_ms)).float().cpu())
        return r

    # 1. черта
    bpos = torch.arange(-bar, bar + 1).float().repeat(n_bar)
    bpos = bpos[torch.randperm(len(bpos), generator=gen)]
    B = []
    for j, p in enumerate(bpos.tolist()):
        B.append(show("|", p))
        if j == 3:
            print(f"  первые 4 показа: {time.time() - t0:.0f} с", flush=True)
    B = torch.stack(B)
    # 2. буквы
    shifts = [0, 1, -1, 2, -2, 3, -3]
    order = torch.arange(L).repeat(n * len(shifts))
    where = torch.tensor(shifts).repeat_interleave(L).repeat(n)
    perm = torch.randperm(len(order), generator=gen)
    order, where = order[perm], where[perm]
    X = torch.stack([show(shown[i], float(d)) for i, d in zip(order.tolist(), where.tolist())])
    base_all = torch.stack(rest_acc).mean(0)
    print(f"показы: {time.time() - t0:.0f} с\n", flush=True)
    # номер показа в своём условии — для деления на половины
    rep = torch.zeros(len(order), dtype=torch.long)
    seen: dict = {}
    for t, key in enumerate(zip(order.tolist(), where.tolist())):
        rep[t] = seen.get(key, 0)
        seen[key] = rep[t] + 1
    first = rep < n // 2

    for name, (idx, cols) in pops.items():
        base = base_all[idx]
        # --- 1. поле
        prof = torch.stack([B[bpos == p][:, idx].mean(0) for p in range(-bar, bar + 1)], 1) - base[:, None]  # [cells, pos]
        sd = torch.stack([B[bpos == p][:, idx].std(0) for p in range(-bar, bar + 1)], 1)
        peak, at = prof.max(1)
        reliable = peak > 3 * (sd.gather(1, at[:, None]).squeeze(1) / n_bar ** 0.5).clamp_min(1e-9)
        reliable &= peak > 2.0
        rel = torch.arange(-bar, bar + 1).float()[None, :] - (cols - fov)[:, None]  # место черты − место клетки
        bins = torch.arange(-bar, bar + 1).float()
        avg = torch.stack([prof[reliable][(rel[reliable] - c).abs() <= 0.5].mean() if reliable.any() else torch.tensor(0.0)
                           for c in bins])
        half = avg.max() / 2
        above = (avg >= half).nonzero().flatten()
        fw_pop = float(bins[above].max() - bins[above].min() + 1) if len(above) else float("nan")
        widths = ((prof[reliable] >= peak[reliable, None] / 2).sum(1)).float()
        offc = (bins[at[reliable]] - (cols[reliable] - fov))
        q = lambda v, a: float(v.quantile(a)) if len(v) else float("nan")  # noqa: E731
        print(f"{name}: клеток {len(idx)}, пауза {float(base.mean()):.2f} Гц; надёжный пик на черту у {int(reliable.sum())} "
              f"({float(reliable.float().mean()):.0%})", flush=True)
        print(f"  поле на полувысоте: средний профиль {fw_pop:.0f} px; по клеткам медиана {q(widths, 0.5):.0f} "
              f"[10% {q(widths, 0.1):.0f}, 90% {q(widths, 0.9):.0f}] px; пик − место клетки "
              f"{float(offc.mean()) if len(offc) else float('nan'):+.1f} ± {float(offc.std()) if len(offc) > 1 else float('nan'):.1f} px", flush=True)
        print("  средний профиль (Гц сверх паузы), −" + str(bar) + "…+" + str(bar) + " px: "
              + " ".join(f"{float(v):.1f}" for v in avg), flush=True)
        # --- 2. сдвиг по клеткам
        x = X[:, idx]
        m0 = where == 0
        F = f_stat(x[m0], order[m0], L)
        Fnull = torch.stack([f_stat(x[m0], order[m0][torch.randperm(int(m0.sum()), generator=gen)], L) for _ in range(20)])
        selective = F > Fnull.flatten().quantile(0.95)
        tune = lambda m: torch.stack([x[m & (order == q_)].mean(0) for q_ in range(L)], 1)  # noqa: E731  [cells, L]
        t0_ = tune(m0)
        ceil = rowwise_r(tune(m0 & first), tune(m0 & ~first))
        print(f"  избирательны к букве у ямки: {int(selective.sum())} ({float(selective.float().mean()):.0%}; "
              f"случайно 5%); у них r половин у ямки (потолок) {float(ceil[selective].nanmean()):.2f}", flush=True)
        line = "  r настройки ямка ↔ сдвиг у избирательных:"
        for d in (1, 2, 3):
            r = torch.stack([rowwise_r(t0_, tune(where == sd_)) for sd_ in (d, -d)]).nanmean(0)
            keep = torch.stack([tune(where == sd_).mean(1) for sd_ in (d, -d)]).mean(0) / t0_.mean(1).clamp_min(1e-9)
            rs = r[selective]
            line += (f" ±{d} px {float(rs.nanmean()):.2f} (>0.5 у {float((rs > 0.5).float().mean()):.0%}), "
                     f"частота {float(keep[selective].median()):.0%};")
        print(line, flush=True)
        # отвечают ли на букву у ямки и на ±3 px (частота сверх паузы > 2 Гц на своей лучшей букве)
        resp = lambda m: (tune(m).max(1).values - base) > 2.0  # noqa: E731
        r0, r3 = resp(m0), resp(where == 3) | resp(where == -3)
        both = r0 & r3
        print(f"  отвечают на букву (> паузы + 2 Гц): у ямки {float(r0.float().mean()):.0%}, на ±3 px {float(r3.float().mean()):.0%}, "
              f"на обоих {float(both.float().mean()):.0%}; из них избирательны у ямки {int((both & selective).sum())}", flush=True)
        # --- 3. подгруппа по первой половине, проверка на второй
        x1, o1, w1 = x[first], order[first], where[first]
        F1 = f_stat(x1[w1 == 0], o1[w1 == 0], L)
        F1null = torch.stack([f_stat(x1[w1 == 0], o1[w1 == 0][torch.randperm(int((w1 == 0).sum()), generator=gen)], L)
                              for _ in range(20)])
        t1 = lambda m: torch.stack([x1[m & (o1 == q_)].mean(0) for q_ in range(L)], 1)  # noqa: E731
        r1 = torch.stack([rowwise_r(t1(w1 == 0), t1(w1 == d)) for d in (3, -3)]).nanmean(0)
        sub = (F1 > F1null.flatten().quantile(0.95)) & (r1 > sel_r)
        x2, o2, w2 = x[~first], order[~first], where[~first]

        def transfer(cells: torch.Tensor) -> tuple[float, float]:
            if len(cells) == 0:
                return float("nan"), float("nan")
            c0 = torch.stack([x2[(w2 == 0) & (o2 == q_)][:, cells].mean(0) for q_ in range(L)])
            ms = (w2 == 3) | (w2 == -3)
            xs = x2[ms][:, cells]
            pred = corr_rows(torch.cat([c0, xs]))[L:, :L].argmax(1)
            acc = float((pred == o2[ms]).float().mean())
            null = torch.tensor([float((pred == o2[ms][torch.randperm(int(ms.sum()), generator=gen)]).float().mean())
                                 for _ in range(200)])
            return acc, float(null.quantile(0.95))

        allc = torch.arange(len(idx))
        rnd = allc[torch.randperm(len(idx), generator=gen)[: max(1, int(sub.sum()))]]
        a_s, n_s = transfer(sub.nonzero().flatten())
        a_a, n_a = transfer(allc)
        a_r, n_r = transfer(rnd)
        print(f"  подгруппа (избирательна и r ямка↔±3 > {sel_r} по 1-й половине): {int(sub.sum())} клеток; перенос ±3 px "
              f"на 2-й половине {a_s:.2f} [95% {n_s:.2f}]; вся популяция {a_a:.2f} [{n_a:.2f}]; "
              f"случайные {len(rnd)} — {a_r:.2f} [{n_r:.2f}] (случай {1 / L:.2f})\n", flush=True)
    print(f"всего {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
