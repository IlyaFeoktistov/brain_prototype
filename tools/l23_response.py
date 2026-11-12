"""
Почему V1 L2/3 E слабо передаёт букву (замер, без вмешательства; мозг как в
letters_dc: LGN A, гомеостаз lgn-d, без пластичности). Буква занимает малую
часть поля V1 (экран 16 × 56, буква ~6 × 10 px), поэтому доли считаются по
месту поля клетки: «на букве» — в квадрате 3 × 3 px вокруг места клетки
есть чернила (на этом показе), «рядом» — чернила в 7 × 7, но не в 3 × 3,
«вне» — прочие. Отрезки показа: 0–50 мс (переход) и 50–350 мс
(устоявшийся), пауза — 150 мс перед показом.

Печатает по V1 L4 E и L2/3 E:
  1. частоты по местам и отрезкам; доля клеток с ростом / падением (парный
     t показ − пауза по показам, где клетка на букве, |t| > 2); доля
     избирательных к букве (F по буквам на отрезке 50–350) против той же меры
     с перемешанными метками;
  2. различение букв (классификатор по корреляции, как letters_dc) по
     отрезкам — все клетки и только клетки у буквы (на букве или рядом хоть
     на одном показе);
  3. у клеток L2/3 E на букве — источники: вход каждого пути (Σ вес ×
     частота источника, мВ·Гц), токи возбуждения и торможения сомы с
     движущей силой, мембрана; пауза → 0–50 → 50–350.

    PYTHONPATH=. .venv/bin/python tools/l23_response.py [letters=абвгдежзик] [label=5] [test=5] [seed=1]
"""

from __future__ import annotations

import sys
import time

import torch
import torch.nn.functional as F

from brain.session import Session
from tools.reading_bench import build

WIN = 10.0
PARTS = ("пауза", "0–50", "50–350")


def main(opts: dict) -> None:
    seed = int(opts.get("seed", 1))
    opts = {"fit": "lgn-a", "homeo": f"lgn-d-pop-s{seed}", "relay_v_t_mv": "-59", "trn_scale": "0.1", **opts}
    mind, b = build(opts)
    cfg = b.cfg
    s = Session(mind, window_ms=WIN)
    letters = opts.get("letters", "абвгдежзик")
    n_label, n_test = int(opts.get("label", 5)), int(opts.get("test", 5))
    show_ms, rest_ms, early_ms = float(opts.get("show", 350)), float(opts.get("rest", 150)), 50.0
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    vis = b.areas["visual"].layers
    pops = {"L4": vis["L4"]["E"], "L2/3": vis["L23"]["E"]}
    post = pops["L2/3"]
    paths = [p for p in b.projections if p.post is post and p.source == "net"]
    place = {k: b.map_pos[p.name].round().long().cpu() for k, p in pops.items()}

    def ink_at(picture: torch.Tensor, k: str, size: int) -> torch.Tensor:
        ink = F.max_pool2d((picture < 0.5).float()[None, None], size, 1, size // 2)[0, 0]
        r = place[k][:, 0].clamp(0, ink.shape[0] - 1)
        c = place[k][:, 1].clamp(0, ink.shape[1] - 1)
        return ink[r, c] > 0

    def window(view: torch.Tensor) -> tuple[dict[str, torch.Tensor], list[torch.Tensor]]:
        r = b.observe(WIN, view)
        v = b.v[post.slice]
        excite = ((cfg.ampa_reversal_mv - v) / (cfg.ampa_reversal_mv - cfg.epsp_holding_mv)).clamp(min=0.0)
        force = ((v - cfg.gaba_reversal_mv) / (cfg.ipsp_holding_mv - cfg.gaba_reversal_mv)).clamp(min=0.0)
        e = (b.i_syn[post.slice] + b.dc_mv[post.slice] if torch.is_tensor(b.dc_mv) else b.i_syn[post.slice]) * excite
        i = (b.i_inh[post.slice] + b.i_ngf[post.slice]) * force
        src = [(p.weight.float() * r[p.pre.long()]).sum(1) for p in paths] + [e, i, e + i, v.clone()]
        return {k: r[p.slice] for k, p in pops.items()}, src

    def segment(view, n: int) -> tuple[dict[str, torch.Tensor], list[torch.Tensor]]:
        rate, src = None, None
        for _ in range(n):
            r, x = window(view(WIN))
            rate = r if rate is None else {k: rate[k] + r[k] for k in r}
            src = x if src is None else [a + c for a, c in zip(src, x)]
        return {k: (v / n).cpu() for k, v in rate.items()}, [(x / n).cpu() for x in src]

    def show(ch: str) -> dict:
        s.line = ""
        out = {}
        out["пауза"] = segment(s._blank, int(rest_ms // WIN))
        s.line, s.gaze = ch, s.screen.width(ch) / 2 + float(torch.randint(-1, 2, (1,), generator=gen))
        picture = s._picture(s.screen, cfg.fovea_col)
        out["0–50"] = segment(s._view, int(early_ms // WIN))
        out["50–350"] = segment(s._view, int((show_ms - early_ms) // WIN))
        s.line = ""
        out["на"] = {k: ink_at(picture, k, 3) for k in pops}
        out["рядом"] = {k: ink_at(picture, k, 7) & ~out["на"][k] for k in pops}
        return out

    s.line, s.gaze = "", 0.0
    b.run(2500.0, s._blank(2500.0))
    t0 = time.time()
    shows, labels = [], []
    for rep in range(n_label + n_test):
        for k in torch.randperm(len(letters), generator=gen).tolist():
            shows.append(show(letters[k]))
            labels.append(k)
        print(f"  проход {rep + 1}/{n_label + n_test}, {time.time() - t0:.0f} с", flush=True)

    y = torch.tensor(labels)
    lab = torch.arange(len(y)) < n_label * len(letters)
    z = lambda x: (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True).clamp_min(1e-9)
    n_rep = n_label + n_test

    def selective(X: torch.Tensor, yy: torch.Tensor) -> torch.Tensor:
        means = torch.stack([X[yy == k].mean(0) for k in range(len(letters))])
        within = torch.stack([X[yy == k].var(0) for k in range(len(letters))]).mean(0)
        return means.var(0) > 2.0 * within / n_rep

    for k, pop in pops.items():
        R = {part: torch.stack([sh[part][0][k] for sh in shows]) for part in PARTS}
        on = torch.stack([sh["на"][k] for sh in shows])
        near = torch.stack([sh["рядом"][k] for sh in shows])
        off = ~(on | near)
        print(f"\nV1 {k} E ({pop.size} клеток): на букве в среднем {float(on.float().sum(1).mean()):.0f} клеток за показ, "
              f"рядом {float(near.float().sum(1).mean()):.0f}", flush=True)
        for name, m in (("на букве", on), ("рядом", near), ("вне", off)):
            rates = " → ".join(f"{float(R[part][m].mean()):.2f}" for part in PARTS)
            line = f"  {name:9s} частота {rates} Гц"
            # Рост / падение: у клетки — показы, где она в этом месте.
            for part in ("0–50", "50–350"):
                d = (R[part] - R["пауза"]).masked_fill(~m, float("nan"))
                cnt = (~d.isnan()).sum(0)
                mu = d.nanmean(0)
                sd = ((d - mu).pow(2).nansum(0) / (cnt - 1).clamp_min(1)).sqrt()
                t = mu / (sd / cnt.clamp_min(1).float().sqrt()).clamp_min(1e-9)
                ok = cnt >= 5
                line += (f"; {part}: вверх {float((t[ok] > 2).float().mean()):.0%}, "
                         f"вниз {float((t[ok] < -2).float().mean()):.0%} (клеток {int(ok.sum())})")
            print(line, flush=True)
        near_any = (on | near).any(0)
        late = R["50–350"]
        sel = selective(late, y)
        sel0 = selective(late, y[torch.randperm(len(y), generator=gen)])
        print(f"  избирательных к букве (50–350): у буквы {float(sel[near_any].float().mean()):.0%} "
              f"(перемешано {float(sel0[near_any].float().mean()):.0%}), вне {float(sel[~near_any].float().mean()):.0%} "
              f"(перемешано {float(sel0[~near_any].float().mean()):.0%})", flush=True)
        line = "  различение (случайно {:.2f}):".format(1 / len(letters))
        for part in ("0–50", "50–350"):
            for name, m in (("все", torch.ones_like(near_any)), ("у буквы", near_any)):
                X = R[part][:, m]
                mean = torch.stack([X[lab][y[lab] == j].mean(0) for j in range(len(letters))])
                acc = float(((z(X[~lab]) @ z(mean).T).argmax(1) == y[~lab]).float().mean())
                line += f" {part} {name} {acc:.2f};"
        print(line, flush=True)

    # Источники у L2/3 E на букве.
    on = torch.stack([sh["на"]["L2/3"] for sh in shows])
    names = [f"вход: {p.name}" for p in paths] + ["ток возбуждения", "ток торможения", "сумма токов", "мембрана"]
    print("\nV1 L2/3 E на букве — источники (мВ·Гц для входов, мВ для токов): пауза → 0–50 → 50–350; вне буквы 50–350")
    for j, name in enumerate(names):
        vals = [float(torch.stack([sh[part][1][j] for sh in shows])[on].mean()) for part in PARTS]
        out_ = float(torch.stack([sh["50–350"][1][j] for sh in shows])[~on].mean())
        print(f"  {name:44s} " + " → ".join(f"{v:9.2f}" for v in vals) + f"   | {out_:9.2f}", flush=True)
    print(f"всего {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
