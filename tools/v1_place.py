"""
Различает ли V1 L2/3 буквы в разных местах строки (DESIGN 2026-11-14,
странность «10 px не учится»: узкая полоса учителя у 28 px — 0.32, у 46 —
0.22, у 10 — на уровне случая).

1. Строение пути V1 → буквы у каждого места: клетки букв L4 E полосы
   (|место клетки − место| ≤ band), сколько у них входов, где места их
   входов V1 L2/3 (среднее и разброс относительно места), сколько
   различных клеток V1 на полосу. Плотность клеток V1 L2/3 у места.
2. Различимость в самой V1: буквы по местам (±1 px, как в letters_field),
   признаки — число спайков клеток V1 L2/3 E в окне ±win px вокруг места
   (win 6 — половина окна пути V1 → буквы 12 px) за показ; классификатор —
   ближайший центр по корреляции с выбросом одного показа (как l4_decode);
   случайно — перемешанные метки. Рядом — частота этих клеток на букве и
   на паузе.

Мозг как в letters_field (без учёбы, веса пути исходные — на V1 не влияют).

    PYTHONPATH=. .venv/bin/python tools/v1_place.py [n=6] [seed=1] [places=10,28,46] [win=6] [band=1.5]
"""

from __future__ import annotations

import sys
import time

import torch

from brain.session import Session
from tools.reading_bench import build


def main(opts: dict) -> None:
    seed = int(opts.get("seed", 1))
    opts = {"fit": "lgn-a", "homeo": f"lgn-d-pop-tcind-s{seed}", "relay_v_t_mv": "-59", "trn_scale": "0.1",
            "us": "visual -> letters", **opts}
    mind, b = build(opts)
    s = Session(mind, window_ms=10.0)
    letters = opts.get("letters", "абвгдежзик")
    L = len(letters)
    n = int(opts.get("n", 6))
    win, band = float(opts.get("win", 6.0)), float(opts.get("band", 1.5))
    places = [float(x) for x in opts.get("places", "10,28,46").split(",")]
    show_ms, rest_ms = 350.0, 150.0
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    dev = b.device
    path = b.us[0]
    P = path.post
    v1 = b.areas["visual"].layers["L23"]["E"]                   # источник пути (как letters_us)
    vcols = b.field_cols[v1.name].to(dev)
    pcols = b.field_cols[P.name].to(dev)

    print(f"V1 L2/3 E {v1.size} клеток, места: мин {float(vcols.min()):.1f} макс {float(vcols.max()):.1f} px; "
          f"буквы L4 E {P.size}, входов на клетку {path.pre.shape[1]}", flush=True)
    hist = torch.histc(vcols.float().cpu(), bins=14, min=0, max=b.field_width)
    print("  клеток V1 L2/3 по 4 px: " + " ".join(f"{int(h)}" for h in hist), flush=True)
    print("\n1. путь V1 → буквы у места (клетки полосы ±%.1f px):" % band)
    for c in places:
        cells = ((pcols - c).abs() <= band).nonzero().flatten()
        pre = path.pre[cells].long() - v1.start
        okp = (pre >= 0) & (pre < v1.size)
        pc = vcols[pre.clamp(0, v1.size - 1)][okp]
        uniq = pre[okp].unique().numel()
        w = path.weight[cells].float()
        print(f"  {c:.0f} px: клеток {len(cells)}, места клеток {float(pcols[cells].min()):.1f}…{float(pcols[cells].max()):.1f}; "
              f"входов из V1 {float(okp.float().mean()):.0%}, места входов − место: среднее {float((pc - c).mean()):+.1f}, "
              f"от {float((pc - c).min()):+.1f} до {float((pc - c).max()):+.1f} px; различных клеток V1 {uniq}; "
              f"клеток V1 в ±{win:.0f} px {int(((vcols - c).abs() <= win).sum())}; вес средний {float(w.mean()):.3g}", flush=True)

    s.line, s.gaze = "", 0.0
    b.run(2500.0, s._blank(2500.0))
    t0 = time.time()
    order = torch.arange(L).repeat(n * len(places))
    where = torch.tensor(places).repeat_interleave(L).repeat(n)
    perm = torch.randperm(len(order), generator=gen)
    order, where = order[perm], where[perm]
    jit = torch.randint(-1, 2, (len(order),), generator=gen).float()
    X: list[torch.Tensor] = []
    rest: list[torch.Tensor] = []
    for k, c, j in zip(order.tolist(), where.tolist(), jit.tolist()):
        s.line = letters[k]
        s.gaze = s.screen.width(letters[k]) / 2 + (b.cfg.fovea_col - (c + j))
        X.append(b.observe(show_ms, s._view(show_ms))[v1.slice].float().cpu())
        s.line = ""
        rest.append(b.observe(rest_ms, s._blank(rest_ms))[v1.slice].float().cpu())
    X_all, R_all = torch.stack(X), torch.stack(rest)
    print(f"\n2. различимость в V1 L2/3 ({n} показов каждой буквы в каждом месте, {time.time() - t0:.0f} с; "
          f"случайно {1 / L:.2f}):", flush=True)

    def corr_rows(x: torch.Tensor) -> torch.Tensor:
        x = x - x.mean(1, keepdim=True)
        x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-9)
        return x @ x.T

    def loo(x: torch.Tensor, y: torch.Tensor) -> float:
        hits = 0
        for i in range(len(y)):
            keep = torch.arange(len(y)) != i
            cents = torch.stack([x[keep & (y == q)].mean(0) for q in range(L)])
            hits += int(int(corr_rows(torch.cat([x[i:i + 1], cents]))[0, 1:].argmax()) == int(y[i]))
        return hits / len(y)

    vc = vcols.cpu()
    for c in places:
        m = where == c
        sel = (vc - c).abs() <= win
        x, y = X_all[m][:, sel], order[m]
        acc = loo(x, y)
        null = torch.tensor([loo(x, y[torch.randperm(len(y), generator=gen)]) for _ in range(20)])
        active = float((x.sum(0) > 0).float().mean())
        print(f"  {c:.0f} px: клеток {int(sel.sum())}, точность {acc:.2f} (перемешано: среднее {float(null.mean()):.2f}, "
              f"макс {float(null.max()):.2f}); частота на букве {float(x.mean()):.2f} Гц, на паузе "
              f"{float(R_all[m][:, sel].mean()):.2f} Гц; хоть раз стреляли {active:.0%}", flush=True)
    print(f"всего {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
