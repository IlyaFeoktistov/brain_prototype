"""
Почему рука печатает на пустом экране (DESIGN 2026-11-16 «Аудит выхода»:
~75 знаков за 5 с при GPi 34 Гц). Без правки мозга и порога считывания.

Мозг как в reading_bench, пустой экран, окна по 10 мс (как Session.live).
1. Нажатия: условие Hand.pressed (блок клавиши сильнее всех и ≥ половины
   из 8 клеток за окно) — сколько раз за секунду, сколько разных клавиш,
   какие чаще; нажатий с правилом отпускания (как печатает Session).
2. Случайность или залп: ожидаемое число окон с ≥ 4 из 8 клеток блока при
   независимых клетках с их собственной частотой (биномиальное по клеткам,
   точный счёт) против наблюдаемого; доля всех L5 E руки, выстреливших в
   окне нажатия, против обычного окна (узкий залп блока или общий).
3. Вход L5 E руки и таламуса VA по путям: мера поставки Σ вес × частота
   источника (кратковременная пластичность синапсов не учтена — мера, не
   ток; знак — знак синапса), среднее поле AMPA, фон bias; в среднем и в
   окне перед нажатием (что растёт перед ложным нажатием).

    PYTHONPATH=. .venv/bin/python tools/hand_rest.py [sec=20] [seed=1]
"""

from __future__ import annotations

import sys
import time

import torch

from brain.session import Session
from tools.reading_bench import build

WIN = 10.0


def at_least(p: torch.Tensor, k: int) -> torch.Tensor:
    """P(≥ k успехов) для независимых испытаний с вероятностями p [..., n]."""

    n = p.shape[-1]
    dist = torch.zeros(*p.shape[:-1], n + 1, dtype=torch.float64)
    dist[..., 0] = 1.0
    for i in range(n):
        q = p[..., i:i + 1].double()
        dist = torch.cat([dist[..., :1] * (1 - q), dist[..., 1:] * (1 - q) + dist[..., :-1] * q], -1)
    return dist[..., k:].sum(-1)


def main(opts: dict) -> None:
    seed = int(opts.get("seed", 1))
    opts = {"fit": "lgn-a", "homeo": f"lgn-d-pop-tcind-s{seed}", "relay_v_t_mv": "-59", "trn_scale": "0.1", **opts}
    mind, b = build(opts)
    s = Session(mind, window_ms=WIN)
    sec = float(opts.get("sec", 20))
    torch.manual_seed(seed)
    H = s.H
    keys, cells = len(s.keyboard), s.hand.cells
    targets = {"рука L5 E": H, "рука L2/3 E": b.areas["hand"].layers["L23"]["E"], "таламус VA": b.areas["thalamus"].E}
    paths = {k: [p for p in b.projections if p.post is P and p.source == "net"] for k, P in targets.items()}
    for k, ps in paths.items():
        print(f"{k}: путей {len(ps)}: " + ", ".join(p.name for p in ps), flush=True)

    s.line, s.gaze = "", 0.0
    b.run(2500.0, s._blank(2500.0))
    n_win = int(sec * 1000 / WIN)
    blank = s._blank(WIN)
    fired_all, drive_all = [], []
    t0 = time.time()
    for t in range(n_win):
        r = b.observe(WIN, blank)
        fired_all.append((r[H.slice] > 0).cpu())
        row = []
        for k, P in targets.items():
            for p in paths[k]:
                row.append((p.weight.float() * r[p.pre.long()]).sum(1).mean())
            q = int(b.mf_post_id[P.start])
            row += [(b.mf_exc[q] * b.mf_rate).sum().float(), b.bias[P.slice].float().mean(), r[P.slice].mean()]
        drive_all.append(torch.stack(row).cpu())
        if t == 49:
            print(f"  первые 0.5 с: {time.time() - t0:.0f} с", flush=True)
    F = torch.stack(fired_all)                                  # [win, cells L5]
    D = torch.stack(drive_all)                                  # [win, features]
    print(f"окон {n_win} ({sec:.0f} с), {time.time() - t0:.0f} с\n", flush=True)

    # 1. нажатия
    blk = F[:, : keys * cells].float().view(n_win, keys, cells)
    share = blk.mean(2)
    best = share.argmax(1)
    cond = share.gather(1, best[:, None]).squeeze(1) >= 0.5
    pressed = torch.where(cond, best, torch.full_like(best, -1))
    typed = sum(1 for i in range(n_win) if pressed[i] >= 0 and (i == 0 or pressed[i] != pressed[i - 1]))
    cnt = torch.bincount(pressed[cond], minlength=keys)
    top = cnt.argsort(descending=True)[:8]
    print(f"1. условие нажатия выполнено в {int(cond.sum())} окнах из {n_win} ({float(cond.float().mean()):.1%}), "
          f"{int(cond.sum()) / sec:.1f} в с; напечатано с отпусканием {typed} ({typed / sec:.1f} в с); "
          f"разных клавиш {int((cnt > 0).sum())} из {keys}; чаще: "
          + " ".join(f"«{s.keyboard.type(int(k))}» {int(cnt[k])}" for k in top if cnt[k] > 0), flush=True)
    rate_blk = blk.mean((0, 2)) / (WIN / 1000)
    rest_cells = F[:, keys * cells:].float()
    print(f"   частота клеток блоков {float(rate_blk.mean()):.1f} Гц (по клавишам {float(rate_blk.min()):.1f}…"
          f"{float(rate_blk.max()):.1f}), прочих L5 E {float(rest_cells.mean()) / (WIN / 1000):.1f} Гц", flush=True)

    # 2. случайность или залп
    p_cell = blk.mean(0)                                        # [keys, cells] вероятность спайка за окно
    exp4 = at_least(p_cell, cells // 2)                         # [keys]
    obs4 = (blk.sum(2) >= cells // 2).float().mean(0)
    print(f"2. окон с ≥ {cells // 2} из {cells} клеток блока: наблюдается {float(obs4.sum()) * n_win:.0f}, при независимых клетках "
          f"ожидалось {float(exp4.sum()) * n_win:.1f} (× {float(obs4.sum() / exp4.sum().clamp_min(1e-12)):.0f})", flush=True)
    # контроль: те же 8 клеток, но вразброс по L5 (не соседи), и соседние восьмёрки вне клавиатуры
    gen = torch.Generator().manual_seed(seed)
    def sync(groups: torch.Tensor) -> tuple[float, float]:
        g = F[:, groups].float()                                # [win, G, 8]
        o = float((g.sum(2) >= cells // 2).float().mean(0).sum()) * n_win
        e = float(at_least(g.mean(0), cells // 2).sum()) * n_win
        return o, e
    G = keys
    rnd = torch.randperm(F.shape[1], generator=gen)[: G * cells].view(G, cells)
    o_r, e_r = sync(rnd)
    n_out = (F.shape[1] - keys * cells) // cells
    nb = (keys * cells + torch.arange(min(G, n_out) * cells)).view(-1, cells)
    o_n, e_n = sync(nb)
    print(f"   контроль, {G} восьмёрок: вразброс по L5 — наблюдается {o_r:.0f}, ожидалось {e_r:.1f}; "
          f"соседние вне клавиатуры ({len(nb)}) — наблюдается {o_n:.0f}, ожидалось {e_n:.1f}", flush=True)
    frac = F.float().mean(1)
    print(f"   доля всех L5 E руки, выстреливших за окно: обычно {float(frac.mean()):.1%} (95% {float(frac.quantile(0.95)):.1%}), "
          f"в окнах нажатия {float(frac[cond].mean()) if cond.any() else float('nan'):.1%}; "
          f"окна с ≥ 20% L5: {int((frac >= 0.2).sum())}", flush=True)
    # автокорреляция общей доли: залпы ритмичны?
    x = frac - frac.mean()
    ac = [float((x[:-l] * x[l:]).mean() / x.var()) for l in range(1, 21)]
    lag = int(torch.tensor(ac[1:]).argmax()) + 2
    print(f"   автокорреляция доли L5 по лагу 10…200 мс: " + " ".join(f"{a:.2f}" for a in ac[:10]) + f" …; пик дальше 10 мс — {lag * WIN:.0f} мс", flush=True)

    # 3. вход
    names = []
    for k in targets:
        names += [f"{k} ← {p.name}" for p in paths[k]] + [f"{k}: среднее поле AMPA", f"{k}: фон bias", f"{k}: частота Гц"]
    pre = torch.zeros(n_win, dtype=torch.bool)
    idx = cond.nonzero().flatten()
    pre[(idx - 1).clamp_min(0)] = True
    print(f"\n3. вход по путям (мера поставки Σ вес × Гц; в среднем / в окне перед нажатием / в окне нажатия):", flush=True)
    for j, nm in enumerate(names):
        print(f"   {nm:60s} {float(D[:, j].mean()):10.3f} {float(D[pre, j].mean()) if pre.any() else float('nan'):10.3f} "
              f"{float(D[cond, j].mean()) if cond.any() else float('nan'):10.3f}", flush=True)
    print(f"всего {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
