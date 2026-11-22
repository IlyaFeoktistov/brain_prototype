"""
MD → ПФК (CHAT_PLAN шаг 3, цикл 15): перекрытие построенных источников MD у
пар клеток ПФК (доля общих среди своих) против случайной проводки, и
синхронность самих клеток MD (CV популяции по окнам 5 мс, корреляция пар
спайков) в опыте B (organs=words, адаптация 3 с, запись 3 с).

    PYTHONPATH=. python tools/md_overlap.py [seed=42] [extpop=prefrontal:1] [cut=src->dst] [t_off=1]
"""

from __future__ import annotations

import sys

import torch

from brain.config import BrainConfig
from brain.mind import Mind
from tools import develop


def main(opts: dict) -> None:
    cfg = BrainConfig()
    cfg.size = "small"
    cfg.seed = int(opts.get("seed", 42))
    cfg.autosave = False
    cfg.homeostasis = True
    cfg.event_overflow = "warn"
    cfg.organs = ("words",)
    develop.override_external(cfg, opts.get("extpop", ""))
    b = Mind(cfg).brain
    b.plasticity = False
    md = b.areas["md"].E
    # cut=src->dst[,…] — перерезка (построенные и среднее поле); t_off=1 — без
    # T-тока у MD. Диагностика: среднее входа не сохраняется.
    for pair in (x for x in opts.get("cut", "").split(",") if x):
        src, dst = pair.split("->")
        hit = [p for p in b.projections if p.name == f"{src} -> {dst}" or p.name.startswith(f"{src} -> {dst}.")]
        for p in hit:
            p.weight.zero_()
        if b.mf_on:
            q = b.mf_post_id[b.areas[dst].cells].unique()
            i = b.mf_pre_id[b.areas[src].cells].unique()
            for m in (b.mf_exc, b.mf_inh, b.mf_ach):
                m[q[:, None], i[None, :]] = 0.0
        print(f"перерезан {src} → {dst}: путей {len(hit)}", flush=True)
    if opts.get("t_off") == "1":
        b.t_mask[md.slice] = False
        print("T-ток MD выключен", flush=True)
    gen = torch.Generator().manual_seed(1)
    for p in b.projections:
        if not p.name.startswith("md -> prefrontal"):
            continue
        pre = p.pre.long()
        n, k = pre.shape
        idx = torch.randperm(n, generator=gen)[:400].to(pre.device)
        sub = torch.zeros(len(idx), md.size, device=pre.device)
        sub.scatter_(1, pre[idx] - md.start, 1.0)
        shared = sub @ sub.T
        off = shared[~torch.eye(len(idx), dtype=torch.bool, device=shared.device)]
        expect = k * k / md.size
        print(f"{p.name}: построено {k} источников на клетку из {md.size}; общих у пары {float(off.mean()):.2f} "
              f"(случайно {expect:.2f}), доля {float(off.mean()) / k:.1%}, максимум {int(off.max())}")

    s = develop.Silence(b)
    b.run(2500.0, s._blank(2500.0))
    for _ in range(3):
        b.homeostasis_update(b.observe(1000.0, s._blank(1000.0)))
    b.cfg.homeostasis = False
    bins = 600
    pick = torch.randperm(md.size, generator=gen)[:200].to(b.device) + md.start
    pop = torch.zeros(bins, device=b.device)
    cells = torch.zeros(bins, 200, device=b.device)
    for t in range(bins):
        r = b.observe(5.0, s._blank(5.0))
        pop[t] = r[md.slice].sum() * 0.005
        cells[t] = r[pick] * 0.005
    rate = float(pop.mean()) / md.size / 0.005
    cv = float(pop.std() / pop.mean())
    cv_ind = 1.0 / (md.size * rate * 0.005) ** 0.5
    x = cells - cells.mean(0)
    sd = x.std(0)
    live = sd > 0
    z = x[:, live] / sd[live]
    cc = (z.T @ z) / (z.shape[0] - 1)
    corr = float(cc[~torch.eye(cc.shape[0], dtype=torch.bool, device=cc.device)].mean())
    share = pop / md.size
    big = share > 5 * share.mean()
    print(f"доля клеток MD в окне 5 мс: средняя {float(share.mean()):.1%}, 95-я перцентиль {float(torch.quantile(share, 0.95)):.1%}, "
          f"максимум {float(share.max()):.1%}; окон-событий (> 5× средней) {float(big.float().mean()):.1%}")
    print(f"MD: {rate:.2f} Гц, CV популяции по окнам 5 мс {cv:.2f} (независимые — {cv_ind:.2f}), "
          f"корреляция пар спайков {corr:.3f} (клеток {int(live.sum())})")


if __name__ == "__main__":
    main(dict(a.split("=", 1) for a in sys.argv[1:]))
