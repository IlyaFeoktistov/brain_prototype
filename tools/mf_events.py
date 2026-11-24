"""
Общее среднее поле против независимых событий у недостающих входов пути
(CHAT_PLAN шаг 3, цикл 38; cfg.mf_independent, engine.set_mf_independent).
Гипотеза цикла 37: волны EC — колебания популяции L5 ПФК, которые среднее
поле пути ПФК → EC (450 из 500 входов) раздаёт всем клеткам EC общим током.

Мозг — как в опыте B (organs=words): разгон 2.5 с, адаптация с гомеостазом
окнами по 1 с, затем снимок; из одного снимка — запись без гомеостаза
(окна 5 мс) с общим полем (контроль) и с независимыми событиями. Явные
синапсы, средний вход и его ход во времени у обоих одинаковы.

Меры: вход EC (возбуждающий ток синапсов + поле, мВ; 500 клеток, отсчёт в
конце окна 5 мс) — среднее, разброс средних между клетками, σ во времени у
клетки, σ общей части (среднее по клеткам), корреляция входа пар; спайки EC
— популяционный CV по окнам 5 мс, корреляция пар; L5 ПФК — частота и CV (тот
же ли источник); гранулярные DG — частота; CA3 PC — частота, окна ≥ 20% и ≥
50% PC. cut=1 (по умолчанию) — третьим из того же снимка путь перерезан
целиком.

    PYTHONPATH=. python tools/mf_events.py [seed=42] [adapt=30] [record=4] [path=prefrontal->entorhinal]
"""

from __future__ import annotations

import sys

import torch

from brain.config import BrainConfig
from brain.mind import Mind
from tools import develop


def pair_corr(x: torch.Tensor) -> float:
    """Средняя корреляция Пирсона по парам столбцов (молчащие — без них)."""

    x = x - x.mean(0)
    sd = x.std(0)
    live = sd > 0
    x = x[:, live] / sd[live]
    cc = (x.T @ x) / (x.shape[0] - 1)
    n = cc.shape[0]
    return float(cc[~torch.eye(n, dtype=torch.bool, device=cc.device)].mean())


def record(b, s, bins: int, gen_seed: int) -> None:
    A = b.areas
    dev = b.device
    ec = A["entorhinal"].E
    gen = torch.Generator().manual_seed(gen_seed)
    pick = (torch.randperm(ec.size, generator=gen)[:500] + ec.start).to(dev)
    l5 = A["prefrontal"].layers["L5"]["E"].slice
    ca3 = A["ca3"].E.slice
    dg = A["dentate"].E.slice
    inp = torch.zeros(bins, 500, device=dev)
    spk = torch.zeros(bins, 500, device=dev)
    ec_pop = torch.zeros(bins, device=dev)
    l5_pop = torch.zeros(bins, device=dev)
    ca3_frac = torch.zeros(bins, device=dev)
    ca3_rate = torch.zeros(bins, device=dev)
    dg_rate = torch.zeros(bins, device=dev)

    for t in range(bins):
        r = b.observe(5.0, s._blank(5.0))
        inp[t] = (b.i_syn + b.last_dc_e)[pick]
        spk[t] = r[pick] * 0.005
        ec_pop[t] = r[ec.slice].mean()
        l5_pop[t] = r[l5].mean()
        ca3_frac[t] = (r[ca3] > 0).float().mean()
        ca3_rate[t] = r[ca3].mean()
        dg_rate[t] = r[dg].mean()

    cell_mean = inp.mean(0)
    common = inp.mean(1)
    print(f"   вход EC, мВ: среднее {float(inp.mean()):.2f}, разброс средних между клетками σ {float(cell_mean.std()):.2f}, "
          f"σ во времени у клетки {float(inp.std(0).mean()):.2f}, σ общей части {float(common.std()):.2f}, "
          f"корр. входа пар {pair_corr(inp):.3f}")
    print(f"   EC: {float(ec_pop.mean()):.2f} Гц, CV по окнам {float(ec_pop.std() / ec_pop.mean()):.2f}, "
          f"корр. спайков пар (5 мс) {pair_corr(spk):.3f}, по 20 мс {pair_corr(spk[: bins // 4 * 4].view(-1, 4, 500).sum(1)):.3f}")
    print(f"   L5 ПФК: {float(l5_pop.mean()):.2f} Гц, CV {float(l5_pop.std() / l5_pop.mean()):.2f}; "
          f"DG гранулярные {float(dg_rate.mean()):.2f} Гц; CA3 PC {float(ca3_rate.mean()):.2f} Гц, "
          f"окон ≥ 20% — {int((ca3_frac >= 0.2).sum())}, ≥ 50% — {int((ca3_frac >= 0.5).sum())} из {bins}", flush=True)


def main(opts: dict) -> None:
    cfg = BrainConfig()
    cfg.size = "small"
    cfg.seed = int(opts.get("seed", 42))
    cfg.autosave = False
    cfg.homeostasis = True
    cfg.event_overflow = "warn"
    cfg.organs = ("words",)
    cfg.mf_independent = (opts.get("path", "prefrontal->entorhinal"),)
    b = Mind(cfg).brain
    b.plasticity = False
    if not b.mf_events:
        raise SystemExit(f"путь {cfg.mf_independent} без недостающих входов (k_data)")
    for post, pre, coeff, n_equiv, w in b.mf_events:
        print(f"{pre.name} → {post.name}: поле {coeff:.1f} мВ/кГц = {n_equiv:.0f} эквивалентных входов × вес {w:.3f}", flush=True)
    b.set_mf_independent(False)
    s = develop.Silence(b)

    b.run(2500.0, s._blank(2500.0))
    for _ in range(int(opts.get("adapt", 30))):
        b.homeostasis_update(b.observe(1000.0, s._blank(1000.0)))
    b.cfg.homeostasis = False
    saved = b._snapshot()
    bins = int(float(opts.get("record", 4)) * 1000 / 5)

    for on in (False, True):
        b._restore(saved)
        b.set_mf_independent(on)
        print(f"\n{'независимые события' if on else 'общее поле (контроль)'}, {bins * 5 / 1000:.0f} с из одного снимка:", flush=True)
        record(b, s, bins, cfg.seed)

    if opts.get("cut", "1") == "1":
        # Перерезка пути целиком (построенные и поле) из того же снимка —
        # несёт ли путь общую часть входа (диагностика, последней).
        b._restore(saved)
        b.set_mf_independent(False)
        src, dst = cfg.mf_independent[0].split("->")
        for p in b.projections:
            if p.name.startswith(f"{src} -> {dst}"):
                p.weight.zero_()
        for post, pre, coeff, _, _ in b.mf_events:
            b.mf_exc[int(b.mf_post_id[post.start]), int(b.mf_pre_id[pre.start])] -= coeff
        b.recompile()
        print(f"\nпуть {src} → {dst} перерезан (построенные и поле), {bins * 5 / 1000:.0f} с из того же снимка:", flush=True)
        record(b, s, bins, cfg.seed)


if __name__ == "__main__":
    main(dict(a.split("=", 1) for a in sys.argv[1:]))
