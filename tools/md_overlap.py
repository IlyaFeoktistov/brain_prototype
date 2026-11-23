"""
MD → ПФК (CHAT_PLAN шаг 3, цикл 15): перекрытие построенных источников MD у
пар клеток ПФК (доля общих среди своих) против случайной проводки, и
синхронность самих клеток MD (CV популяции по окнам 5 мс, корреляция пар
спайков) в опыте B (organs=words, адаптация 3 с, запись 3 с).

    PYTHONPATH=. python tools/md_overlap.py [seed=42] [extpop=prefrontal:1] [cut=src->dst] [t_off=1]
        [smooth=src->dst] [ach=none|trn|md|both] [adapt=3]
"""

from __future__ import annotations

import sys

import torch

from brain.config import BrainConfig
from brain.mind import Mind
from tools import develop


def pair_corr(cells: torch.Tensor) -> float:
    """Средняя корреляция пар клеток по окнам [окна, клетки] (молчащие — мимо)."""

    x = cells - cells.mean(0)
    sd = x.std(0)
    live = sd > 0
    z = x[:, live] / sd[live]
    cc = (z.T @ z) / (z.shape[0] - 1)
    return float(cc[~torch.eye(cc.shape[0], dtype=torch.bool, device=cc.device)].mean())


def inhibition(b, tcells: torch.Tensor, md, trn) -> None:
    """
    Тормозный вход TRN MD → MD по построенным синапсам: сумма весов спайков
    источников за окно 5 мс [мВ·спайк] на клетку MD. Среднее, колебания во
    времени (CV средней по MD), разброс по клеткам (CV средней за запись).
    """

    paths = [p for p in b.projections if p.name.startswith("trn_md -> md")]
    if not paths:
        return
    inp = torch.zeros(tcells.shape[0], md.size, device=tcells.device)
    for p in paths:
        pre = p.pre.long() - trn.start
        w = p.weight.float()
        ok = (pre >= 0) & (pre < trn.size)
        post = torch.arange(p.post.slice.start, p.post.slice.stop, device=tcells.device) - md.start
        # [окна, пост, k] — спайки источников × вес; построенные пути — на E MD.
        got = tcells[:, pre.clamp(0, trn.size - 1)] * (w * ok)
        inp[:, post] += got.sum(-1)
    mean_t = inp.mean(1)
    per_cell = inp.mean(0)
    print(f"вход TRN → MD: {float(mean_t.mean()):+.2f} мВ·сп за 5 мс на клетку; во времени CV {float(mean_t.std() / mean_t.mean().abs()):.2f}, "
          f"окна > 3× среднего {float((mean_t.abs() > 3 * mean_t.mean().abs()).float().mean()):.1%}; "
          f"по клеткам CV {float(per_cell.std() / per_cell.mean().abs()):.2f}")


def main(opts: dict) -> None:
    cfg = BrainConfig()
    cfg.size = "small"
    cfg.seed = int(opts.get("seed", 42))
    cfg.autosave = False
    cfg.homeostasis = True
    cfg.event_overflow = "warn"
    cfg.organs = ("words",)
    cfg.thalamic_ach = opts.get("ach", "both") != "none" or opts.get("mono") == "1"
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
    # ach=none|trn|md|both — у каких клеток петли действует проводимость ACh
    # (anatomy.RECEPTOR_CONDUCTANCE; по умолчанию — как в сборке, both).
    # mono=1 — у TRN MD ещё и NA + 5-HT (закрывают K⁺); без него их строки нулевые.
    ach = opts.get("ach", "both")
    names = list(b.MODULATORS)
    zones = {"trn": "trn_md", "md": "md"}
    for i, name in enumerate(names):
        for key, zone in zones.items():
            on = (ach in (key, "both")) if name == "acetylcholine" else (key == "trn" and opts.get("mono") == "1")
            if not on:
                b.receptor_g[i, b.areas[zone].E.slice] = 0.0
    b.receptor_g_on = bool(b.receptor_g.abs().sum() > 0)
    print(f"проводимость ACh: {ach}, NA + 5-HT на TRN: {opts.get('mono') == '1'}", flush=True)
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
    # adapt=N — окон гомеостаза (с): частоты возвращаются к покою при новой
    # проводимости (фон ядер подобран без неё).
    for _ in range(int(opts.get("adapt", 3))):
        b.homeostasis_update(b.observe(1000.0, s._blank(1000.0)))
    b.cfg.homeostasis = False
    lim = b.cfg.homeostasis_max_mv
    for zone in ("md", "trn_md"):
        sh = b.homeo_shift[b.areas[zone].E.slice]
        print(f"гомеостаз {zone}: сдвиг {float(sh.mean()):+.1f} мВ, у предела ±{lim:.0f} {float((sh.abs() >= lim - 1e-3).float().mean()):.0%}", flush=True)
    # smooth=src->dst — после адаптации: тот же средний вход пути без колебаний
    # (develop.smooth_paths: у источника без поля — постоянный ток по фактической
    # частоте и ресурсу).
    develop.smooth_paths(b, opts.get("smooth", ""))
    bins = 600
    pick = torch.randperm(md.size, generator=gen)[:200].to(b.device) + md.start
    pop = torch.zeros(bins, device=b.device)
    cells = torch.zeros(bins, 200, device=b.device)
    tsum = torch.zeros(bins, device=b.device)
    trn = b.areas["trn_md"].E
    tcells = torch.zeros(bins, trn.size, device=b.device)
    pfc = b.areas["prefrontal"].layers["L4"]["E"]
    pfc_pop = torch.zeros(bins, device=b.device)
    v_md = torch.zeros(bins, device=b.device)
    below = torch.zeros(bins, device=b.device)
    ach_occ = 0.0
    g_trn = 0.0
    for t in range(bins):
        r = b.observe(5.0, s._blank(5.0))
        pop[t] = r[md.slice].sum() * 0.005
        cells[t] = r[pick] * 0.005
        tcells[t] = r[trn.slice] * 0.005
        tsum[t] = tcells[t].sum()
        pfc_pop[t] = r[pfc.slice].mean()
        v_md[t] = b.v[md.slice].mean()
        below[t] = (b.v[md.slice] < b.cfg.t_vh_mv).float().mean()
        ach_occ += float(b.occupancy("acetylcholine")) / bins
        occ = torch.tensor([b.occupancy(n) for n in b.MODULATORS], device=b.device)
        g_trn += float((occ @ b.receptor_g[:, trn.slice]).mean()) / bins
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
    t_rate = float(tsum.mean()) / trn.size / 0.005
    t_cv = float(tsum.std() / tsum.mean()) if float(tsum.mean()) > 0 else float("nan")
    t_ind = 1.0 / max(1e-9, trn.size * t_rate * 0.005) ** 0.5
    a, z = tsum - tsum.mean(), pop - pop.mean()
    lag = max(range(-10, 11), key=lambda k: float((a[max(0, -k): bins - max(0, k)] * z[max(0, k): bins - max(0, -k)]).mean()))
    print(f"TRN MD: {trn.size} клеток, {t_rate:.2f} Гц, CV {t_cv:.2f} (независимые — {t_ind:.2f}); "
          f"MD после TRN на {lag * 5:+d} мс (наибольшая связь)")
    print(f"TRN MD: корреляция пар спайков {pair_corr(tcells):.3f}; спайков в пачках (≥ 2 за 5 мс) "
          f"{float((tcells * (tcells >= 1.5)).sum() / tcells.sum().clamp_min(1)):.0%}, одиночные {float((tcells == 1).sum()) / bins / trn.size / 0.005:.2f} Гц")
    print(f"занятость: ACh {ach_occ:.2f}, NA {b.occupancy('noradrenaline'):.2f}, 5-HT {b.occupancy('serotonin'):.2f}; "
          f"TRN: Δg/g_L {g_trn:+.2f}, R_вх ×{1 / (1 + g_trn):.2f}")
    print(f"MD: потенциал {float(v_md.mean()):.1f} мВ, ниже V_h T-тока {float(below.mean()):.0%} клеток; "
          f"ПФК L4 E {float(pfc_pop.mean()):.2f} Гц, CV по окнам {float(pfc_pop.std() / pfc_pop.mean().clamp_min(1e-9)):.2f}")
    inhibition(b, tcells, md, trn)
    print(f"MD: {rate:.2f} Гц, CV популяции по окнам 5 мс {cv:.2f} (независимые — {cv_ind:.2f}), "
          f"корреляция пар спайков {corr:.3f} (клеток {int(live.sum())})")


if __name__ == "__main__":
    main(dict(a.split("=", 1) for a in sys.argv[1:]))
