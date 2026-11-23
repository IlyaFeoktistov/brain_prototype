"""
Колонки ПФК и смысловой коры в одном мозге и при одной адаптации (CHAT_PLAN
шаг 3, цикл 27): почему колонка ПФК при частотах, близких к данным, даёт
общий быстрый ритм E–PV, а смысловая с той же проводкой (цикл 13) — нет.

Мозг — как в опыте B (organs=words): разгон 2.5 с, адаптация с гомеостазом
окнами по 1 с, затем запись без гомеостаза с шагом 1 мс (кусок шага движка
CHUNK → 1 мс, как в tools/md_sync.py). По популяциям каждой колонки:

  - частота, CV популяции по окнам 5 мс, корреляция пар спайков (окно 5 мс,
    до 200 клеток);
  - токи клеток (мВ движка) раз в 1 мс у 200 клеток: возбуждение (i_syn —
    построенные синапсы и внешний фон), торможение (i_inh); среднее, σ во
    времени у клетки, корреляция колебаний между клетками (общий ли вход);
  - задержка торможения за возбуждением у клеток E: лаг наибольшей
    корреляции средних по популяции i_syn и −i_inh (плюс — торможение позже);
  - ККГ популяций L2/3 E → L2/3 PV (спайки, 1 мс, ±30 мс): пик и его лаг,
    отношение пика к уровню на краях (плюс — PV позже E);
  - сдвиг гомеостаза и доля клеток в упоре ±homeostasis_max_mv.

extpop=prefrontal:1 — внешний фон ПФК по данным (цикл 26: с подобранным фоном
регулятор ПФК при 30 с в упоре).

    PYTHONPATH=. python tools/column_compare.py [areas=prefrontal,association] [seed=42] [adapt=30] [record=3] [extpop=…] [smooth=md->prefrontal] [indep_md=1] [md_scale=1]
"""

from __future__ import annotations

import sys

import torch

from brain.config import BrainConfig
from brain.mind import Mind
from tools import develop


def pair_corr(x: torch.Tensor) -> float:
    x = x - x.mean(0)
    sd = x.std(0)
    live = sd > 0
    if int(live.sum()) < 3:
        return float("nan")
    z = x[:, live] / sd[live]
    cc = (z.T @ z) / (z.shape[0] - 1)
    return float(cc[~torch.eye(cc.shape[0], dtype=torch.bool, device=cc.device)].mean())


def xcorr(a: torch.Tensor, b: torch.Tensor, lag: int) -> list[float]:
    """Корреляция a(t) и b(t + k) для k = −lag…lag (плюс — b позже a)."""

    a = (a - a.mean()) / a.std().clamp_min(1e-12)
    b = (b - b.mean()) / b.std().clamp_min(1e-12)
    n = a.shape[0]
    return [float((a[max(0, -k): n - max(0, k)] * b[max(0, k): n - max(0, -k)]).mean()) for k in range(-lag, lag + 1)]


def independent_md(b, s, scale: float = 1.0) -> None:
    """
    indep_md=1 — вход MD → ПФК (L4 E и L4 PV) без общей части: построенные
    синапсы → 0, у каждой клетки свой пуассон (engine.tc_missing — механизм
    недостающих таламических входов, van Albada et al. 2015): k событий с
    частотой её собственных клеток MD (сглажена cfg.tc_missing_tau_ms), вес
    синапса тот же, × ресурс депрессии тех же клеток MD. Среднее и разброс
    у клетки те же; убраны перекрытие источников и корреляции пар MD.
    scale — множитель веса событий: подогнать средний ток L4 к контролю
    (синхронные спайки построенных синапсов дают меньше тока — депрессия
    и движущая сила в залпе). Диагностика, не механизм.
    """

    rate = b.observe(1000.0, s._blank(1000.0)) / 1000.0
    for p in [p for p in b.projections if p.name in ("md -> prefrontal", "md -> prefrontal.I")]:
        pre = p.pre.long()
        w = p.weight.float().mean(1).clone() * scale
        b.tc_missing.append((p.post, pre, float(pre.shape[1]), w))
        print(f"{p.name}: построенные → независимые события, {pre.shape[1]} на клетку, вес {float(w.mean()):.3f}, "
              f"частота MD {float(rate[pre].mean()) * 1000:.1f} Гц", flush=True)
        p.weight.zero_()
    b.tc_rate = rate[: b.n_slots].clone() if rate.numel() >= b.n_slots else torch.cat(
        [rate, torch.zeros(b.n_slots - rate.numel(), device=rate.device)])
    b._graphs.clear()
    b.observe(2000.0, s._blank(2000.0))


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
    s = develop.Silence(b)
    names = opts.get("areas", "prefrontal,association").split(",")
    pops = {(a, f"{layer}.{kind}"): p for a in names for layer, group in b.areas[a].layers.items()
            for kind, p in group.items()}

    b.run(2500.0, s._blank(2500.0))
    for _ in range(int(opts.get("adapt", 30))):
        b.homeostasis_update(b.observe(1000.0, s._blank(1000.0)))
    b.cfg.homeostasis = False
    # После адаптации: путь — тем же средним без колебаний (диагностика,
    # develop.smooth_paths), 1 с на переход до записи.
    if opts.get("smooth"):
        develop.smooth_paths(b, opts["smooth"])
        b.observe(1000.0, s._blank(1000.0))
    if opts.get("indep_md") == "1":
        independent_md(b, s, float(opts.get("md_scale", 1.0)))

    b.CHUNK = int(round(1.0 / b.cfg.dt_ms))
    b._graphs.clear()
    bins = int(float(opts.get("record", 3)) * 1000)
    gen = torch.Generator().manual_seed(1)
    picks = {k: torch.randperm(p.size, generator=gen)[:200].to(b.device) + p.start for k, p in pops.items()}
    exc = {k: torch.zeros(bins, len(picks[k]), device=b.device) for k in pops}
    inh = {k: torch.zeros(bins, len(picks[k]), device=b.device) for k in pops}
    spk = {k: torch.zeros(bins, len(picks[k]), device=b.device) for k in pops}
    popn = {k: torch.zeros(bins, device=b.device) for k in pops}
    for t in range(bins):
        r = b.observe(1.0, s._blank(1.0)) * 0.001
        for k, p in pops.items():
            exc[k][t] = b.i_syn[picks[k]].float()
            inh[k][t] = b.i_inh[picks[k]].float()
            spk[k][t] = r[picks[k]]
            popn[k][t] = r[p.slice].sum()

    shift = getattr(b, "homeo_shift", None)
    cap = b.cfg.homeostasis_max_mv
    w5 = bins // 5
    print(f"зерно {cfg.seed}, адаптация {opts.get('adapt', 30)} с, запись {bins / 1000:.0f} с"
          f"{', внешний фон ' + opts['extpop'] if opts.get('extpop') else ''}"
          f"{', сглажен ' + opts['smooth'] if opts.get('smooth') else ''}"
          f"{', MD → ПФК независимыми событиями' if opts.get('indep_md') == '1' else ''}")
    for a in names:
        print(f"== {a}: популяция  частота | CV 5 мс, корр. пар | возб.: среднее, σ, r | торм.: среднее, σ, r | гомеостаз (в упоре)")
        for (area, k), p in pops.items():
            if area != a:
                continue
            key = (area, k)
            hz = float(popn[key].sum()) / p.size / (bins / 1000)
            pn = popn[key][: w5 * 5].view(w5, 5).sum(1)
            cv = float(pn.std() / pn.mean()) if float(pn.mean()) > 0 else float("nan")
            sp = spk[key][: w5 * 5].view(w5, 5, -1).sum(1)
            e, i = exc[key], inh[key]
            hs = (f"{float(shift[p.slice].mean()):+.1f} ({float((shift[p.slice].abs() >= cap - 1e-3).float().mean()):.0%})"
                  if shift is not None else "—")
            print(f"   {k:8s} {hz:6.2f} Гц | {cv:5.2f}, {pair_corr(sp):6.3f} | {float(e.mean()):6.2f}, {float(e.std(0).mean()):5.2f}, "
                  f"{pair_corr(e):5.2f} | {float(i.mean()):6.2f}, {float(i.std(0).mean()):5.2f}, {pair_corr(i):5.2f} | {hs}", flush=True)
        # Торможение за возбуждением у E (средние токи популяции, ±20 мс).
        parts = []
        for layer in ("L23", "L4", "L5"):
            key = (a, f"{layer}.E")
            if key not in pops:
                continue
            cc = xcorr(exc[key].mean(1), -inh[key].mean(1), 20)
            k = max(range(41), key=lambda j: cc[j])
            parts.append(f"{layer} r {cc[k]:.2f} при {k - 20:+d} мс")
        print("   торможение за возбуждением у E (корр. средних токов):", "; ".join(parts))
        e_key, pv_key = (a, "L23.E"), (a, "L23.PV")
        cc = xcorr(popn[e_key], popn[pv_key], 30)
        k = max(range(61), key=lambda j: cc[j])
        edge = (sum(cc[:5]) + sum(cc[-5:])) / 10
        print(f"   ККГ L2/3 E → PV (популяции, 1 мс): пик r {cc[k]:.3f} при {k - 30:+d} мс, края {edge:.3f}; "
              f"r при 0 / +2 / +5 / +10 мс: {cc[30]:.3f} / {cc[32]:.3f} / {cc[35]:.3f} / {cc[40]:.3f}", flush=True)


if __name__ == "__main__":
    main(dict(a.split("=", 1) for a in sys.argv[1:]))
