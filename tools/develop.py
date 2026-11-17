"""
Фаза «развития» с гомеостазом возбудимости (cfg.homeostasis, engine.
homeostasis_update): мозг живёт с пустым экраном окнами win мс, после каждого
окна регулируемые клетки сдвигают фон к частоте покоя своего типа. В конце —
сдвиги по популяциям (среднее, доля упёршихся в предел) и файл сдвигов
~/.brain/fits/<out>-homeo.pt (стенды берут его по homeo=<out>; гомеостаз
на замерах выключен — сдвиг заморожен).

    PYTHONPATH=. python tools/develop.py [seconds=60] [win=1000] [seed=42] [fit=lgn-a]
        [relay_v_t_mv=-59] [trn_scale=0.1] [out=lgn-d-pop-s1] [every=10] [scope=population|cell] [built_shares=0|1] [map_ee_sheet=зоны]
        [coincidence=0|1] [dend_count=15] [dend_window=5] [probe=1]

scope=population — общий шаг популяции (cfg.homeostasis_scope): разброс
клеток остаётся от проводки; в отчёте — распределение частот клеток зрительных
популяций (медиана, 10–90%, доля без спайков за окно).
"""

from __future__ import annotations

import sys

import torch

from brain import anatomy
from brain.config import BRAIN_HOME, BrainConfig
from brain.mind import Mind
from brain.session import Session


def populations(b) -> list[tuple[str, slice]]:
    out = []
    for name, area in b.areas.items():
        if area.layers is None:
            out.append((name, area.E.slice))
        else:
            for layer, group in area.layers.items():
                for kind, pop in group.items():
                    out.append((f"{name}.{layer}{kind}", pop.slice))
    return out


def report(b, rates, label: str) -> None:
    print(f"\n{label}: популяция — частота / цель Гц, сдвиг мВ, у предела")
    lim = b.cfg.homeostasis_max_mv
    for name, sl in populations(b):
        t = b.homeo_target[sl]
        if t.isnan().all():
            continue
        shift = b.homeo_shift[sl]
        edge = float((shift.abs() >= lim - 1e-3).float().mean())
        print(f"   {name:22s} {float(rates[sl].mean()):6.2f} / {float(t.mean()):5.2f}   {float(shift.mean()):+6.2f}"
              + (f"   {edge:.0%}" if edge > 0 else ""), flush=True)
    for name in ("visual.L23E", "visual.L4E", "letters.L4E", "letters.L23E"):
        sl = dict(populations(b))[name]
        x = rates[sl].float()
        q = torch.quantile(x, torch.tensor([0.1, 0.5, 0.9], device=x.device))
        print(f"   распределение {name:14s} медиана {float(q[1]):.2f}, 10–90% {float(q[0]):.2f}–{float(q[2]):.2f}, "
              f"без спайков {float((x == 0).float().mean()):.0%}, > 1 Гц {float((x > 1).float().mean()):.0%}", flush=True)


PROBE = ("motor.L23E", "hand.L23E", "premotor.L23E", "letters.L23E")


def probed_window(b, s, win: float, step: float = 10.0) -> torch.Tensor:
    """
    Окно отчёта кусками по step мс (те же частоты за окно) со снимками L2/3 E
    зон PROBE: доля клеток с плато ветки > 0.5 (детектор — состояние плато,
    иначе сигмоида тока), самая горячая ветка (счёт синапсов за окно при
    детекторе, иначе мВ), разброс доли плато по снимкам (постоянное плато —
    ровно, от совпадений — вспышками), расстояние v до порога v_t.
    """

    cfg = b.cfg
    nb = max(1, cfg.dendrite_branches)
    pops = dict(populations(b))
    snaps = {k: {"pl": [], "hot": [], "gap": []} for k in PROBE}
    total = torch.zeros(b.n, device=b.device)
    n = int(win / step)
    for _ in range(n):
        total += b.observe(step, s._blank(step))
        for k in PROBE:
            sl = pops[k]
            br = b.i_dend.view(b.n_slots, nb)[sl]
            hot = br.amax(1)
            if cfg.dendrite_coincidence:
                pl = b.dend_plateau.view(b.n_slots, nb)[sl].amax(1)
            else:
                pl = torch.sigmoid((hot - cfg.dendrite_threshold_mv) / cfg.dendrite_slope_mv)
            snaps[k]["pl"].append(float((pl > 0.5).float().mean()))
            snaps[k]["hot"].append(hot.float())
            snaps[k]["gap"].append(float((b.v_t[sl] - b.v[sl]).median()))
    u = "синапсов за окно" if cfg.dendrite_coincidence else "мВ"
    print(f"\n   ветки L2/3 E (снимки каждые {step:.0f} мс; горячая ветка — {u}):", flush=True)
    for k in PROBE:
        pl = torch.tensor(snaps[k]["pl"])
        hot = torch.cat(snaps[k]["hot"])
        hot = hot[:: max(1, hot.numel() // 1_000_000)]
        hq = torch.quantile(hot, torch.tensor([0.5, 0.9, 0.99], device=hot.device))
        print(f"   {k:14s} плато {float(pl.mean()):5.1%} (по снимкам 10–90% {float(pl.quantile(0.1)):.1%}–{float(pl.quantile(0.9)):.1%}, "
              f"снимков с плато > 1% клеток {float((pl > 0.01).float().mean()):.0%}); горячая ветка медиана {float(hq[0]):.1f}, "
              f"90% {float(hq[1]):.1f}, 99% {float(hq[2]):.1f}; v_t − v медиана {sum(snaps[k]['gap']) / n:.1f} мВ", flush=True)
    return total / n


def main(opts: dict) -> None:
    cfg = BrainConfig()
    cfg.size = "small"
    cfg.seed = int(opts.get("seed", 42))
    cfg.autosave = False
    cfg.homeostasis = True
    cfg.homeostasis_scope = opts.get("scope", cfg.homeostasis_scope)
    if opts.get("bg"):
        # Структура фона недостающих входов (cfg.background_mode).
        cfg.background_mode = opts["bg"]
        cfg.background_patch_px = float(opts.get("bg_patch", cfg.background_patch_px))
    if "tc_indep" in opts:
        # Недостающие входы LGN → L4 своими событиями (cfg.tc_missing_independent, умолчание вкл.;
        # tc_indep=0 — прежняя проводка как контроль; tc_tau — сглаживание частоты).
        cfg.tc_missing_independent = opts["tc_indep"] != "0"
        cfg.tc_missing_tau_ms = float(opts.get("tc_tau", cfg.tc_missing_tau_ms))
    if opts.get("tonic"):
        # Постоянная тормозная проводимость (cfg.tonic_inh_g): зона.слой.тип:g, через запятую.
        cfg.tonic_inh_g = tuple((*k.split("."), float(g)) for k, g in (e.split(":") for e in opts["tonic"].split(",")))
    if opts.get("bg_scale"):
        # Доза внешнего фона (cfg.background_scale).
        cfg.background_scale = float(opts["bg_scale"])
    if opts.get("ee_skip"):
        cfg.column_ee_human_skip = tuple(opts["ee_skip"].split(","))
    if opts.get("nmda"):
        # NMDA у клеток колонок, E и тормозных (cfg.nmda, cfg.nmda_inh_ratio).
        cfg.nmda = True
        cfg.nmda_chariker = opts["nmda"] == "chariker"
    for k in ("relay_v_t_mv", "relay_rest_mv"):
        if k in opts:
            setattr(cfg, k, float(opts[k]))
    if opts.get("built_shares"):
        # Доли тормозных типов в проводке колонки — среди построенных (cfg.column_built_shares, ISSUES 25).
        cfg.column_built_shares = opts["built_shares"] != "0"
    if opts.get("map_ee_sheet"):
        # Местные E → E слоя 2/3 зон-карт — по расстоянию на листе (cfg.map_ee_sheet).
        cfg.map_ee_sheet = tuple(opts["map_ee_sheet"].split(","))
    if opts.get("coincidence"):
        # Ветки дендрита — детектор совпадений (cfg.dendrite_coincidence, ISSUES 26); порог/окно — по желанию.
        cfg.dendrite_coincidence = opts["coincidence"] != "0"
        cfg.dendrite_count_threshold = float(opts.get("dend_count", cfg.dendrite_count_threshold))
        cfg.dendrite_window_ms = float(opts.get("dend_window", cfg.dendrite_window_ms))
    if opts.get("fit"):
        anatomy.use_fit(cfg, opts["fit"])
    mind = Mind(cfg)
    b = mind.brain
    b.plasticity = False
    # Пустой экран — ямка видит ровную страницу (Session._blank).
    s = Session(mind)
    for p in b.projections:
        if p.name == "trn -> lgn":
            p.weight.mul_(float(opts.get("trn_scale", 1.0)))
    b.run(2500.0, s._blank(2500.0))
    win, every = float(opts.get("win", 1000)), int(opts.get("every", 10))
    windows = int(float(opts.get("seconds", 60)) * 1000 / win)

    for w in range(windows):
        shown = w == 0 or (w + 1) % every == 0
        # Окна отчёта — кусками со снимками веток (probed_window); частоты те же, шум генератора — свой.
        rates = probed_window(b, s, win) if shown and opts.get("probe", "1") != "0" else b.observe(win, s._blank(win))
        b.homeostasis_update(rates)
        if shown:
            report(b, rates, f"окно {w + 1} из {windows}")

    out = opts.get("out", "lgn-a")
    path = BRAIN_HOME / "fits" / f"{out}-homeo.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(b.homeo_shift.cpu(), path)
    print(f"\nсдвиги записаны в {path}")


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
