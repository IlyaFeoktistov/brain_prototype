"""
Откуда залп возбуждения перед спайком V1 L2/3 (DESIGN 2026-11-12, после
silence_check: g_E +0.65 g_L за 2 мс до спайка).

На пустом экране у traced клеток (у ямки) возбуждающий вход раскладывается
по источникам — путям, по популяции пресинаптических клеток (L4 E, L2/3 E —
соседи, прочие зоны). Вклад источника в g_E: отфильтрованная сумма весов
выстреливших входов, как в движке (ступенька тока w · τ_m/τ_syn, спад τ_syn,
задержка пути), в долях g_L (÷ (E_AMPA − V_замера EPSP)). Кратковременная
пластичность и ACh-подавление не учтены — их и внешний фон даёт остаток:
измеренный g_E − Σ источников − постоянная часть. Ещё: STA числа спайков
каждого источника за 2 / 5 / 10 мс до спайка (на клетку-получатель),
STA частоты всей популяции L2/3 E и L4 E (общий залп или местный) и
корреляция счёта входов источника у соседних traced клеток.

    PYTHONPATH=. .venv/bin/python tools/burst_sources.py [seconds=10] [seed=1] [traced=300] [tonic=…] [bg_scale=…]
"""

from __future__ import annotations

import sys
import time

import torch

from brain.engine import SpikingBrain
from brain.session import Session

SpikingBrain.CHUNK = 1

from tools.reading_bench import build  # noqa: E402


def main(opts: dict) -> None:
    seed = int(opts.get("seed", 1))
    opts = {"fit": "lgn-a", "homeo": f"lgn-d-pop-s{seed}", "relay_v_t_mv": "-59", "trn_scale": "0.1", **opts}
    mind, b = build(opts)
    s = Session(mind, window_ms=10.0)
    cfg = b.cfg
    dt = cfg.dt_ms
    target = b.areas["visual"].layers["L23"]["E"]
    l4 = b.areas["visual"].layers["L4"]["E"]
    n_tr = int(opts.get("traced", 300))
    cells = (b.field_cols[target.name] - cfg.fovea_col).abs().argsort()[:n_tr].to(b.device)
    e_span = cfg.ampa_reversal_mv - cfg.epsp_holding_mv

    # Возбуждающие пути к target: по источнику (имя популяции пре).
    owner = {}
    for area in b.areas.values():
        for _, pop in area.populations:
            owner[(pop.start, pop.stop)] = pop.name

    def source_of(pre: torch.Tensor) -> str:
        i = int(pre.flatten()[0])
        for (a, z), name in owner.items():
            if a <= i < z:
                return name
        return "?"

    paths = []
    for p in b.projections:
        if p.source != "net" or p.post is not target:
            continue
        w = p.weight[cells]
        if float(w.mean()) <= 0:
            continue
        src = source_of(p.pre)
        if src == target.name:
            src = "L2/3 E (соседи)"
        elif src == l4.name:
            src = "L4 E"
        paths.append((src, p.pre[cells].long(), w.float(), b._path_delay_steps(p)))
    names = sorted({x[0] for x in paths})
    print("пути к", target.name, ":", ", ".join(f"{src} k={pre.shape[1]} w={float(w.mean()):.3f}" for src, pre, w, _ in paths))

    s.line, s.gaze = "", 0.0
    b.run(2500.0, s._blank(2500.0))
    steps = int(float(opts.get("seconds", 10)) * 1000 / dt)
    S = len(names)
    tau_syn = float(b.tau_syn[target.start])
    scale = float(b.tau_m[target.start]) / tau_syn
    decay = torch.exp(torch.tensor(-dt / tau_syn))
    max_delay = max(d for *_, d in paths) + 1
    ring = torch.zeros(max_delay + 1, S, n_tr, device=b.device)
    g_src = torch.zeros(S, n_tr, device=b.device)
    rec_g = torch.zeros(steps, S, n_tr, device=b.device)
    rec_n = torch.zeros(steps, S, n_tr, device=b.device)
    rec_ge = torch.zeros(steps, n_tr, device=b.device)
    rec_sp = torch.zeros(steps, n_tr, device=b.device)
    pop_rate = torch.zeros(steps, 2, device=b.device)
    # Кто из входов L4 стрелял (uint8): разные клетки или пачка одной.
    l4_paths = [i for i, x in enumerate(paths) if x[0] == "L4 E"]
    l4_fired = torch.zeros(steps, n_tr, paths[l4_paths[0]][1].shape[1], dtype=torch.uint8, device=b.device) if l4_paths else None
    view = s._blank(steps * dt)
    t0 = time.time()
    for t in range(steps):
        b.run(dt, view[t:t + 1])
        sp = b.spikes.float()
        slot = t % (max_delay + 1)
        for src, pre, w, d in paths:
            j = names.index(src)
            fired = sp[pre]
            rec_n[t, j] += fired.sum(1)
            if l4_fired is not None and src == "L4 E":
                l4_fired[t] = fired.to(torch.uint8)
            ring[(t + d) % (max_delay + 1), j] += (fired * w).sum(1)
        # Пришедшее сейчас — в ток источника (как i_syn: скачок, спад τ_syn).
        g_src.mul_(decay).add_(ring[slot] * scale / e_span)
        ring[slot].zero_()
        rec_g[t] = g_src
        rec_ge[t] = (b.i_syn[target.slice][cells] + b.last_dc_e[target.slice][cells]) / e_span
        rec_sp[t] = sp[target.slice][cells]
        pop_rate[t, 0] = sp[target.slice].mean() / (dt / 1000)
        pop_rate[t, 1] = sp[l4.slice].mean() / (dt / 1000)
    print(f"пустой экран {steps * dt / 1000:.1f} с, {time.time() - t0:.0f} с", flush=True)

    # Сверка разложения: средние за всё время.
    dc = float((b.last_dc_e[target.slice][cells] / e_span).mean())
    mean_src = rec_g.mean((0, 2))
    resid = rec_ge.mean() - mean_src.sum() - dc
    print("среднее g_E / g_L: измерено", f"{float(rec_ge.mean()):.3f}", "; источники",
          ", ".join(f"{n} {float(m):.3f}" for n, m in zip(names, mean_src)),
          f"; постоянная часть {dc:.3f}; остаток (внешний фон, STP, ACh) {float(resid):.3f}")

    # STA (спайки после ≥ 100 мс тишины).
    win, quiet = int(50 / dt), int(100 / dt)
    idx = rec_sp.nonzero().tolist()
    last = [-10 ** 9] * n_tr
    picks = []
    for t_s, c in idx:
        if t_s - last[c] >= quiet and t_s >= win:
            picks.append((t_s, c))
        last[c] = t_s
    if not picks:
        print("нет спайков после тишины")
        return
    T = torch.tensor([p[0] for p in picks], device=b.device)
    C = torch.tensor([p[1] for p in picks], device=b.device)
    lags = torch.arange(-win, 0, device=b.device)
    tt = T[:, None] + lags[None, :]
    g_mean = rec_g.mean(0)
    ge_mean = rec_ge.mean(0)
    sta_src = (rec_g[tt, :, C[:, None]] - g_mean.T[C][:, None, :]).mean(0)  # [win, S]
    sta_ge = (rec_ge[tt, C[:, None]] - ge_mean[C][:, None]).mean(0)
    pts = [20, 10, 5, 2, 1]
    at = lambda x, ms: float(x[win - int(ms / dt)])
    print(f"\nSTA за {'/'.join(map(str, pts))} мс до спайка ({len(picks)} спайков после ≥100 мс тишины), Δ g / g_L от среднего клетки:")
    print("  измерено g_E      " + " ".join(f"{at(sta_ge, ms):+.3f}" for ms in pts))
    total = torch.zeros(win, device=b.device)
    for j, n in enumerate(names):
        total += sta_src[:, j]
        print(f"  {n:18s} " + " ".join(f"{at(sta_src[:, j], ms):+.3f}" for ms in pts))
    rest = sta_ge - total
    print("  остаток (фон, STP) " + " ".join(f"{at(rest, ms):+.3f}" for ms in pts))
    peak = int(sta_ge[-int(2 / dt):].argmax()) + win - int(2 / dt)
    share = {n: float(sta_src[peak, j]) / max(float(sta_ge[peak]), 1e-9) for j, n in enumerate(names)}
    print("  доля в приросте на пике (" + f"{(win - peak) * dt:.1f} мс до спайка): "
          + ", ".join(f"{n} {v:.0%}" for n, v in share.items()) + f", остаток {float(rest[peak]) / max(float(sta_ge[peak]), 1e-9):.0%}")

    # Число спайков входов за окно до спайка против среднего.
    n_mean = rec_n.mean(0)
    print("\nспайков входов за окно до спайка (на клетку; в скобках — ожидание по среднему):")
    for j, n in enumerate(names):
        parts = []
        for ms in (2, 5, 10):
            k = int(ms / dt)
            got = float(rec_n[T[:, None] + torch.arange(-k, 0, device=b.device)[None, :], j, C[:, None]].sum(1).mean())
            exp = float(n_mean[j].mean()) * k
            parts.append(f"{ms} мс {got:.1f} ({exp:.1f})")
        print(f"  {n:18s} " + ", ".join(parts))

    if l4_fired is not None:
        for ms in (2, 5, 10):
            k = int(ms / dt)
            got = l4_fired[T[:, None] + torch.arange(-k, 0, device=b.device)[None, :], C[:, None]]  # [spk, k, inputs]
            distinct = (got.sum(1) > 0).sum(1).float().mean()
            spikes = got.sum((1, 2)).float().mean()
            print(f"  L4 E за {ms} мс: спайков {float(spikes):.1f} от разных клеток {float(distinct):.1f}")
    # Общий ли залп: частота всей популяции вокруг спайка.
    base = pop_rate.mean(0)
    for i, label in enumerate(("L2/3 E вся зона", "L4 E вся зона")):
        sta = pop_rate[tt, i].mean(0)
        print(f"  {label}: частота за {'/'.join(map(str, pts))} мс до спайка "
              + " ".join(f"{at(sta, ms):.2f}" for ms in pts) + f" Гц (среднее {float(base[i]):.2f})")

    # Корреляция счёта входов источника (окна 5 мс) у соседних traced клеток.
    k = int(5 / dt)
    for j, n in enumerate(names):
        cnt = rec_n[: steps // k * k, j].view(-1, k, n_tr).sum(1)
        ok = cnt.std(0) > 0
        cc = torch.corrcoef(cnt[:, ok].T)
        off = cc[~torch.eye(int(ok.sum()), dtype=torch.bool, device=cc.device)]
        print(f"  корреляция счёта входов «{n}» за 5 мс у traced клеток: медиана {float(off.nanmedian()):+.3f}")


if __name__ == "__main__":
    main(dict(a.split("=", 1) for a in sys.argv[1:]))
