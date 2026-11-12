"""
Откуда залп перед спайком клетки колонки (DESIGN 2026-11-12: у V1 L2/3
залп g_E за 2 мс до спайка, на 84% из синхронных групп L4).

На пустом экране у traced клеток target (у ямки) вход раскладывается по
источникам — путям, по популяции пресинаптических клеток: возбуждающие в
g_E, тормозные в g_I. Вклад источника — отфильтрованная сумма весов
выстреливших входов, как в движке (ступенька тока w · τ_m/τ_syn, спад τ_syn
своего знака, задержка пути), в долях g_L: g_E ÷ (E_AMPA − V_замера EPSP),
g_I ÷ (V_замера IPSP − E_ГАМК). Кратковременная пластичность, ACh и
усиление торможения не учтены — их и внешний фон даёт остаток: измеренное
− Σ источников − постоянная часть.

Ещё: число спайков входов каждого источника за 2 / 5 / 10 мс до спайка,
разных клеток главного источника (focus — по умолчанию самый большой
вклад в залп), частота всей зоны target и focus вокруг спайка, корреляция
счёта входов у соседних traced клеток и — для пар traced клеток,
выстреливших в пределах ±1 мс, — сколько выстреливших за 5 мс входов
focus у них общие (против той же пары в случайный момент).

    PYTHONPATH=. .venv/bin/python tools/burst_sources.py [target=visual.L23E] [focus=…] [seconds=10] [seed=1] [traced=300] [tonic=…]
"""

from __future__ import annotations

import random
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
    area, rest = opts.get("target", "visual.L23E").split(".")
    layer, kind = (rest[:-1], "E") if rest.endswith("E") else (rest[:-2], rest[-2:])
    target = b.areas[area].layers[layer][kind]
    n_tr = int(opts.get("traced", 300))
    cells = (b.field_cols[target.name] - cfg.fovea_col).abs().argsort()[:n_tr].to(b.device)
    e_span = cfg.ampa_reversal_mv - cfg.epsp_holding_mv
    i_span = cfg.ipsp_holding_mv - cfg.gaba_reversal_mv

    owner = {}
    for a in b.areas.values():
        for _, pop in a.populations:
            owner[(pop.start, pop.stop)] = pop.name

    def source_of(pre: torch.Tensor) -> str:
        i = int(pre.flatten()[0])
        for (lo, hi), name in owner.items():
            if lo <= i < hi:
                return name + (" (соседи)" if name == target.name else "")
        return "?"

    # (источник, знак, пре [n_tr, k], вес [n_tr, k], задержка)
    paths = []
    for p in b.projections:
        if p.source != "net" or p.post is not target:
            continue
        w = p.weight[cells].float()
        sign = 1 if float(w.mean()) > 0 else -1
        paths.append((source_of(p.pre), sign, p.pre[cells].long(), w.abs(), b._path_delay_steps(p)))
    names = sorted({(x[0], x[1]) for x in paths}, key=lambda x: (-x[1], x[0]))
    print("пути к", target.name, ":", ", ".join(f"{src}{'' if sg > 0 else ' (торм.)'} k={pre.shape[1]} w={float(w.mean()):.3f}"
                                              for src, sg, pre, w, _ in paths))

    s.line, s.gaze = "", 0.0
    b.run(2500.0, s._blank(2500.0))
    steps = int(float(opts.get("seconds", 10)) * 1000 / dt)
    S = len(names)
    tau_e = float(b.tau_syn[target.start])
    tau_i = float(b.tau_syn_inh[target.start])
    tau_m = float(b.tau_m[target.start])
    decay = torch.tensor([torch.exp(torch.tensor(-dt / (tau_e if sg > 0 else tau_i))).item() for _, sg in names], device=b.device)[:, None]
    gain = torch.tensor([tau_m / (tau_e if sg > 0 else tau_i) / (e_span if sg > 0 else i_span) for _, sg in names], device=b.device)[:, None]
    max_delay = max(x[4] for x in paths) + 1
    ring = torch.zeros(max_delay + 1, S, n_tr, device=b.device)
    g_src = torch.zeros(S, n_tr, device=b.device)
    rec_g = torch.zeros(steps, S, n_tr, device=b.device)
    rec_n = torch.zeros(steps, S, n_tr, device=b.device)
    rec_ge = torch.zeros(steps, n_tr, device=b.device)
    rec_gi = torch.zeros(steps, n_tr, device=b.device)
    rec_sp = torch.zeros(steps, n_tr, device=b.device)
    # Главный источник: задан или — после прогона по STA; пока пишем «кто стрелял» у всех E-путей (uint8).
    e_paths = [i for i, x in enumerate(paths) if x[1] > 0]
    fired_rec = {i: torch.zeros(steps, n_tr, paths[i][2].shape[1], dtype=torch.uint8, device=b.device) for i in e_paths}
    src_pops = {}
    for src, *_ in paths:
        base = src.replace(" (соседи)", "")
        for (lo, hi), name in owner.items():
            if name == base:
                src_pops[src] = slice(lo, hi)
    pop_rate = {k: torch.zeros(steps, device=b.device) for k in [target.name, *src_pops]}
    view = s._blank(steps * dt)
    t0 = time.time()
    for t in range(steps):
        b.run(dt, view[t:t + 1])
        sp = b.spikes.float()
        slot = t % (max_delay + 1)
        for i, (src, sg, pre, w, d) in enumerate(paths):
            j = names.index((src, sg))
            fired = sp[pre]
            rec_n[t, j] += fired.sum(1)
            ring[(t + d) % (max_delay + 1), j] += (fired * w).sum(1)
            if i in fired_rec:
                fired_rec[i][t] = fired.to(torch.uint8)
        g_src.mul_(decay).add_(ring[slot] * gain)
        ring[slot].zero_()
        rec_g[t] = g_src
        rec_ge[t] = (b.i_syn[target.slice][cells] + b.last_dc_e[target.slice][cells]) / e_span
        rec_gi[t] = (b.i_inh[target.slice][cells] + b.i_ngf[target.slice][cells] + b.last_dc_i[target.slice][cells]).abs() / i_span
        rec_sp[t] = sp[target.slice][cells]
        pop_rate[target.name][t] = sp[target.slice].mean() / (dt / 1000)
        for k, sl in src_pops.items():
            pop_rate[k][t] = sp[sl].mean() / (dt / 1000)
    print(f"пустой экран {steps * dt / 1000:.1f} с, {time.time() - t0:.0f} с", flush=True)

    is_e = torch.tensor([sg > 0 for _, sg in names], device=b.device)
    mean_src = rec_g.mean((0, 2))
    dc_e = float((b.last_dc_e[target.slice][cells] / e_span).mean())
    dc_i = float((b.last_dc_i[target.slice][cells].abs() / i_span).mean())
    for label, meas, mask, dc in (("g_E", rec_ge, is_e, dc_e), ("g_I", rec_gi, ~is_e, dc_i)):
        parts = ", ".join(f"{n[0]} {float(m):.3f}" for n, m, k in zip(names, mean_src, mask) if k)
        print(f"среднее {label} / g_L: измерено {float(meas.mean()):.3f}; источники {parts}; постоянная часть {dc:.3f}; "
              f"остаток {float(meas.mean() - mean_src[mask].sum() - dc):.3f}")

    win, quiet = int(50 / dt), int(100 / dt)
    all_sp = rec_sp.nonzero().tolist()
    last = [-10 ** 9] * n_tr
    picks = []
    for t_s, c in all_sp:
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
    sta_src = (rec_g[tt, :, C[:, None]] - g_mean.T[C][:, None, :]).mean(0)  # [win, S]
    sta_ge = (rec_ge[tt, C[:, None]] - rec_ge.mean(0)[C][:, None]).mean(0)
    sta_gi = (rec_gi[tt, C[:, None]] - rec_gi.mean(0)[C][:, None]).mean(0)
    pts = [20, 10, 5, 2, 1]
    at = lambda x, ms: float(x[win - int(ms / dt)])
    print(f"\nSTA за {'/'.join(map(str, pts))} мс до спайка ({len(picks)} спайков после ≥100 мс тишины), Δ g / g_L от среднего клетки:")
    for label, meas, mask in (("g_E", sta_ge, is_e), ("g_I", sta_gi, ~is_e)):
        print(f"  {label} измерено        " + " ".join(f"{at(meas, ms):+.3f}" for ms in pts))
        total = torch.zeros(win, device=b.device)
        for j, n in enumerate(names):
            if bool(mask[j]):
                total += sta_src[:, j]
                print(f"    {n[0]:20s} " + " ".join(f"{at(sta_src[:, j], ms):+.3f}" for ms in pts))
        print(f"    {'остаток':20s} " + " ".join(f"{at(meas - total, ms):+.3f}" for ms in pts))
    peak = int(sta_ge[-int(2 / dt):].argmax()) + win - int(2 / dt)
    shares = {n[0]: float(sta_src[peak, j]) / max(float(sta_ge[peak]), 1e-9) for j, n in enumerate(names) if bool(is_e[j])}
    print(f"  доля в приросте g_E на пике ({(win - peak) * dt:.1f} мс до спайка): "
          + ", ".join(f"{n} {v:.0%}" for n, v in shares.items()))

    n_mean = rec_n.mean(0)
    print("\nспайков входов за окно до спайка (на клетку; в скобках — ожидание по среднему):")
    for j, n in enumerate(names):
        parts = []
        for ms in (2, 5, 10):
            k = int(ms / dt)
            got = float(rec_n[T[:, None] + torch.arange(-k, 0, device=b.device)[None, :], j, C[:, None]].sum(1).mean())
            parts.append(f"{ms} мс {got:.1f} ({float(n_mean[j].mean()) * k:.1f})")
        print(f"  {n[0]:20s}{'' if n[1] > 0 else ' (торм.)'} " + ", ".join(parts))

    focus_name = opts.get("focus") or max(shares, key=shares.get)
    focus = [i for i in e_paths if paths[i][0] == focus_name]
    print(f"\nглавный источник: {focus_name}")
    if focus:
        fr = fired_rec[focus[0]]
        pre = paths[focus[0]][2]
        for ms in (2, 5, 10):
            k = int(ms / dt)
            got = fr[T[:, None] + torch.arange(-k, 0, device=b.device)[None, :], C[:, None]]
            print(f"  за {ms} мс: спайков входов {float(got.sum((1, 2)).float().mean()):.1f}, разных клеток {float((got.sum(1) > 0).sum(1).float().mean()):.1f}")

        # Пары traced клеток, выстреливших в пределах ±1 мс: общие выстрелившие входы focus.
        k5, near = int(5 / dt), int(1 / dt)
        by_t = {}
        for t_s, c in all_sp:
            by_t.setdefault(t_s, []).append(c)
        rng = random.Random(seed)
        pairs = []
        for t_s, c in picks:
            for t2 in range(t_s - near, t_s + near + 1):
                for c2 in by_t.get(t2, []):
                    if c2 != c and t2 >= k5:
                        pairs.append((t_s, c, t2, c2))
        rng.shuffle(pairs)
        pairs = pairs[:3000]

        def common(t1, c1, t2, c2):
            a = (fr[t1 - k5:t1, c1].sum(0) > 0)
            bb = (fr[t2 - k5:t2, c2].sum(0) > 0)
            ida = set(pre[c1][a].tolist())
            idb = set(pre[c2][bb].tolist())
            return len(ida & idb), len(ida), len(idb)

        if pairs:
            sync, ctrl, struct = [], [], []
            for t1, c1, t2, c2 in pairs:
                sync.append(common(t1, c1, t2, c2)[0])
                shift = rng.randrange(int(100 / dt), int(1000 / dt))
                u1, u2 = t1 - shift, t2 - shift
                if u1 >= k5 and u2 >= k5:
                    ctrl.append(common(u1, c1, u2, c2)[0])
                struct.append(len(set(pre[c1].tolist()) & set(pre[c2].tolist())))
            m = lambda x: sum(x) / max(len(x), 1)
            print(f"  пары traced клеток, выстреливших в ±1 мс: {len(pairs)}; общих входов по проводке {m(struct):.1f} из {pre.shape[1]}; "
                  f"общих ВЫСТРЕЛИВШИХ за 5 мс {m(sync):.2f} (та же пара в случайный момент {m(ctrl):.2f})")

    print()
    for k, rate in pop_rate.items():
        sta = rate[tt].mean(0)
        print(f"  {k} вся зона: частота за {'/'.join(map(str, pts))} мс до спайка "
              + " ".join(f"{at(sta, ms):.2f}" for ms in pts) + f" Гц (среднее {float(rate.mean()):.2f})")
    k = int(5 / dt)
    for j, n in enumerate(names):
        cnt = rec_n[: steps // k * k, j].view(-1, k, n_tr).sum(1)
        ok = cnt.std(0) > 0
        if int(ok.sum()) < 2:
            continue
        cc = torch.corrcoef(cnt[:, ok].T)
        off = cc[~torch.eye(int(ok.sum()), dtype=torch.bool, device=cc.device)]
        print(f"  корреляция счёта входов «{n[0]}» за 5 мс у traced клеток: медиана {float(off.nanmedian()):+.3f}")


if __name__ == "__main__":
    main(dict(a.split("=", 1) for a in sys.argv[1:]))
