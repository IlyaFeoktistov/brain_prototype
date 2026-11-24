"""
Синхронность зубчатой (CHAT_PLAN шаг 3, ISSUES 34): идут ли гранулярные
независимо или общими залпами. Мозг — как в опыте B (tools/develop.py,
organs=words): 2.5 с разгона, адаптация с гомеостазом окнами по 1 с, затем
запись без гомеостаза окнами по 5 мс (один кусок шага движка).

Меры:
  - проводка: сколько мшистых у гранулярной (доля популяции), доля общих
    мшистых у пары гранулярных;
  - частоты по типам сети;
  - популяционная частота гранулярных по окнам 5 мс: коэффициент вариации
    (у независимых пуассоновских клеток ≈ 1/√(N·r·0.005)), доля окон-залпов
    (> 5× средней), доля гранулярных в залпе;
  - корреляция Пирсона счётов в окнах 5 и 20 мс по парам гранулярных
    (выборка; у независимых — ≈ 0);
  - поток событий DG → CA3 (спайки гранулярных × синапсов пути на клетку),
    потерянные события мозга и сетей.

  - порядок волн по зонам: сдвиг наибольшей корреляции частот популяций
    (окна 5 мс) CA3, CA1, EC Vb, гранулярных относительно EC II/III;
    корреляция пар клеток EC.

smooth=src->dst — тот же средний вход пути без колебаний (построенные →
среднее поле). cut=src->dst[,src->dst] — диагностическая перерезка (не механизм): нули
построенным синапсам пути (имя «src -> dst…») и его среднему полю
(недостающим входам, mf_exc/mf_inh/mf_ach пары популяций). Гомеостаз и
силы не меняются.

    PYTHONPATH=. python tools/dg_sync.py [seed=42] [adapt=10] [record=4] [cut=ca1->entorhinal_vb] [save_ec=путь.pt] [raster=файл.png] [raster_ms=1000] [drop=имя пути] [diag=1] [diag_ms=2000]
"""

from __future__ import annotations

import sys

import torch

from brain.config import BrainConfig
from brain.mind import Mind
from tools import develop


def main(opts: dict) -> None:
    if opts.get("drop"):
        # drop=имя пути[;имя] — путь не строится (контроль правки в той же
        # серии), например drop="entorhinal -> dentate basket".
        from brain import anatomy

        names = set(opts["drop"].split(";"))
        built = anatomy.pathways
        anatomy.pathways = lambda c: [p for p in built(c) if (p.name or f"{p.src} -> {p.dst}") not in names]
        print("без путей:", ", ".join(sorted(names)), flush=True)
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

    for pair in (x for x in opts.get("cut", "").split(",") if x):
        src, dst = pair.split("->")
        hit = [p for p in b.projections if p.name.startswith(f"{src} -> {dst}")]
        for p in hit:
            p.weight.zero_()
        mf = 0
        if b.mf_on:
            q = b.mf_post_id[b.areas[dst].cells].unique()
            i = b.mf_pre_id[b.areas[src].cells].unique()
            for m in (b.mf_exc, b.mf_inh, b.mf_ach):
                block = m[q[:, None], i[None, :]]
                mf += int((block != 0).sum())
                m[q[:, None], i[None, :]] = 0.0
        print(f"перерезан {src} → {dst}: путей {len(hit)}, членов среднего поля {mf}", flush=True)

    for pair in (x for x in opts.get("smooth", "").split(",") if x):
        # Тот же средний вход пути, без колебаний: построенные синапсы → 0, их
        # средний ток (Σw × частота × τm) — в среднее поле (сглажено
        # mean_field_tau_ms). Диагностика, не механизм.
        src, dst = pair.split("->")
        i = int(b.mf_pre_id[b.areas[src].E.start])
        for p in [p for p in b.projections if p.name in (f"{src} -> {dst}", f"{src} -> {dst}.I")]:
            post = p.post
            q = int(b.mf_post_id[post.start])
            coeff = float(p.weight.float().sum(1).mean()) * float(b.tau_m[post.start])
            b.mf_exc[q, i] += coeff
            p.weight.zero_()
            print(f"сглажен {p.name}: построенные → поле, +{coeff:.1f} мВ/кГц", flush=True)

    c = b.circuits["dentate"]
    gc = c.type_of == 0
    cells = b.areas["dentate"].cells
    n_gc = int(gc.sum())

    from brain import circuit as circuit_mod

    net = circuit_mod.load(b.cfg.areas[[a.name for a in b.cfg.areas].index("dentate")].circuit)
    print(f"зерно {cfg.seed}: гранулярных {n_gc}, типы " + ", ".join(f"{g} {c.size[g]}" for g in c.names), flush=True)
    if net.get("model") == "chavlis":
        print(f"   сеть Chavlis 2017: синапсов {c.synapses}, входных клеток EC {len(c.ec_cells)}", flush=True)
    elif not c.topo:
        # Вероятность связи у нас (TypedCircuit._p, до 1) = доля популяции-
        # источника у клетки = доля общих источников у пары (случайная).
        for link in net["links"]:
            p = c._p(link, net["groups"], c.size)
            if p > 0.2:
                print(f"   {link['pre']} → {link['post']}: у клетки {min(1.0, p):.0%} источников"
                      + (f" (по данным нужно {p:.2f}× популяции — сила × {p:.2f})" if p > 1 else "")
                      + f"; у крысы {link['p']:.2%}")
    else:
        print(f"   топография: построено {c.synapses} синапсов, недостающих входов в среднем на слот "
              f"{float(c.g_missing.mean()):.1f}")

    b.run(2500.0, s._blank(2500.0))
    for _ in range(int(opts.get("adapt", 10))):
        b.homeostasis_update(b.observe(1000.0, s._blank(1000.0)))
    b.cfg.homeostasis = False

    dropped0 = float(b.dropped)
    cdrop0 = {k: float(x.dropped) for k, x in b.circuits.items()}
    bins = int(float(opts.get("record", 4)) * 1000 / 5)
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
    sample = torch.randperm(n_gc, generator=gen)[:500].to(c.device)
    counts = torch.zeros(bins, 500, device=c.device)
    pop = torch.zeros(bins, len(c.names), device=c.device)
    ca3 = torch.zeros(bins, device=c.device)
    ca3_frac = torch.zeros(bins, device=c.device)
    gc_frac = torch.zeros(bins, device=c.device)
    ec = torch.zeros(bins, device=c.device)
    ec_e = b.areas["entorhinal"].E.slice
    zones = {"CA3": b.areas["ca3"].E.slice, "CA1": b.areas["ca1"].E.slice,
             "EC Vb": b.areas["entorhinal_vb"].E.slice}
    # Корковые входы EC (association, prefrontal → entorhinal): слои L2/3 и L5.
    for zone in ("association", "prefrontal"):
        for layer in ("L23", "L5"):
            layer_pop = b.areas[zone].layers[layer]["E"]
            zones[f"{zone[:5]} {layer}"] = layer_pop.slice
    series = {k: torch.zeros(bins, device=c.device) for k in zones}
    ec_n = ec_e.stop - ec_e.start
    ec_sample = torch.randperm(ec_n, generator=gen)[:500].to(c.device) + ec_e.start
    ec_counts = torch.zeros(bins, 500, device=c.device)
    ca3_e = b.areas["ca3"].E.slice

    for t in range(bins):
        r = b.observe(5.0, s._blank(5.0))
        x = r[cells] * 0.005
        counts[t] = x[gc][sample]
        for i in range(len(c.names)):
            pop[t, i] = x[c.type_of == i].sum()
        ca3[t] = r[ca3_e].mean()
        ca3_frac[t] = (r[ca3_e] > 0).float().mean()
        gc_frac[t] = (x[gc] > 0).float().mean()
        ec[t] = r[ec_e].mean()
        for k, sl in zones.items():
            series[k][t] = r[sl].mean()
        ec_counts[t] = r[ec_sample] * 0.005

    sec = bins * 0.005
    print("частоты по типам, Гц:", "  ".join(f"{g} {float(pop[:, i].sum()) / c.size[g] / sec:.2f}" for i, g in enumerate(c.names)))

    g_pop = pop[:, 0]
    mean = float(g_pop.mean())
    cv = float(g_pop.std() / g_pop.mean()) if mean > 0 else float("nan")
    rate = mean / n_gc / 0.005
    cv_poisson = 1.0 / max(1e-9, (n_gc * rate * 0.005)) ** 0.5
    bursts = g_pop > 5 * mean
    part = float((g_pop[bursts] / n_gc).mean()) if bool(bursts.any()) else 0.0
    print(f"гранулярные по окнам 5 мс: в среднем {mean:.1f} спайков, CV {cv:.2f} (независимые — {cv_poisson:.2f}); "
          f"залпов (> 5× среднего) {float(bursts.float().mean()):.1%} окон, в залпе {part:.1%} гранулярных")
    top = torch.topk(g_pop, 5).values.tolist()
    print("  самые большие окна:", [int(v) for v in top], f"(из {n_gc})")

    for w in (1, 4):
        x = counts[: bins // w * w].view(bins // w, w, 500).sum(1)
        x = x - x.mean(0)
        sd = x.std(0)
        live = sd > 0
        x = x[:, live] / sd[live]
        cc = (x.T @ x) / (x.shape[0] - 1)
        n = cc.shape[0]
        off = cc[~torch.eye(n, dtype=torch.bool, device=cc.device)]
        print(f"  корреляция пар гранулярных, окно {5 * w} мс: средняя {float(off.mean()):.3f}, "
              f"медиана {float(off.median()):.3f} (пар {n * (n - 1) // 2}, без молчащих)")

    if opts.get("save_ec"):
        # Частота EC II/III по окнам 5 мс — для tools/ca3_margin.py trace=.
        torch.save(ec.cpu(), opts["save_ec"])
    e_cv = float(ec.std() / ec.mean())
    a, z = ec - ec.mean(), g_pop - g_pop.mean()
    lags = range(-10, 11)
    cc = [float((a[max(0, -k): bins - max(0, k)] * z[max(0, k): bins - max(0, -k)]).mean() / (a.std() * z.std())) for k in lags]
    best = max(lags, key=lambda k: cc[k + 10])
    print(f"EC по окнам 5 мс: {float(ec.mean()):.2f} Гц, CV {e_cv:.2f}; связь EC → гранулярные: наибольшая корреляция "
          f"{cc[best + 10]:.2f} при сдвиге {best * 5:+d} мс (плюс — гранулярные после EC)")
    def lag(y):
        a2, z2 = ec - ec.mean(), y - y.mean()
        if float(z2.std()) == 0:
            return 0.0, 0
        cc2 = [float((a2[max(0, -k): bins - max(0, k)] * z2[max(0, k): bins - max(0, -k)]).mean()
                     / (a2.std() * z2.std())) for k in range(-20, 21)]
        k = max(range(-20, 21), key=lambda j: cc2[j + 20])
        return cc2[k + 20], k * 5

    parts = []
    for k, y in [*series.items(), ("гранулярные", g_pop)]:
        r_, l_ = lag(y)
        cv_ = float(y.std() / y.mean()) if float(y.mean()) > 0 else float("nan")
        parts.append(f"{k} {float(y.mean()) if k != 'гранулярные' else rate:.2f} Гц CV {cv_:.2f} r {r_:.2f} {l_:+d} мс")
    print("порядок относительно EC (плюс — позже EC):", " | ".join(parts))
    x = ec_counts - ec_counts.mean(0)
    sd = x.std(0)
    live = sd > 0
    x = x[:, live] / sd[live]
    cc3 = (x.T @ x) / (x.shape[0] - 1)
    off3 = cc3[~torch.eye(cc3.shape[0], dtype=torch.bool, device=cc3.device)]
    print(f"корреляция пар EC, окно 5 мс: средняя {float(off3.mean()):.3f} (клеток {int(live.sum())})")
    syn_out = sum(int(p.weight.numel()) for p in b.projections if p.name.startswith("dentate -> ca3"))
    out_per_gc = syn_out / n_gc
    events = float(g_pop.sum()) * out_per_gc / sec
    print(f"поток DG → CA3: {out_per_gc:.0f} синапсов на гранулярную, {events:.3g} событий/с; "
          f"CA3 PC {float(ca3.mean()):.2f} Гц, CV по окнам {float(ca3.std() / ca3.mean()):.2f}")
    # CA3: доля PC, выстреливших в окне 5 мс. У SWR в событии ~10–20% PC,
    # срыв — 85–99% (CHAT_PLAN шаг 3, цикл 10); массовое — ≥ 50%.
    print(f"CA3 PC по окнам 5 мс: доля в окне средняя {float(ca3_frac.mean()):.2%}, наибольшая {float(ca3_frac.max()):.1%}; "
          f"окон ≥ 20% — {int((ca3_frac >= 0.2).sum())}, массовых (≥ 50%) — {int((ca3_frac >= 0.5).sum())} "
          f"за {sec:.0f} с")
    print(f"гранулярные по окнам 5 мс: доля в окне средняя {float(gc_frac.mean()):.3%}, наибольшая {float(gc_frac.max()):.2%}; "
          f"активных за запись {float((counts.sum(0) > 0).float().mean()):.1%} (выборка 500)")
    lost = float(b.dropped) - dropped0
    print(f"потеряно за запись: мозг {lost:.3g}, " + ", ".join(
        f"{k} {float(x.dropped) - cdrop0[k]:.3g}" for k, x in b.circuits.items()))
    if opts.get("raster"):
        raster(b, s, c, cells, opts["raster"], float(opts.get("raster_ms", 1000)), cfg.seed)
    if opts.get("diag"):
        diagnose(b, s, c, cells, float(opts.get("diag_ms", 2000)), cfg.seed)


def raster(b, s, c, cells, path: str, ms: float, seed: int) -> None:
    """
    Растр после записи: ms мс по 1 мс — по 80 клеток каждого типа DG, EC II/III
    и CA3 PC; внизу — частота популяций (доля клеток типа за 1 мс). Видно, кто
    стреляет первым в залпе (успевают ли корзинчатые раньше гранулярных).
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gen = torch.Generator(device="cpu").manual_seed(seed + 7)
    rows = []
    for i, g in enumerate(c.names):
        idx = torch.nonzero(c.type_of == i).squeeze(1).cpu()
        pick = idx[torch.randperm(len(idx), generator=gen)[:80]]
        rows.append((g.replace("DG_", "DG "), pick.to(b.device) + cells.start, idx.to(b.device) + cells.start))
    for name, area in (("EC II/III", "entorhinal"), ("CA3 PC", "ca3")):
        e = b.areas[area].E
        pick = torch.randperm(e.size, generator=gen)[:80] + e.start
        rows.append((name, pick.to(b.device), torch.arange(e.start, e.stop, device=b.device)))
    steps = int(ms)
    spk = [torch.zeros(steps, len(r[1]), dtype=torch.bool, device=b.device) for r in rows]
    frac = torch.zeros(steps, len(rows), device=b.device)
    for t in range(steps):
        r = b.observe(1.0, s._blank(1.0))
        for j, (_, pick, allc) in enumerate(rows):
            spk[j][t] = r[pick] > 0
            frac[t, j] = (r[allc] > 0).float().mean()
    colors = ["#1f5fbf", "#3a9a3a", "#d0453a", "#a35bc4", "#e08a1e", "#444444"]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    base = 0
    for j, (name, pick, _) in enumerate(rows):
        t_idx, n_idx = torch.nonzero(spk[j], as_tuple=True)
        ax.scatter(t_idx.cpu(), (n_idx + base).cpu(), s=2, color=colors[j % len(colors)], marker="|")
        ax.text(-ms * 0.01, base + len(pick) / 2, name, ha="right", va="center", fontsize=9)
        base += len(pick) + 10
        ax2.plot(frac[:, j].cpu(), color=colors[j % len(colors)], lw=1, label=name)
    ax.set_yticks([])
    ax.set_title(f"зерно {seed}: растр по 80 клеток типа, шаг 1 мс")
    ax2.set_ylabel("доля клеток за 1 мс")
    ax2.set_yscale("symlog", linthresh=1e-3)
    ax2.set_xlabel("мс")
    ax2.legend(fontsize=8, ncol=6, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"растр: {path}")




def diagnose(b, s, c, cells, ms: float, seed: int) -> None:
    """
    diag=1 — режим DG по клеткам (CHAT_PLAN шаг 3, цикл 35), ms мс по 1 мс:
    частоты и доля активных клеток по типам; CV межспайковых интервалов
    гранулярных; вход мозга гранулярным (мВ движка, то, что идёт в сеть):
    разброс среднего по клеткам, σ во времени у клетки и σ общей части
    (среднее по клеткам за 1 мс); потенциал v гранулярных: разброс между
    клетками в тихие окна и за 20 мс до залпа; первый спайк типов в залпе
    (залп — 1 мс, где выстрелило > 2% гранулярных); HIPP — наибольшая
    частота клетки (наименьший ISI) и доля ISI ≤ 1 мс.
    """

    dev = b.device
    n = c.n
    T = int(ms)
    gc = c.type_of == 0
    n_gc = int(gc.sum())
    pick = torch.randperm(n_gc, generator=torch.Generator().manual_seed(seed + 11))[:500].to(dev)
    last_ext = torch.zeros(n, device=dev)
    step = c.step

    def wrapped(ext_mv, fired_prev, *rest):
        last_ext.copy_(ext_mv)
        return step(ext_mv, fired_prev, *rest)

    c.step = wrapped
    b._graphs = {}
    spk = torch.zeros(T, n, dtype=torch.bool, device=dev)
    inp = torch.zeros(T, 500, device=dev)
    inp_mean = torch.zeros(T, device=dev)
    vv = torch.zeros(T, 500, device=dev)
    try:
        for t in range(T):
            r = b.observe(1.0, s._blank(1.0))
            spk[t] = r[cells] > 0
            x = last_ext[:n_gc]
            inp[t] = x[pick]
            inp_mean[t] = x.mean()
            vv[t] = c.v[:n_gc][pick]
    finally:
        del c.step
        b._graphs = {}
    sec = T / 1000.0
    print(f"--- режим DG по клеткам, {T} мс по 1 мс ---")
    for i, g in enumerate(c.names):
        m = c.type_of == i
        x = spk[:, m]
        cnt = x.sum(0).float()
        print(f"{g}: {float(cnt.mean()) / sec:.2f} Гц, активных клеток {float((cnt > 0).float().mean()):.1%}, "
              f"самая частая {float(cnt.max()) / sec:.0f} Гц")
    # CV ISI гранулярных (клетки с ≥ 3 спайками).
    g_spk = spk[:, gc]
    cvs = []
    for j in torch.nonzero(g_spk.sum(0) >= 3).squeeze(1)[:2000].tolist():
        ts = torch.nonzero(g_spk[:, j]).squeeze(1).float()
        isi = ts[1:] - ts[:-1]
        cvs.append(float(isi.std() / isi.mean()))
    if cvs:
        cv = torch.tensor(cvs)
        print(f"CV ISI гранулярных (клеток {len(cvs)}): медиана {float(cv.median()):.2f}, 90% {float(cv.quantile(0.9)):.2f}")
    m_cell = inp.mean(0)
    print(f"вход гранулярным, мВ: среднее {float(m_cell.mean()):.2f}, разброс среднего между клетками σ "
          f"{float(m_cell.std()):.2f}, σ во времени у клетки {float(inp.std(0).mean()):.2f}, σ общей части "
          f"{float(inp_mean.std()):.2f}")
    frac = g_spk.float().mean(1)
    burst = torch.nonzero(frac > 0.02).squeeze(1)
    onsets = [int(t) for k, t in enumerate(burst.tolist()) if k == 0 or t - burst[k - 1] > 20]
    quiet = (frac == 0)
    print(f"залпов (> 2% гранулярных за 1 мс): {len(onsets)}")
    vq = vv[quiet]
    if len(vq):
        print(f"v гранулярных в тихие мс: среднее {float(vq.mean()):.1f} мВ, разброс между клетками σ "
              f"{float(vq.mean(0).std()):.2f}, σ во времени {float(vq.std(0).mean()):.2f}")
    pre = [vv[t - 20:t] for t in onsets if t >= 20]
    if pre:
        vp = torch.stack(pre)
        print(f"v за 20 мс до залпа: среднее {float(vp.mean()):.1f} мВ, разброс между клетками σ "
              f"{float(vp.mean(1).std(1).mean()):.2f}; вход за 20 мс до залпа {float(torch.stack([inp[t - 20:t] for t in onsets if t >= 20]).mean()):.2f} "
              f"против тихих {float(inp[quiet].mean()):.2f}")
    for t0 in onsets[:6]:
        first = []
        for i, g in enumerate(c.names):
            w = spk[max(0, t0 - 30): t0 + 40, c.type_of == i].any(1)
            hit = torch.nonzero(w).squeeze(1)
            first.append(f"{g.replace('DG_', '')} {int(hit[0]) - min(30, t0):+d}" if len(hit) else f"{g.replace('DG_', '')} —")
        print(f"залп {t0} мс, первый спайк типа (мс от начала залпа): " + ", ".join(first))
    h = spk[:, c.type_of == len(c.names) - 1]
    isis = []
    for j in range(min(h.shape[1], 300)):
        ts = torch.nonzero(h[:, j]).squeeze(1)
        if len(ts) > 1:
            isis.append(ts[1:] - ts[:-1])
    if isis:
        a = torch.cat(isis).float()
        print(f"HIPP: наименьший ISI {float(a.min()):.0f} мс, доля ISI ≤ 1 мс {float((a <= 1).float().mean()):.1%}, "
              f"медиана ISI {float(a.median()):.0f} мс")


if __name__ == "__main__":
    main(dict(a.split("=", 1) for a in sys.argv[1:]))
