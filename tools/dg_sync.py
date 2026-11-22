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

    PYTHONPATH=. python tools/dg_sync.py [seed=42] [adapt=10] [record=4] [cut=ca1->entorhinal_vb]
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

    # Проводка: вероятность связи у нас (TypedCircuit._p, до 1) = доля
    # популяции-источника у клетки = доля общих источников у пары (случайная).
    from brain import circuit as circuit_mod

    net = circuit_mod.load(b.cfg.areas[[a.name for a in b.cfg.areas].index("dentate")].circuit)
    print(f"зерно {cfg.seed}: гранулярных {n_gc}, мшистых {c.size['DG_Mossy']}", flush=True)
    if not c.topo:
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
    ec = torch.zeros(bins, device=c.device)
    ec_e = b.areas["entorhinal"].E.slice
    zones = {"CA3": b.areas["ca3"].E.slice, "CA1": b.areas["ca1"].E.slice,
             "EC Vb": b.areas["entorhinal_vb"].E.slice}
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
    lost = float(b.dropped) - dropped0
    print(f"потеряно за запись: мозг {lost:.3g}, " + ", ".join(
        f"{k} {float(x.dropped) - cdrop0[k]:.3g}" for k, x in b.circuits.items()))


if __name__ == "__main__":
    main(dict(a.split("=", 1) for a in sys.argv[1:]))
