"""
Ворота базальных ганглиев на пустом экране (DESIGN 2026-11-16 «Почему рука
печатает»: GPi 34 Гц при цели 60, VA держится подобранным током +76 мВ,
залпы VA → рука дают ложные нажатия). Без правки мозга; va_bias — флаг
разбора одной величины (не одновременно с чем-то ещё).

Ориентиры частот (сравнение с диапазоном, не цель): бодрствующие грызуны —
MSN ~0.5, FSI ~15, GPe ~40, STN ~15, SNr/GPi ~60 Гц (сводка Lindahl &
Hellgren Kotaleski 2016, https://doi.org/10.1523/ENEURO.0026-16.2016,
табл. 1); человек — только при болезни Паркинсона (Du et al. 2018,
https://doi.org/10.1016/j.neulet.2018.06.021): STN 44.2 ± 7.6, GPi
80.9 ± 9.6, VL 26.7 ± 5.0 Гц.

Окна 10 мс, sec секунд:
1. Частоты: D1, D2, FSI, GPe, STN, GPi, VA, VLp, TRN моторный, рука L2/3 и L5.
2. Вход GPi и VA по путям: мера поставки Σ PSP × частота источника, у путей
   с депрессией (cfg.nucleus_stp) — × ресурс источника до окна (как отдаёт
   движок), плюс фон bias; среднее и в окне перед провалом GPi (нижние 10%
   частоты) — что меняется перед ослаблением ворот.
3. Нажатия (условие Hand.pressed) в секунду.

    PYTHONPATH=. .venv/bin/python tools/bg_gate.py [sec=20] [seed=1] [va_bias=0]
"""

from __future__ import annotations

import sys
import time

import torch

from brain.session import Session
from tools.reading_bench import build, observe_pre

WIN = 10.0


def main(opts: dict) -> None:
    seed = int(opts.get("seed", 1))
    opts = {"fit": "lgn-a", "homeo": f"lgn-d-pop-tcind-s{seed}", "relay_v_t_mv": "-59", "trn_scale": "0.1", **opts}
    mind, b = build(opts)
    s = Session(mind, window_ms=WIN)
    sec, va_bias = float(opts.get("sec", 20)), float(opts.get("va_bias", 0))
    torch.manual_seed(seed)
    A = b.areas
    if va_bias:
        b.bias[A["thalamus"].E.slice] += va_bias
    pops = {"D1": A["striatum_d1"].E, "D2": A["striatum_d2"].E, "FSI": A["fsi"].E, "GPe": A["gpe"].E,
            "STN": A["stn"].E, "GPi": A["gpi"].E, "VA": A["thalamus"].E, "VLp": A["vl_cb"].E,
            "TRN мот.": A["trn_motor"].E, "рука L2/3": A["hand"].layers["L23"]["E"], "рука L5": s.H}
    stp = {(src, dst) for src, dst, *_ in b.cfg.nucleus_stp}
    targets = {"GPi": A["gpi"].E, "VA": A["thalamus"].E}
    paths = {k: [p for p in b.projections if p.post is P and p.source == "net"] for k, P in targets.items()}
    depl = {p.name: any(p.name.startswith(src + " -> ") and p.name.split(" -> ")[1].startswith(dst) for src, dst in stp)
            for ps in paths.values() for p in ps}
    keys, cells = len(s.keyboard), s.hand.cells
    print(f"va_bias {va_bias:+.0f} мВ (фон VA {float(b.bias[A['thalamus'].E.slice].mean()):.1f}); пути: "
          + "; ".join(f"{k}: " + ", ".join(p.name + (" [депр.]" if depl[p.name] else "") for p in ps) for k, ps in paths.items()),
          flush=True)

    s.line, s.gaze = "", 0.0
    b.run(2500.0, s._blank(2500.0))
    n_win = int(sec * 1000 / WIN)
    blank = s._blank(WIN)
    R, D, press, G = [], [], [], []
    gpi_pop = A["gpi"].E
    t0 = time.time()
    for t in range(n_win):
        r, st = observe_pre(b, WIN, blank)
        R.append(torch.stack([r[P.slice].mean() for P in pops.values()]).cpu())
        G.append((r[gpi_pop.slice] * WIN / 1000).round().cpu())
        row = []
        for k, P in targets.items():
            for p in paths[k]:
                src = r[p.pre.long()]
                if depl[p.name]:
                    src = src * st["x"][p.pre.long()]
                row.append((p.weight.float() * src).sum(1).mean())
            row.append(b.bias[P.slice].float().mean())
        D.append(torch.stack(row).cpu())
        sh = (r[s.H.slice][: keys * cells] > 0).float().view(keys, cells).mean(1)
        press.append(float(sh.max()) >= 0.5)
        if t == 49:
            print(f"  первые 0.5 с: {time.time() - t0:.0f} с; GPi {float(torch.stack(R)[:, 5].mean()):.1f} Гц, "
                  f"VA {float(torch.stack(R)[:, 6].mean()):.1f} Гц", flush=True)
    R, D = torch.stack(R), torch.stack(D)
    press = torch.tensor(press)
    G = torch.stack(G)                                          # [win, клетки GPi] спайков за окно
    print(f"окон {n_win} ({sec:.0f} с), {time.time() - t0:.0f} с\n", flush=True)

    print("1. частоты, Гц (среднее; 10% / 90% по окнам 10 мс сглажено по 50 мс):", flush=True)
    sm = torch.nn.functional.avg_pool1d(R.T[None], 5, 1).squeeze(0).T
    for j, k in enumerate(pops):
        print(f"   {k:10s} {float(R[:, j].mean()):7.2f}   {float(sm[:, j].quantile(0.1)):7.2f} / {float(sm[:, j].quantile(0.9)):7.2f}",
              flush=True)
    gpi = sm[:, 5]
    low = torch.zeros(n_win, dtype=torch.bool)
    low[2: len(gpi) + 2] = gpi <= gpi.quantile(0.1)             # центр окна сглаживания
    before = torch.zeros(n_win, dtype=torch.bool)
    before[:-1] = low[1:]
    print(f"\n2. вход по путям (Σ PSP мВ × Гц; среднее / перед провалом GPi / в провале):", flush=True)
    j = 0
    for k in targets:
        for p in paths[k] + [None]:
            nm = f"{k} ← {p.name}" if p is not None else f"{k}: фон bias, мВ"
            print(f"   {nm:44s} {float(D[:, j].mean()):10.1f} {float(D[before, j].mean()):10.1f} {float(D[low, j].mean()):10.1f}",
                  flush=True)
            j += 1
    print(f"   частота VA: среднее {float(R[:, 6].mean()):.1f}, в провале GPi {float(R[low, 6].mean()):.1f} Гц; "
          f"нажатий в провале {float(press[low].float().mean()):.0%} окон против {float(press.float().mean()):.0%} в среднем", flush=True)
    # 4. структура активности GPi и связь с VA
    def corr(a: torch.Tensor, c: torch.Tensor) -> float:
        a, c = a - a.mean(), c - c.mean()
        return float((a * c).sum() / (a.norm() * c.norm()).clamp_min(1e-12))

    def binned(x: torch.Tensor, k: int) -> torch.Tensor:
        n = (len(x) // k) * k
        return x[:n].view(-1, k, *x.shape[1:]).sum(1)

    print("\n4. GPi: структура и связь с VA", flush=True)
    for k in (1, 10):
        gb = binned(G, k).float()
        fano = float((gb.var(0) / gb.mean(0).clamp_min(1e-9)).median())
        sub = gb[:, torch.randperm(gb.shape[1])[:200]]
        z = (sub - sub.mean(0)) / sub.std(0).clamp_min(1e-9)
        cc = (z.T @ z) / (len(z) - 1)
        pair = float(cc[~torch.eye(len(cc), dtype=torch.bool)].mean())
        print(f"   окно {k * WIN:.0f} мс: фактор Фано клетки (медиана) {fano:.2f} (пуассон 1); средняя корреляция пар клеток {pair:.3f}", flush=True)
    pauses = (binned(G, 5) == 0).float().mean()
    print(f"   доля окон 50 мс без спайков у клетки (пауза): {float(pauses):.1%}", flush=True)
    for j, nm in ((5, "GPi"), (4, "STN"), (3, "GPe"), (6, "VA")):
        x = R[:, j] - R[:, j].mean()
        spec = torch.fft.rfft(x).abs() ** 2
        f = torch.fft.rfftfreq(len(x), WIN / 1000)
        band = (f >= 2) & (f <= 45)
        fpk = float(f[band][spec[band].argmax()])
        share = float(spec[band][(f[band] - fpk).abs() <= 2].sum() / spec[band].sum())
        print(f"   {nm}: пик спектра частоты популяции {fpk:.1f} Гц, доля мощности ±2 Гц около пика (2–45 Гц) {share:.0%}", flush=True)
    for k in (1, 5, 10):
        print(f"   корреляция GPi–VA, окна {k * WIN:.0f} мс: {corr(binned(R[:, 5], k), binned(R[:, 6], k)):+.2f}", flush=True)
    xc = [corr(R[:-l or None, 5], R[l:, 6]) for l in range(0, 11)]
    print("   GPi(t) → VA(t + лаг), лаг 0…100 мс: " + " ".join(f"{c:+.2f}" for c in xc), flush=True)
    # вход за 100–300 мс до провала GPi
    ev = low.nonzero().flatten()
    ev = ev[ev >= 30]
    far = torch.zeros(n_win, dtype=torch.bool)
    for d in range(10, 31):
        far[ev - d] = True
    j = 0
    line = "   вход GPi за 100–300 мс до провала (среднее → там): "
    for p in paths["GPi"]:
        line += f"{p.name} {float(D[:, j].mean()):.0f} → {float(D[far, j].mean()):.0f}; "
        j += 1
    print(line, flush=True)
    print(f"\n3. условие нажатия: {int(press.sum())} окон, {int(press.sum()) / sec:.1f} в с", flush=True)
    print(f"всего {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
