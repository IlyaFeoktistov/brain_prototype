"""
Баланс возбуждения и торможения в участке L2/3 руки перед ложным залпом
(DESIGN 2026-11-16 «Что ведёт залп руки»: участок у клавиши разгоняется сам
за ~150 мс, VLp не ведёт). Вопрос: растёт ли торможение вместе с
возбуждением или запаздывает; это токи, а не частоты. Без правки сети,
кроме флага va_bias (−76 — как в hand_onset, VA молчит).

Пустой экран, окна 10 мс; в конце каждого окна — снимок токов клеток L2/3 E
и PV руки (мВ, как в движке): возбуждающий спайковый i_syn (+ NMDA, если
включён) × движущая сила (E_AMPA − v), тормозной i_inh × (v − E_ГАМК)
(нормировки движка); среднее поле недостающих соседей — одно на
популяцию, по частоте всей популяции (не местное), печатается отдельно.
Поставка местных путей (Σ PSP × частота источника): E → E, PV → E (и прочие
тормозные → E), E → PV. Событие и участок — как hand_onset; средний ход за
lag окон до начала нажатия: местные E и PV — частота, токи, E/I; доля
клеток участка, выстреливших в окне нажатия. Гомеостаз: сдвиги клеток L2/3
E и PV руки из homeo-файла и доля у предела ±cfg.homeostasis_max_mv.

    PYTHONPATH=. .venv/bin/python tools/hand_ei.py [sec=20] [seed=1] [va_bias=-76] [near=0.005] [lag=20]
"""

from __future__ import annotations

import sys
import time

import torch

from brain.session import Session
from tools.reading_bench import BRAIN_HOME, build

WIN = 10.0


def main(opts: dict) -> None:
    seed = int(opts.get("seed", 1))
    opts = {"fit": "lgn-a", "homeo": f"lgn-d-pop-tcind-s{seed}", "relay_v_t_mv": "-59", "trn_scale": "0.1", **opts}
    mind, b = build(opts)
    cfg = b.cfg
    s = Session(mind, window_ms=WIN)
    sec, va_bias, near = float(opts.get("sec", 20)), float(opts.get("va_bias", -76)), float(opts.get("near", 0.005))
    n_lag = int(opts.get("lag", 20))
    torch.manual_seed(seed)
    A = b.areas
    if va_bias:
        b.bias[A["thalamus"].E.slice] += va_bias
    E, PV, L5 = A["hand"].layers["L23"]["E"], A["hand"].layers["L23"]["PV"], s.H
    cE, cP, c5 = (b.field_cols[P.name].float().cpu() for P in (E, PV, L5))
    keys, cells = len(s.keyboard), s.hand.cells

    shift = torch.load(BRAIN_HOME / "fits" / f"{opts['homeo']}-homeo.pt", map_location="cpu")
    lim = cfg.homeostasis_max_mv
    for nm, P in (("L2/3 E", E), ("L2/3 PV", PV), ("L5 E", L5), ("L5 PV", A["hand"].layers["L5"]["PV"])):
        x = shift[P.slice].float()
        print(f"гомеостаз руки {nm}: сдвиг {float(x.mean()):+.2f} мВ, у предела ±{lim:.0f}: {float((x.abs() >= lim - 0.01).float().mean()):.0%}",
              flush=True)

    into_E = [p for p in b.projections if p.post is E and p.source == "net" and p.name.startswith("hand:")]
    into_P = [p for p in b.projections if p.post is PV and p.source == "net" and p.name.startswith("hand:")]
    ee = [p for p in into_E if float(p.weight.mean()) > 0]
    ie = [p for p in into_E if float(p.weight.mean()) < 0]
    ep = [p for p in into_P if float(p.weight.mean()) > 0]
    print("местные пути: E ← " + ", ".join(p.name for p in ee) + "; E ⊣ " + ", ".join(p.name for p in ie)
          + "; PV ← " + ", ".join(p.name for p in ep), flush=True)

    # размах местных путей по карте: |место входа − место клетки|
    cols_of = {P.name: b.field_cols[P.name].float().cpu() for P in (E, PV)}
    for p in ee[:1] + ie[:1] + ep[:1]:
        pre_pop = next(P for P in (E, PV) if P.start <= int(p.pre.min()) < P.start + P.size)
        pre = p.pre.long().cpu() - pre_pop.start
        post_c = cols_of[p.post.name][:, None]
        d = (cols_of[pre_pop.name][pre.clamp(0, pre_pop.size - 1)] - post_c).abs()
        d = torch.minimum(d, 1 - d)
        print(f"  размах {p.name}: входов на клетку {p.pre.shape[1]}, |Δместо| медиана {float(d.median()):.4f}, 90% {float(d.quantile(0.9)):.4f} "
              f"(участок клавиши ±{near}); вес (ед. движка) {float(p.weight.float().mean()):.3f}", flush=True)

    def mf(P) -> tuple[float, float]:
        if not b.mf_on:
            return 0.0, 0.0
        q = int(b.mf_post_id[P.start])
        return float((b.mf_exc[q] * b.mf_rate).sum()), float((b.mf_inh[q] * b.mf_rate).sum())

    excite = lambda v: ((cfg.ampa_reversal_mv - v) / (cfg.ampa_reversal_mv - cfg.epsp_holding_mv)).clamp(min=0.0)  # noqa: E731
    force = lambda v: ((v - cfg.gaba_reversal_mv) / (cfg.ipsp_holding_mv - cfg.gaba_reversal_mv)).clamp(min=0.0)  # noqa: E731

    s.line, s.gaze = "", 0.0
    b.run(2500.0, s._blank(2500.0))
    n_win = int(sec * 1000 / WIN)
    blank = s._blank(WIN)
    keep = {k: [] for k in ("rE", "rP", "r5", "eE", "iE", "eP", "iP", "sEE", "sIE", "sEP")}
    mfs = []
    t0 = time.time()
    for t in range(n_win):
        r = b.observe(WIN, blank)
        exc = b.i_syn + (b.i_nmda if b.nmda_on else 0.0)
        for key, P in (("E", E), ("P", PV)):
            v = b.v[P.slice]
            keep["e" + key].append((exc[P.slice] * excite(v)).half().cpu())
            keep["i" + key].append((b.i_inh[P.slice] * force(v)).half().cpu())
        keep["rE"].append(r[E.slice].half().cpu())
        keep["rP"].append(r[PV.slice].half().cpu())
        keep["r5"].append(r[L5.slice].half().cpu())
        for key, ps in (("sEE", ee), ("sIE", ie), ("sEP", ep)):
            keep[key].append(sum((p.weight.float() * r[p.pre.long()]).sum(1) for p in ps).cpu())
        mfs.append(mf(E) + mf(PV))
        if t == 49:
            print(f"  первые 0.5 с: {time.time() - t0:.0f} с", flush=True)
    K = {k: torch.stack(v).float() for k, v in keep.items()}
    mfs = torch.tensor(mfs)
    print(f"окон {n_win}, {time.time() - t0:.0f} с; среднее поле недостающих соседей (на популяцию): "
          f"E возб {float(mfs[:, 0].mean()):.2f} / торм {float(mfs[:, 1].mean()):.2f}; PV возб {float(mfs[:, 2].mean()):.2f} / "
          f"торм {float(mfs[:, 3].mean()):.2f} (единицы движка до движущей силы)", flush=True)
    print(f"в среднем по L2/3 E: ток возб {float(K['eE'].mean()):.3f}, торм {float(K['iE'].mean()):.3f}; PV: возб "
          f"{float(K['eP'].mean()):.3f}, торм {float(K['iP'].mean()):.3f}; частота E {float(K['rE'].mean()):.2f}, PV {float(K['rP'].mean()):.2f} Гц",
          flush=True)

    blk = (K["r5"][:, : keys * cells] > 0).float().view(n_win, keys, cells).mean(2)
    best = blk.argmax(1)
    cond = blk.gather(1, best[:, None]).squeeze(1) >= 0.5
    onset = [(t, int(best[t])) for t in range(n_lag + 1, n_win) if cond[t] and not cond[t - 1]]
    print(f"начал нажатия {len(onset)} ({len(onset) / sec:.1f} в с)", flush=True)
    lags = list(range(n_lag, -1, -1))
    rows = {k: [] for k in ("E Гц", "PV Гц", "E: ток возб", "E: ток торм", "E: возб/торм", "PV: ток возб", "PV: ток торм",
                            "E ← E (поставка)", "E ⊣ торм. (поставка)", "PV ← E (поставка)")}
    part = []
    for t, k in onset:
        pos = float(c5[k * cells: (k + 1) * cells].mean())
        mE, mP = (cE - pos).abs() <= near, (cP - pos).abs() <= near
        ts = torch.tensor([t - d for d in lags])
        rows["E Гц"].append(K["rE"][ts][:, mE].mean(1))
        rows["PV Гц"].append(K["rP"][ts][:, mP].mean(1))
        e, i = K["eE"][ts][:, mE].mean(1), K["iE"][ts][:, mE].mean(1)
        rows["E: ток возб"].append(e)
        rows["E: ток торм"].append(i)
        rows["E: возб/торм"].append(e / (-i).clamp_min(1e-9))
        rows["PV: ток возб"].append(K["eP"][ts][:, mP].mean(1))
        rows["PV: ток торм"].append(K["iP"][ts][:, mP].mean(1))
        rows["E ← E (поставка)"].append(K["sEE"][ts][:, mE].mean(1))
        rows["E ⊣ торм. (поставка)"].append(K["sIE"][ts][:, mE].mean(1))
        rows["PV ← E (поставка)"].append(K["sEP"][ts][:, mP].mean(1))
        part.append(float((K["rE"][t][mE] > 0).float().mean()))
    print(f"\nсредний ход у клавиши (абсолютные значения), лаг, мс: " + " ".join(f"{-d * WIN:>7.0f}" for d in lags))
    for k, v in rows.items():
        m = torch.stack(v).mean(0)
        print(f"  {k:22s} " + " ".join(f"{float(x):7.2f}" for x in m), flush=True)
    part = torch.tensor(part)
    print(f"\nклеток L2/3 E участка, выстреливших в окне нажатия: медиана {float(part.median()):.0%} "
          f"[10% {float(part.quantile(0.1)):.0%}, 90% {float(part.quantile(0.9)):.0%}]; обычно в окне {float((K['rE'] > 0).float().mean()):.1%}")
    print(f"всего {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
