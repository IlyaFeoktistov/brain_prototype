"""
Где теряется различие A/B у клеток букв L4 E — по источникам входа, без
вмешательства. На каждом показе фразы A или B (как letters_select) по
каждой клетке копятся:

  - вход каждого построенного пути в клетку (Σ вес × спайки пресинапса за
    окно; возбуждающие и тормозные пути отдельно — V1, местные слои, PV…);
  - ток возбуждения и торможения сомы (с движущей силой), их сумма;
  - мембрана, расстояние до порога (v_t − v) и доля окон в 2 мВ от порога;
  - спайки.

Для каждой величины:
  d′ клетки = (среднее A − среднее B) / разброс внутри класса; печатается
  медиана |d′| по клеткам и та же медиана при перемешанных метках (уровень
  шума) — эффект есть, только если первая заметно выше второй;
  надёжность половин A−B (как letters_select);
  корр. A−B с входом V1 по клеткам: переносится ли рисунок различия V1 в эту
  величину (у тормозных путей «+» — вычитают именно различие).

Вопрос — на каком шаге (вход → сумма токов → мембрана → спайки) различие
падает до уровня шума; механизм выбирать после.

    PYTHONPATH=. python tools/letters_currents.py seed=1 ee_skip=visual homeo=lgn-a-ee-s1 [n=30] [rep=0]
"""

from __future__ import annotations

import sys

import torch

from brain.session import Session
from tools.reading_bench import build

WIN = 10.0


def main(opts: dict) -> None:
    opts = {"fit": "lgn-a", "homeo": f"lgn-a-s{opts.get('seed', 1)}", "relay_v_t_mv": "-59",
            "relay_rest_mv": "-48", "trn_scale": "0.5", **opts}
    mind, b = build(opts)
    cfg = b.cfg
    s = Session(mind, window_ms=WIN)
    n_rep, fix, step = int(opts.get("n", 30)), float(opts.get("fix", 225)), float(opts.get("step", 5))
    texts = (opts.get("a", "мама мыла раму"), opts.get("b", "папа пила чай"))
    gen = torch.Generator().manual_seed(int(opts.get("seed", 1)) + 1000 * int(opts.get("rep", 0)))
    if opts.get("rep"):
        # Повтор на том же мозге: другой шум и порядок показов.
        torch.manual_seed(1000 + int(opts["rep"]))
    L4 = b.areas["letters"].layers["L4"]["E"]
    paths = [p for p in b.projections if p.post.name == L4.name and p.source == "net"]
    names = [p.name for p in paths]
    v1 = next(n for n in names if n.split(" -> ")[0] == "visual")
    v_t = b.v_t[L4.slice]
    print(f"пути в {L4.name}: " + ", ".join(f"{p.name} {p.pre.shape[1]}" for p in paths), flush=True)

    s.line, s.gaze = "", 0.0
    b.run(2500.0)

    def gazes(line: str) -> list[float]:
        width = s.screen.width(line) / max(1, len(line))
        n = max(1, int((len(line) + step - 1) // step))
        return [i * step * width for i in range(n)][:3]

    keys = [*names, "возбуждение", "торможение", "сумма токов", "мембрана", "до порога", "у порога", "спайки"]

    def show(lab: int) -> dict[str, torch.Tensor]:
        s.line = ""
        b.run(500.0, s._blank(500.0))
        line = texts[lab] + " "
        acc = {k: torch.zeros(L4.size) for k in keys}
        n_win = 0
        for i, g in enumerate(gazes(line)):
            if i:
                s.line, s.gaze = line, g
                b.run(cfg.saccade_ms, s._blank(cfg.saccade_ms))
            s.line, s.gaze = line, g
            for _ in range(int(fix // WIN)):
                r = b.observe(WIN, s._view(WIN)) * (WIN / 1000.0)
                for p in paths:
                    acc[p.name] += (p.weight.float() * r[p.pre.long()]).sum(1).cpu()
                v = b.v[L4.slice]
                excite = ((cfg.ampa_reversal_mv - v) / (cfg.ampa_reversal_mv - cfg.epsp_holding_mv)).clamp(min=0.0)
                force = ((v - cfg.gaba_reversal_mv) / (cfg.ipsp_holding_mv - cfg.gaba_reversal_mv)).clamp(min=0.0)
                e = (b.i_syn[L4.slice] * excite).float()
                i_ = (b.i_inh[L4.slice] * force).float()
                acc["возбуждение"] += e.cpu()
                acc["торможение"] += i_.cpu()
                acc["сумма токов"] += (e + i_).cpu()
                acc["мембрана"] += v.float().cpu()
                acc["до порога"] += (v_t - v).float().cpu()
                acc["у порога"] += ((v_t - v) < 2.0).float().cpu()
                acc["спайки"] += r[L4.slice].float().cpu()
                n_win += 1
        for k in keys:
            if k != "спайки" and k not in names:
                acc[k] /= n_win
        return acc

    order = [0, 1] * n_rep
    order = [order[i] for i in torch.randperm(len(order), generator=gen)]
    feats = []
    for j, lab in enumerate(order):
        feats.append(show(lab))
        if (j + 1) % 20 == 0:
            print(f"  показов {j + 1}/{len(order)}", flush=True)
    y = torch.tensor(order)
    half = len(order) // 2

    def dprime(x: torch.Tensor, lab: torch.Tensor) -> torch.Tensor:
        a, c = x[lab == 0], x[lab == 1]
        sd = ((a.var(0) + c.var(0)) / 2).sqrt()
        return torch.where(sd > 0, (a.mean(0) - c.mean(0)) / sd.clamp_min(1e-9), torch.zeros_like(sd))

    def halves(x: torch.Tensor) -> float:
        d = [x[part][y[part] == 0].mean(0) - x[part][y[part] == 1].mean(0)
             for part in (slice(0, half), slice(half, None))]
        d = [u - u.mean() for u in d]
        return float((d[0] * d[1]).sum() / ((d[0].norm() * d[1].norm()).clamp_min(1e-12)))

    def corr(u: torch.Tensor, w: torch.Tensor) -> float:
        u, w = u - u.mean(), w - w.mean()
        return float((u * w).sum() / (u.norm() * w.norm()).clamp_min(1e-12))

    data = {k: torch.stack([f[k] for f in feats]) for k in keys}
    diff_v1 = data[v1][y == 0].mean(0) - data[v1][y == 1].mean(0)
    shuf = y[torch.randperm(len(y), generator=gen)]
    print(f"\n=== {L4.name}: «{texts[0]}» против «{texts[1]}», по {n_rep} показов")
    print(f"  {'величина':34s} {'среднее':>9s} {'|d′| мед.':>9s} {'шум':>6s} {'d′>0.5':>7s} {'шум':>6s} "
          f"{'надёжн.':>8s} {'корр. с V1':>10s}")
    for k in keys:
        x = data[k]
        d, d0 = dprime(x, y), dprime(x, shuf)
        diff = x[y == 0].mean(0) - x[y == 1].mean(0)
        print(f"  {k:34s} {float(x.mean()):9.3f} {float(d.abs().median()):9.3f} {float(d0.abs().median()):6.3f} "
              f"{float((d.abs() > 0.5).float().mean()):7.1%} {float((d0.abs() > 0.5).float().mean()):6.1%} "
              f"{halves(x):+8.3f} {corr(diff, diff_v1):+10.3f}", flush=True)


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
