"""
Возбуждающий и тормозный вход клеток колонки по источникам (CHAT_PLAN шаг 3,
цикл 28): откуда общий ток ПФК (цикл 27: корреляция тока между клетками
L2/3 E 0.85, PV 0.98, у смысловой 0.03–0.04).

Мозг — как в tools/column_compare.py (organs=words, разгон 2.5 с, адаптация
с гомеостазом, запись без гомеостаза с шагом 1 мс). Вклад проекции в клетку
за окно — Σ весов её входов × спайки их источников в окне (мВ движка; без
кинетики синапса и STP — для корреляций в окнах 5 мс хватает). Ещё один
источник — среднее поле недостающих входов (mf_exc / mf_inh × mf_rate, одно
на популяцию — общее по построению). Внешний фон (независимый у клетки) не
входит.

По популяции-получателю (200 клеток) и источнику, окна 5 мс:
  - средний вклад (мВ движка за окно);
  - σ во времени у клетки и корреляция колебаний между клетками;
  - доля в общей части суммы: колебания средней по клеткам суммы
    (общая часть) раскладываются по источникам — cov(источник, сумма) /
    var(сумма), в сумме 1 (возбуждение и торможение — отдельно).

    PYTHONPATH=. python tools/input_sources.py [areas=prefrontal,association] [pops=L23.E,L23.PV,L4.E,L4.PV]
        [seed=42] [adapt=30] [record=3] [extpop=prefrontal:1]
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
    develop.override_external(cfg, opts.get("extpop", ""))
    b = Mind(cfg).brain
    b.plasticity = False
    s = develop.Silence(b)
    names = opts.get("areas", "prefrontal,association").split(",")
    wanted = opts.get("pops", "L23.E,L23.PV,L4.E,L4.PV").split(",")

    gen = torch.Generator().manual_seed(1)
    targets = {}
    for a in names:
        area = b.areas[a]
        for key in wanted:
            layer, kind = key.split(".")
            pop = area.layers[layer][kind]
            rows = torch.randperm(pop.size, generator=gen)[:200]
            srcs = []
            for p in b.projections:
                if p.source != "net" or p.post.start != pop.start:
                    continue
                r = rows.to(p.pre.device)
                srcs.append((p.name, p.pre[r].long(), p.weight[r].float()))
            q = int(b.mf_post_id[pop.start]) if b.mf_on else None
            targets[(a, key)] = (pop, srcs, q)

    b.run(2500.0, s._blank(2500.0))
    for _ in range(int(opts.get("adapt", 30))):
        b.homeostasis_update(b.observe(1000.0, s._blank(1000.0)))
    b.cfg.homeostasis = False
    b.CHUNK = int(round(1.0 / b.cfg.dt_ms))
    b._graphs.clear()

    bins = int(float(opts.get("record", 3)) * 1000) // 5 * 5
    series = {k: {name: torch.zeros(bins, 200, device=b.device) for name, _, _ in srcs}
              for k, (_, srcs, _) in targets.items()}
    field_e = {k: torch.zeros(bins, device=b.device) for k in targets}
    field_i = {k: torch.zeros(bins, device=b.device) for k in targets}
    # Частоты источников среднего поля по окнам — разложить поле по источникам.
    mf_rates = torch.zeros(bins, b.mf_rate.numel(), device=b.device) if b.mf_on else None
    mf_names = {}
    if b.mf_on:
        for area in b.areas.values():
            for _, pop in area.populations:
                i = int(b.mf_pre_id[pop.start])
                if i < b.mf_rate.numel() - 1:
                    mf_names[i] = pop.name
    for t in range(bins):
        spikes = b.observe(1.0, s._blank(1.0)) * 0.001
        de = b.mf_exc @ b.mf_rate if b.mf_on else None
        di = b.mf_inh @ b.mf_rate if b.mf_on else None
        if mf_rates is not None:
            mf_rates[t] = b.mf_rate
        for k, (_, srcs, q) in targets.items():
            for name, pre, w in srcs:
                series[k][name][t] = (w * spikes[pre]).sum(1)
            if q is not None:
                field_e[k][t] = de[q]
                field_i[k][t] = di[q]

    w5 = bins // 5
    print(f"зерно {cfg.seed}, адаптация {opts.get('adapt', 30)} с, запись {bins / 1000:.0f} с"
          f"{', внешний фон ' + opts['extpop'] if opts.get('extpop') else ''}; окна 5 мс, мВ движка за окно")
    for k, (pop, srcs, _) in targets.items():
        print(f"== {k[0]} {k[1]}")
        for sign, label in ((1, "возбуждение"), (-1, "торможение")):
            parts = {}
            for name, _, w in srcs:
                if (float(w.mean()) > 0) == (sign > 0):
                    parts[name] = series[k][name][: w5 * 5].view(w5, 5, 200).sum(1)
            fld = (field_e if sign > 0 else field_i)[k][: w5 * 5].view(w5, 5).mean(1) * 5 * sign
            if float(fld.abs().mean()) > 0:
                parts["среднее поле (общее)"] = fld[:, None].expand(w5, 200)
            total = sum(parts.values())
            m = total.mean(1)
            var = float(m.var())
            print(f"   {label}: сумма {float(total.mean()):.2f}, корр. между клетками {corr(total):.2f}")
            for name, x in sorted(parts.items(), key=lambda kv: -float(kv[1].mean().abs())):
                share = float(((x.mean(1) - x.mean()) * (m - m.mean())).mean()) / var if var > 0 else float("nan")
                print(f"      {name:34s} {float(x.mean()):8.3f}  σ {float(x.std(0).mean()):7.3f}  "
                      f"корр. {corr(x):5.2f}  доля общей части {share:6.1%}", flush=True)
            if sign > 0 and mf_rates is not None and targets[k][2] is not None:
                q = targets[k][2]
                terms = mf_rates[: w5 * 5].view(w5, 5, -1).mean(1) * b.mf_exc[q][None, :] * 5
                fm = terms.sum(1)
                fv = float(fm.var())
                order = torch.argsort(-terms.mean(0))[:8].tolist()
                print("      поле по источникам (среднее; σ во времени; доля колебаний поля):")
                for i in order:
                    if float(terms[:, i].mean()) <= 0:
                        continue
                    sh = float(((terms[:, i] - terms[:, i].mean()) * (fm - fm.mean())).mean()) / fv if fv > 0 else float("nan")
                    print(f"         {mf_names.get(i, '?'):28s} {float(terms[:, i].mean()):8.2f}  σ {float(terms[:, i].std()):7.2f}  {sh:6.1%}")


def corr(x: torch.Tensor) -> float:
    x = x - x.mean(0)
    sd = x.std(0)
    live = sd > 0
    if int(live.sum()) < 3:
        return float("nan")
    z = x[:, live] / sd[live]
    cc = (z.T @ z) / (z.shape[0] - 1)
    return float(cc[~torch.eye(cc.shape[0], dtype=torch.bool, device=cc.device)].mean())


if __name__ == "__main__":
    main(dict(a.split("=", 1) for a in sys.argv[1:]))
