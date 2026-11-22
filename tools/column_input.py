"""
Вход клеток колонки по популяциям (CHAT_PLAN шаг 3, цикл 12): что именно
держит колонку в ровном режиме или во вспышках. Мозг как в опыте B
(organs=words), адаптация с гомеостазом, затем запись без гомеостаза:
каждые 5 мс — состояние токов клеток (мВ движка):

  - возбуждение построенных синапсов и внешнего фона (engine.i_syn);
  - торможение построенных синапсов (engine.i_inh);
  - среднее поле недостающих соседей (mf_exc / mf_inh × сглаженная частота
    источников, mf_rate) — одно на популяцию.

По популяции: среднее; разброс во времени у клетки (σ по окнам, среднее по
клеткам); корреляция колебаний возбуждения между клетками (выборка 200
клеток) — общий ли вход.

    PYTHONPATH=. python tools/column_input.py [area=prefrontal] [seed=42] [adapt=3] [record=3] [extcopy=…] [extpop=…] [smooth=src->dst] [mf_frozen=1] [indep=g]
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
    develop.copy_external(cfg, opts.get("extcopy", ""))
    b = Mind(cfg).brain
    b.plasticity = False
    s = develop.Silence(b)
    if opts.get("indep"):
        # Независимый внешний вход колонки при том же среднем: событий λ/g,
        # вес × g — среднее λw то же, разброс λw² × g, у каждой клетки свой.
        g = float(opts["indep"])
        target = b.areas[opts.get("area", "prefrontal")]
        cells = target.cells
        b.noise_mean[cells] /= g
        b.noise_gain[cells] *= g
        b.noise_std[cells] = b.noise_mean[cells].sqrt()
        print(f"внешний вход {target.name}: разброс × {g} при том же среднем", flush=True)
    if opts.get("mf_frozen") == "1":
        # Среднее поле на частотах покоя по данным (старт mf_rate), без
        # колебаний частоты колонки: до первого шага (шаг пишется в CUDA-граф).
        b.mf_decay = 1.0
        print("среднее поле заморожено на частотах покоя", flush=True)
    area = b.areas[opts.get("area", "prefrontal")]
    pops = {f"{layer}.{kind}": p for layer, group in area.layers.items() for kind, p in group.items()}

    b.run(2500.0, s._blank(2500.0))
    for _ in range(int(opts.get("adapt", 3))):
        b.homeostasis_update(b.observe(1000.0, s._blank(1000.0)))
    b.cfg.homeostasis = False
    # После адаптации: замена путей средним по фактической частоте (develop.smooth_paths).
    develop.smooth_paths(b, opts.get("smooth", ""))

    bins = int(float(opts.get("record", 3)) * 1000 / 5)
    gen = torch.Generator().manual_seed(1)
    picks = {k: torch.randperm(p.size, generator=gen)[:200].to(b.device) + p.start for k, p in pops.items()}
    exc = {k: torch.zeros(bins, 200, device=b.device) for k in pops}
    inh = {k: torch.zeros(bins, 200, device=b.device) for k in pops}
    mf_e = {k: torch.zeros(bins, device=b.device) for k in pops}
    mf_i = {k: torch.zeros(bins, device=b.device) for k in pops}
    rate = {k: 0.0 for k in pops}
    spikes = {k: torch.zeros(bins, 200, device=b.device) for k in pops}
    popn = {k: torch.zeros(bins, device=b.device) for k in pops}
    for t in range(bins):
        r = b.observe(5.0, s._blank(5.0))
        dc_e = b.mf_exc @ b.mf_rate if b.mf_on else None
        dc_i = b.mf_inh @ b.mf_rate if b.mf_on else None
        for k, p in pops.items():
            exc[k][t] = b.i_syn[picks[k]].float()
            inh[k][t] = b.i_inh[picks[k]].float()
            if dc_e is not None:
                q = int(b.mf_post_id[p.start])
                mf_e[k][t] = dc_e[q]
                mf_i[k][t] = dc_i[q]
            rate[k] += float(r[p.slice].mean()) / bins
            spikes[k][t] = r[picks[k]] * 0.005
            popn[k][t] = r[p.slice].sum() * 0.005

    print(f"зерно {cfg.seed}, {area.name}: вход клеток, мВ (среднее; σ во времени у клетки; корреляция колебаний между клетками)")
    print("популяция   частота    возб. синапсы+фон            торм. синапсы        поле возб. / торм.   | спайки: CV популяции, корр. пар")
    for k in pops:
        e, i = exc[k], inh[k]
        x = e - e.mean(0)
        sd = x.std(0)
        live = sd > 0
        corr = float("nan")
        if int(live.sum()) > 2:
            z = x[:, live] / sd[live]
            cc = (z.T @ z) / (z.shape[0] - 1)
            corr = float(cc[~torch.eye(cc.shape[0], dtype=torch.bool, device=cc.device)].mean())
        pn = popn[k]
        cv_pop = float(pn.std() / pn.mean()) if float(pn.mean()) > 0 else float("nan")
        y = spikes[k] - spikes[k].mean(0)
        sy = y.std(0)
        ly = sy > 0
        sp_corr = float("nan")
        if int(ly.sum()) > 2:
            zy = y[:, ly] / sy[ly]
            cy = (zy.T @ zy) / (zy.shape[0] - 1)
            sp_corr = float(cy[~torch.eye(cy.shape[0], dtype=torch.bool, device=cy.device)].mean())
        print(f"   {k:8s} {rate[k]:6.2f} Гц   {float(e.mean()):6.2f}; σ {float(e.std(0).mean()):5.2f}; r {corr:5.2f}"
              f"     {float(i.mean()):6.2f}; σ {float(i.std(0).mean()):5.2f}"
              f"     {float(mf_e[k].mean()):6.2f} / {float(mf_i[k].mean()):6.2f}"
              f"   | CV {cv_pop:5.2f}, r {sp_corr:5.3f}", flush=True)


if __name__ == "__main__":
    main(dict(a.split("=", 1) for a in sys.argv[1:]))
