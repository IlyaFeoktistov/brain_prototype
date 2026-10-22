"""
Фаза «развития» с гомеостазом возбудимости (cfg.homeostasis, engine.
homeostasis_update): мозг живёт с пустым экраном окнами win мс, после каждого
окна регулируемые клетки сдвигают фон к частоте покоя своего типа. В конце —
сдвиги по популяциям (среднее, доля упёршихся в предел) и файл сдвигов
~/.brain/fits/<out>-homeo.pt (стенды берут его по homeo=<out>; гомеостаз
на замерах выключен — сдвиг заморожен).

    PYTHONPATH=. python tools/develop.py [seconds=60] [win=1000] [seed=42] [fit=lgn-a]
        [relay_v_t_mv=-59] [relay_rest_mv=-48] [trn_scale=0.5] [out=lgn-a] [every=10]
"""

from __future__ import annotations

import sys

import torch

from brain import anatomy
from brain.config import BRAIN_HOME, BrainConfig
from brain.mind import Mind


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


def main(opts: dict) -> None:
    cfg = BrainConfig()
    cfg.size = "small"
    cfg.seed = int(opts.get("seed", 42))
    cfg.autosave = False
    cfg.homeostasis = True
    for k in ("relay_v_t_mv", "relay_rest_mv"):
        if k in opts:
            setattr(cfg, k, float(opts[k]))
    if opts.get("fit"):
        anatomy.use_fit(cfg, opts["fit"])
    b = Mind(cfg).brain
    b.plasticity = False
    for p in b.projections:
        if p.name == "trn -> lgn":
            p.weight.mul_(float(opts.get("trn_scale", 1.0)))
    b.run(2500.0)
    win, every = float(opts.get("win", 1000)), int(opts.get("every", 10))
    windows = int(float(opts.get("seconds", 60)) * 1000 / win)

    for w in range(windows):
        rates = b.observe(win)
        b.homeostasis_update(rates)
        if w == 0 or (w + 1) % every == 0:
            report(b, rates, f"окно {w + 1} из {windows}")

    out = opts.get("out", "lgn-a")
    path = BRAIN_HOME / "fits" / f"{out}-homeo.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(b.homeo_shift.cpu(), path)
    print(f"\nсдвиги записаны в {path}")


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
