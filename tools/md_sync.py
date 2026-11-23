"""
Синхронность клеток MD в сопоставимых с данными мерах (CHAT_PLAN шаг 3,
цикл 22): спайки 200 клеток MD с шагом 1 мс в опыте B (словесный мозг,
адаптация, запись), по парам:
- корреляция Пирсона счёта в окнах 5 мс (как tools/md_overlap.py);
- кросс-коррелограмма ±100 мс с шагом 1 мс; синхронность — доля её площади
  в ±7.5 мс (Borden et al. 2022, https://doi.org/10.1016/j.neuron.2022.06.008;
  Wang et al. 2010) — у независимых 15/201 ≈ 0.075; пары с ≥ 50 совпадениями;
- то же после джиттера: каждый спайк сдвигается случайно внутри своего окна
  25 мс (медленные общие колебания частоты остаются, точное совпадение
  уходит); разность сырое − джиттер — избыток точной синхронности.

    PYTHONPATH=. python tools/md_sync.py [seed=42] [adapt=30] [record=3] [area=md]
"""

from __future__ import annotations

import sys

import torch

from brain.config import BrainConfig
from brain.mind import Mind
from tools import develop


def pearson(x: torch.Tensor) -> float:
    """Средняя корреляция пар по окнам [окна, клетки] (молчащие — мимо)."""

    x = x - x.mean(0)
    sd = x.std(0)
    live = sd > 0
    z = x[:, live] / sd[live]
    cc = (z.T @ z) / (z.shape[0] - 1)
    return float(cc[~torch.eye(cc.shape[0], dtype=torch.bool, device=cc.device)].mean())


def ccg(x: torch.Tensor, lag: int = 100) -> torch.Tensor:
    """Кросс-коррелограммы всех пар [лаги, клетки, клетки] по растру [мс, клетки]."""

    t = x.shape[0]
    out = []
    for k in range(-lag, lag + 1):
        a, b = (x[: t - k], x[k:]) if k >= 0 else (x[-k:], x[: t + k])
        out.append(a.T @ b)
    return torch.stack(out)


def synchrony(x: torch.Tensor, half: float = 7.5, lag: int = 100, min_events: int = 50) -> tuple[float, int, torch.Tensor]:
    """Мера Borden: доля площади ККГ в ±half мс; средняя по парам с ≥ min_events совпадений."""

    c = ccg(x, lag)
    centre = int(half)
    mid = c[lag - centre: lag + centre + 1].sum(0)
    total = c.sum(0)
    n = x.shape[1]
    pair = torch.triu(torch.ones(n, n, dtype=torch.bool, device=x.device), 1) & (mid >= min_events)
    s = (mid[pair] / total[pair]) if pair.any() else torch.tensor([float("nan")])
    grand = c[:, torch.triu(torch.ones(n, n, dtype=torch.bool, device=x.device), 1)].sum(1)
    return float(s.mean()), int(pair.sum()), grand / grand.sum()


def jitter(x: torch.Tensor, window: int = 25, gen=None) -> torch.Tensor:
    """Каждый спайк — в случайное место своего окна window мс (счёт окна сохраняется)."""

    t, n = x.shape
    t = t // window * window
    w = x[:t].view(t // window, window, n)
    counts = w.sum(1).long()
    out = torch.zeros_like(w)
    total = int(counts.max()) if counts.numel() else 0
    for k in range(total):
        has = counts > k
        pos = torch.randint(0, window, counts.shape, device=x.device, generator=gen)
        out.scatter_add_(1, pos.unsqueeze(1), has.unsqueeze(1).float())
    return out.view(t, n)


def report(x: torch.Tensor, label: str, gen) -> None:
    j = jitter(x, gen=gen)
    x = x[: j.shape[0]]
    p, pj = pearson(x.view(-1, 5, x.shape[1]).sum(1)), pearson(j.view(-1, 5, j.shape[1]).sum(1))
    s, ns, grand = synchrony(x)
    sj, _, gj = synchrony(j)
    print(f"{label}: {float(x.sum(0).mean()) / x.shape[0] * 1000:.2f} Гц; корр. 5 мс {p:.3f} (джиттер {pj:.3f}, избыток {p - pj:+.3f}); "
          f"синхронность ±7.5 мс {s:.3f} по {ns} парам (джиттер {sj:.3f}, независимые 0.075)", flush=True)
    lags = [0, 2, 5, 10, 20, 50, 100]
    print("   общая ККГ / независимые, лаг мс: " + ", ".join(f"{k}: {float(grand[100 + k]) * 201:.2f}" for k in lags)
          + f"; после джиттера 0: {float(gj[100]) * 201:.2f}", flush=True)


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
    b.run(2500.0, s._blank(2500.0))
    for _ in range(int(opts.get("adapt", 30))):
        b.homeostasis_update(b.observe(1000.0, s._blank(1000.0)))
    b.cfg.homeostasis = False
    # Запись с шагом 1 мс: кусок шага движка (CHUNK, 10 шагов = 5 мс) → 2
    # шага; граф CUDA пересобирается. Динамика та же, только счёт мельче.
    b.CHUNK = int(round(1.0 / b.cfg.dt_ms))
    b._graphs.clear()
    gen = torch.Generator(device=b.device).manual_seed(1)
    areas = opts.get("area", "md,trn_md").split(",")
    picks = {}
    for name in areas:
        pop = b.areas[name].E
        picks[name] = torch.randperm(pop.size, generator=gen, device=b.device)[:200] + pop.start
    ms = int(float(opts.get("record", 3)) * 1000)
    rasters = {name: torch.zeros(ms, len(p), device=b.device) for name, p in picks.items()}
    blank = s._blank(1.0)
    for t in range(ms):
        r = b.observe(1.0, blank)
        for name, p in picks.items():
            rasters[name][t] = (r[p] * 0.001).round()
    for name, x in rasters.items():
        report(x, name, gen)


if __name__ == "__main__":
    main(dict(a.split("=", 1) for a in sys.argv[1:]))
