"""
Цепочка вспышек ПФК → волны EC → залпы CA3 одним замером (CHAT_PLAN шаг 3,
цикл 25): сравнение сборок на одном зерне и с одними начальными условиями.
Мозг — как в опыте B (organs=words): разгон 2.5 с, адаптация с гомеостазом
окнами по 1 с, запись без гомеостаза окнами по 5 мс.

Событие уровня — окно 5 мс, где спайков популяции больше thr× средней;
подряд идущие окна — одно событие. По событиям: число в секунду,
длительность, межсобытийные интервалы (среднее, CV), амплитуда (пик ×
средней), доля клеток, выстреливших в окне события ±10 мс (против случайных
окон). Пороги — 5× (ПФК L2/3 E, CA3 PC, как в циклах 7–11) и 2× (EC II/III:
волны EC мельче, CV ~0.5) — одинаковы для обеих сборок, не цель из данных.
Ещё: CV популяции по окнам, корреляция пар (500 клеток, 5 мс), MD — для
проверки, что сборки различаются; потерянные события (переполнение таблицы —
нагрузка счёта, не физиология).

old=1 — прежняя петля реле ↔ TRN (случайная проводка, topographic 0, до
цикла 24) через tools/develop.override_paths; path= — любые другие пути.

    PYTHONPATH=. python tools/chain_sync.py [seed=42] [adapt=10] [record=6] [old=1] [path=…]
"""

from __future__ import annotations

import sys

import torch

from brain.config import BrainConfig
from brain.mind import Mind
from tools import develop

# Петля реле ↔ TRN до цикла 24 (anatomy.pathways: 20 входов, ±1.5 мВ, случайно).
OLD_LOOP = ",".join(
    f"{a}->{b}:20:{psp}:0"
    for relay, trn in (("mgn", "trn_mgn"), ("md", "trn_md"), ("thalamus", "trn_motor"), ("vl_cb", "trn_motor"))
    for a, b, psp in ((relay, trn, 1.5), (trn, relay, -1.5))
)


def events(y: torch.Tensor, thr: float, bins: int) -> list[tuple[int, int]]:
    """(начало, длина в окнах) серий окон выше thr× средней, вдали от краёв записи."""

    mean = float(y.mean())
    if mean <= 0:
        return []
    above = (y > thr * mean).tolist()
    out, t = [], 0
    while t < bins:
        if above[t]:
            u = t
            while u < bins and above[u]:
                u += 1
            if 2 <= t and u <= bins - 2:
                out.append((t, u - t))
            t = u
        else:
            t += 1
    return out


def pair_corr(x: torch.Tensor) -> float:
    x = x - x.mean(0)
    sd = x.std(0)
    live = sd > 0
    if int(live.sum()) < 2:
        return float("nan")
    x = x[:, live] / sd[live]
    cc = (x.T @ x) / (x.shape[0] - 1)
    return float(cc[~torch.eye(cc.shape[0], dtype=torch.bool, device=cc.device)].mean())


def report(name: str, y: torch.Tensor, fired: torch.Tensor, sample: torch.Tensor, thr: float, size: int, bins: int) -> None:
    sec = bins * 0.005
    mean = float(y.mean())
    hz = mean / size / 0.005
    cv = float(y.std() / y.mean()) if mean > 0 else float("nan")
    ev = events(y, thr, bins)
    head = f"{name}: {hz:.2f} Гц, CV по окнам 5 мс {cv:.2f}, корреляция пар {pair_corr(sample):.3f}"
    if not ev:
        print(f"{head}; событий (> {thr:g}× средней) нет", flush=True)
        return
    starts = [t for t, _ in ev]
    dur = sum(n for _, n in ev) / len(ev) * 5
    ibi = torch.tensor([b - a for a, b in zip(starts, starts[1:])], dtype=torch.float) * 5
    ibi_s = f"{float(ibi.mean()):.0f} мс, CV {float(ibi.std() / ibi.mean()):.2f}" if len(ibi) > 1 else "—"
    amp = sum(float(y[t: t + n].max()) for t, n in ev) / len(ev) / mean

    def share(ts: list[int]) -> float:
        return float(torch.stack([fired[max(0, t - 2): t + 3].any(0).float().mean() for t in ts]).mean())

    gen = torch.Generator().manual_seed(1)
    rand = torch.randint(2, bins - 3, (200,), generator=gen).tolist()
    print(f"{head}\n   событий (> {thr:g}× средней) {len(ev) / sec:.2f}/с, длительность {dur:.1f} мс, "
          f"интервалы {ibi_s}, пик × средней {amp:.1f}, в событии ±10 мс {share(starts):.1%} клеток "
          f"(в случайном окне {share(rand):.1%})", flush=True)


def main(opts: dict) -> None:
    spec = ",".join(x for x in (OLD_LOOP if opts.get("old") == "1" else "", opts.get("path", "")) if x)
    develop.override_paths(spec)
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
    for _ in range(int(opts.get("adapt", 10))):
        b.homeostasis_update(b.observe(1000.0, s._blank(1000.0)))
    b.cfg.homeostasis = False

    levels = {
        "MD": (b.areas["md"].E, 5.0),
        "ПФК L2/3 E": (b.areas["prefrontal"].layers["L23"]["E"], 5.0),
        "ПФК L5 E": (b.areas["prefrontal"].layers["L5"]["E"], 5.0),
        "EC II/III": (b.areas["entorhinal"].E, 2.0),
        "CA3 PC": (b.areas["ca3"].E, 5.0),
    }
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
    picks = {k: torch.randperm(p.size, generator=gen)[:500].to(b.device) + p.start for k, (p, _) in levels.items()}
    bins = int(float(opts.get("record", 6)) * 1000 / 5)
    series = {k: torch.zeros(bins, device=b.device) for k in levels}
    fired = {k: torch.zeros(bins, p.size, dtype=torch.bool, device=b.device) for k, (p, _) in levels.items()}
    sample = {k: torch.zeros(bins, len(picks[k]), device=b.device) for k in levels}
    dropped0 = float(b.dropped)
    cdrop0 = {k: float(x.dropped) for k, x in b.circuits.items()}
    for t in range(bins):
        r = b.observe(5.0, s._blank(5.0)) * 0.005
        for k, (p, _) in levels.items():
            x = r[p.slice]
            series[k][t] = x.sum()
            fired[k][t] = x > 0
            sample[k][t] = r[picks[k]]

    print(f"зерно {cfg.seed}, {'прежняя петля реле ↔ TRN' if opts.get('old') == '1' else 'сборка по умолчанию'}, "
          f"запись {bins * 0.005:.0f} с", flush=True)
    for k, (p, thr) in levels.items():
        report(k, series[k], fired[k], sample[k], thr, p.size, bins)
    lost = float(b.dropped) - dropped0
    print(f"потеряно событий за запись: мозг {lost:.3g}, " + ", ".join(
        f"{k} {float(x.dropped) - cdrop0[k]:.3g}" for k, x in b.circuits.items()), flush=True)


if __name__ == "__main__":
    main(dict(a.split("=", 1) for a in sys.argv[1:]))
