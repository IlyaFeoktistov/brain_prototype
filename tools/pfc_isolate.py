"""
Колонка ПФК сама по себе или от мозга (CHAT_PLAN шаг 3, цикл 40, этап A):
воспроизводится ли колебание популяционного выхода ПФК без обратных
влияний остального мозга при тех же входах.

Мозг — как в опыте B (organs=words): разгон 2.5 с, адаптация с гомеостазом
окнами по 1 с (сдвиги гомеостаза остаются), снимок; затем из одного снимка
запись с шагом 1 мс:
  1) мозг целиком;
  2) ПФК изолирована: все клетки вне ПФК — независимые пуассоновские
     источники со своими средними частотами из записи 1 (engine.clamp_p).
     Проводка, веса, задержки, синапсы, среднее поле, фон, гомеостаз,
     вещества — те же; средний вход ПФК тот же, общих колебаний извне и
     ответа мозга на ПФК нет.

Меры по популяциям ПФК (слой × тип): частота; CV популяционной активности
по окнам 5 мс (у независимых клеток ≈ 1/√(N·r·0.005)); корреляция спайков
пар (окно 5 мс, до 200 активных клеток); CV ISI клеток (≥ 5 спайков);
спектр популяционной активности E (окна 1 мс, отрезки 256 мс) — частота
пика 5–200 Гц и пик / медиана мощности.

    PYTHONPATH=. python tools/pfc_isolate.py [seed=42] [adapt=30] [record=3] [area=prefrontal] [live=md+trn_md;association] [addpath=trn_md->trn_md:47:-0.12:1]
"""

from __future__ import annotations

import sys

import torch

from brain.config import BrainConfig
from brain.engine import KINDS
from brain.mind import Mind
from tools import develop


def pair_corr(x: torch.Tensor) -> float:
    x = x - x.mean(0)
    sd = x.std(0)
    live = sd > 0
    if int(live.sum()) < 2:
        return float("nan")
    x = x[:, live] / sd[live]
    cc = (x.T @ x) / (x.shape[0] - 1)
    n = cc.shape[0]
    return float(cc[~torch.eye(n, dtype=torch.bool, device=cc.device)].mean())


def cv_isi(spk: torch.Tensor) -> float:
    """Средний CV межспайковых интервалов клеток с ≥ 5 спайками (spk [T, N], 1 мс)."""

    out = []
    for j in range(spk.shape[1]):
        t = torch.nonzero(spk[:, j]).squeeze(1).float()
        if len(t) >= 5:
            d = t[1:] - t[:-1]
            out.append(float(d.std() / d.mean()))
    return sum(out) / len(out) if out else float("nan")


def spectrum(y: torch.Tensor) -> tuple[float, float]:
    """Пик спектра популяционной активности (окна 1 мс) в 5–200 Гц и пик / медиана."""

    seg = 256
    n = len(y) // seg
    if n == 0:
        return float("nan"), float("nan")
    x = y[: n * seg].view(n, seg)
    x = x - x.mean(1, keepdim=True)
    p = (torch.fft.rfft(x * torch.hann_window(seg, device=x.device), dim=1).abs() ** 2).mean(0)
    f = torch.fft.rfftfreq(seg, d=0.001).to(x.device)
    band = (f >= 5) & (f <= 200)
    pb, fb = p[band], f[band]
    i = int(pb.argmax())
    return float(fb[i]), float(pb[i] / pb.median())


def record(b, s, ms: int, pops: dict, pick: dict) -> tuple[torch.Tensor, dict]:
    dev = b.device
    total = None
    spk = {k: torch.zeros(ms, len(v), dtype=torch.bool, device=dev) for k, v in pick.items()}
    pop = {k: torch.zeros(ms, device=dev) for k in pops}
    for t in range(ms):
        r = b.observe(1.0, s._blank(1.0))
        total = r.clone() if total is None else total + r
        fired = r > 0
        for k, sl in pops.items():
            pop[k][t] = fired[sl].float().sum()
        for k, idx in pick.items():
            spk[k][t] = fired[idx]
    return total / ms, {"pop": pop, "spk": spk}


def report(label: str, rec: dict, pops: dict, sizes: dict, ms: int) -> None:
    print(f"\n{label}:")
    print("   популяция      Гц    CV поп. 5 мс (незав.)  корр. пар 5 мс  CV ISI")
    for k in pops:
        y = rec["pop"][k]
        n = sizes[k]
        hz = float(y.sum()) / n / (ms / 1000)
        y5 = y[: ms // 5 * 5].view(-1, 5).sum(1)
        cv = float(y5.std() / y5.mean()) if float(y5.mean()) > 0 else float("nan")
        cv0 = 1.0 / max(1e-9, n * hz * 0.005) ** 0.5
        line = f"   {k:12s} {hz:6.2f}   {cv:5.2f} ({cv0:4.2f})"
        if k in rec["spk"]:
            x = rec["spk"][k].float()
            x5 = x[: ms // 5 * 5].view(-1, 5, x.shape[1]).sum(1)
            active = x.sum(0) > 0
            line += f"         {pair_corr(x5[:, active][:, :200]):6.3f}        {cv_isi(rec['spk'][k]):4.2f}"
        print(line, flush=True)
    for k in ("E все", "L5.E", "L23.E", "MD", "TRN_MD"):
        if k in rec["pop"]:
            f, ratio = spectrum(rec["pop"][k])
            print(f"   спектр {k}: пик {f:.0f} Гц, пик / медиана {ratio:.1f}", flush=True)


def main(opts: dict) -> None:
    cfg = BrainConfig()
    cfg.size = "small"
    cfg.seed = int(opts.get("seed", 42))
    cfg.autosave = False
    cfg.homeostasis = True
    cfg.event_overflow = "warn"
    cfg.organs = ("words",)
    develop.override_external(cfg, opts.get("extpop", ""))
    develop.add_paths(opts.get("addpath", ""))
    b = Mind(cfg).brain
    b.plasticity = False
    s = develop.Silence(b)
    name = opts.get("area", "prefrontal")
    area = b.areas[name]
    L = area.layers

    b.run(2500.0, s._blank(2500.0))
    for _ in range(int(opts.get("adapt", 30))):
        b.homeostasis_update(b.observe(1000.0, s._blank(1000.0)))
    b.cfg.homeostasis = False
    # Запись с шагом 1 мс (как tools/md_sync.py): кусок шага движка → 2 шага.
    b.CHUNK = int(round(1.0 / b.cfg.dt_ms))
    b._graphs.clear()

    pops = {f"{l}.{k}": g[k].slice for l, g in L.items() for k in KINDS if k in g}
    sizes = {k: sl.stop - sl.start for k, sl in pops.items()}
    e_all = torch.cat([torch.arange(g["E"].start, g["E"].stop) for g in L.values()])
    gen = torch.Generator().manual_seed(cfg.seed)
    pick = {}
    for k in ("L23.E", "L5.E", "L23.PV", "L5.PV"):
        sl = pops[k]
        pick[k] = (torch.randperm(sl.stop - sl.start, generator=gen)[:300] + sl.start).to(b.device)
    for z in ("md", "trn_md"):
        e = b.areas[z].E
        pick[z.upper()] = (torch.randperm(e.size, generator=gen)[:200] + e.start).to(b.device)
    ms = int(float(opts.get("record", 3)) * 1000)
    saved = b._snapshot()

    # Популяция «все E» — через маску.
    mask_e = torch.zeros(b.n, dtype=torch.bool)
    mask_e[e_all] = True
    pops_ext = dict(pops)
    sizes["E все"] = int(mask_e.sum())
    pops_ext["E все"] = mask_e.to(b.device)
    for z in ("md", "trn_md"):
        if z in b.areas:
            pops_ext[z.upper()] = b.areas[z].E.slice
            sizes[z.upper()] = b.areas[z].E.size

    rates, rec = record(b, s, ms, pops_ext, pick)
    report("мозг целиком", rec, pops_ext, sizes, ms)

    # Изоляция: вне ПФК — пуассон со своей средней частотой.
    b._restore(saved)
    cells = area.cells
    outside = torch.ones(b.spikes.numel(), dtype=torch.bool, device=b.device)
    outside[cells] = False
    outside[b.n:] = False
    b.clamp_mask.copy_(outside)
    p = torch.zeros(b.spikes.numel(), device=b.device)
    p[: rates.numel()] = (rates * (b.cfg.dt_ms / 1000.0)).clamp(max=1.0)
    b.clamp_p = p
    b.clamp_on = True
    b.recompile()
    b.CHUNK = int(round(1.0 / b.cfg.dt_ms))
    _, rec2 = record(b, s, ms, pops_ext, pick)
    report("ПФК изолирована (вне ПФК — пуассон со средними частотами)", rec2, pops_ext, sizes, ms)

    # Этап C: вне ПФК — пуассон, кроме живых зон (live=md+trn_md;association);
    # «!» в начале группы — и сама ПФК пуассон (живы только зоны группы).
    for group in filter(None, opts.get("live", "").split(";")):
        b._restore(saved)
        mask = outside.clone()
        if group.startswith("!"):
            group = group[1:]
            mask[cells] = True
        for z in group.split("+"):
            mask[b.areas[z].cells] = False
        b.clamp_mask.copy_(mask)
        b.recompile()
        b.CHUNK = int(round(1.0 / b.cfg.dt_ms))
        _, rec3 = record(b, s, ms, pops_ext, pick)
        label = f"живые только {group} (ПФК и остальное — пуассон)" if bool(mask[cells].all()) else f"ПФК + живые {group} (остальное — пуассон)"
        report(label, rec3, pops_ext, sizes, ms)


if __name__ == "__main__":
    main(dict(a.split("=", 1) for a in sys.argv[1:]))
