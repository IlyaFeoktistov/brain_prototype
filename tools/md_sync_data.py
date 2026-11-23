"""
Те же меры синхронности, что tools/md_sync.py, на записях MD бодрствующих
мышей (IBL Brain-wide map, https://doi.org/10.1101/2023.07.04.548681;
открытая база openalyx, ONE-api; данные в ~/.brain/ibl). Клетки — хорошие
(label 1) в MD (Allen CCF 362, 617, 626, 636) одного зонда; пары ближе
min_um по глубине — мимо (ошибки сортировки). Отрезки: спонтанная активность
пассивного протокола (покой наяву, ~300 с) и 300 с задачи с её середины.

    PYTHONPATH=. python tools/md_sync_data.py [min_um=40] [device=cuda]
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

from tools.md_sync import ccg, jitter

SESSIONS = [  # (мышь, eid, зонд) — больше всего хороших клеток MD (перепись 2026-11-23)
    ("ZM_2241", "ee40aece-cffd-4edb-a4b6-155f158c666a", "probe01"),
    ("ZM_2240", "510b1a50-825d-44ce-86f6-9678f5396e02", "probe00"),
    ("DY_010", "bb6a5aae-2431-401d-8f6a-9fdd6de655a9", "probe00"),
]
MD_IDS = (362, 617, 626, 636)


def metrics(x: torch.Tensor, pairs: torch.Tensor, gen) -> dict:
    """x — растр [мс, клетки] (0/1); pairs — маска пар [клетки, клетки]."""

    def pearson(y):
        y = y - y.mean(0)
        sd = y.std(0).clamp_min(1e-12)
        z = y / sd
        cc = (z.T @ z) / (z.shape[0] - 1)
        return float(cc[pairs].mean())

    j = jitter(x, gen=gen)
    x = x[: j.shape[0]]
    w5 = lambda y: y.view(-1, 5, y.shape[1]).sum(1)
    out = {"p": pearson(w5(x)), "pj": pearson(w5(j))}
    for key, y in (("raw", x), ("jit", j)):
        c = ccg(y)
        mid, total = c[92:109].sum(0), c.sum(0)
        ok = pairs & (mid >= 50)
        out[f"sync_{key}"] = float((mid[ok] / total[ok]).mean()) if ok.any() else float("nan")
        out[f"n_{key}"] = int(ok.sum())
        g = c[:, pairs].sum(1)
        out[f"grand_{key}"] = (g / g.sum() * 201).cpu()
    return out


def main(opts: dict) -> None:
    from one.api import ONE

    one = ONE(base_url="https://openalyx.internationalbrainlab.org", username="intbrainlab", password="international",
              silent=True, cache_dir=os.path.expanduser("~/.brain/ibl"))
    dev = torch.device(opts.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    gen = torch.Generator(device=dev).manual_seed(1)
    min_um = float(opts.get("min_um", 40))
    for mouse, eid, probe in SESSIONS:
        c = f"alf/{probe}/pykilosort"
        met = one.load_dataset(eid, "clusters.metrics.pqt", collection=c)
        ch = one.load_dataset(eid, "clusters.channels.npy", collection=c)
        loc = one.load_dataset(eid, "channels.brainLocationIds_ccf_2017.npy", collection=c)
        depth = one.load_dataset(eid, "clusters.depths.npy", collection=c)
        units = np.flatnonzero(np.isin(loc[ch], MD_IDS) & (met["label"].to_numpy() == 1))
        times = one.load_dataset(eid, "spikes.times.npy", collection=c)
        clus = one.load_dataset(eid, "spikes.clusters.npy", collection=c)
        keep = np.isin(clus, units)
        times, clus = times[keep], clus[keep]
        per = one.load_dataset(eid, "_ibl_passivePeriods.intervalsTable.csv", collection="alf").set_index("Unnamed: 0")
        spont = (float(per.loc["start", "spontaneousActivity"]), float(per.loc["stop", "spontaneousActivity"]))
        task_mid = float(per.loc["start", "passiveProtocol"]) / 2
        windows = {"покой (спонтанная)": spont, "задача (середина)": (task_mid - 150, task_mid + 150)}
        d = torch.tensor(depth[units], device=dev)
        far = (d[:, None] - d[None, :]).abs() >= min_um
        idx = {u: i for i, u in enumerate(units)}
        for label, (a, b) in windows.items():
            ms = int((b - a) * 1000)
            sel = (times >= a) & (times < b)
            t = ((times[sel] - a) * 1000).astype(np.int64)
            col = np.array([idx[u] for u in clus[sel]])
            x = torch.zeros(ms, len(units), device=dev)
            x[torch.tensor(t, device=dev).clamp(max=ms - 1), torch.tensor(col, device=dev)] = 1.0
            rate = x.sum(0) / (ms / 1000)
            live = rate >= 1.0
            pairs = torch.triu(far & live[:, None] & live[None, :], 1)
            m = metrics(x, pairs, gen)
            g, gj = m["grand_raw"], m["grand_jit"]
            print(f"{mouse} {label}: клеток MD {int(live.sum())} (≥ 1 Гц, медиана {float(rate[live].median()):.1f} Гц), пар {int(pairs.sum())}; "
                  f"корр. 5 мс {m['p']:.3f} (джиттер {m['pj']:.3f}, избыток {m['p'] - m['pj']:+.3f}); "
                  f"синхронность ±7.5 мс {m['sync_raw']:.3f} по {m['n_raw']} парам (джиттер {m['sync_jit']:.3f}; независимые 0.075)", flush=True)
            print("   общая ККГ / независимые, лаг мс: " + ", ".join(f"{k}: {float(g[100 + k]):.2f}" for k in (0, 2, 5, 10, 20, 25, 50, 100))
                  + f"; после джиттера 0: {float(gj[100]):.2f}", flush=True)


if __name__ == "__main__":
    main(dict(a.split("=", 1) for a in sys.argv[1:]))
