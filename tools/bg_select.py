"""
Авторская проверка каналов BG Lindahl & Hellgren Kotaleski 2016 — выбор из двух действий
(рис. 6–7 статьи, https://doi.org/10.1523/ENEURO.0156-16.2016), на порте tools/bg_lindahl.py.

Как у авторов (код тега 1.0.0, python/core/network/manager.py 701–800, get_input_Go_NoGo;
scripts_inhibition/eNeuro_fig_06_0.2_rest_7x7.py):
  - сеть 20000 (max_size), 2 канала: n_sets = 2 у M1, M2, GI, SN и входов C1, C2; FS, GA, ST — 1;
    other_scenario=True: M1→SN, M2→GI и GI→SN — set-set (канальны и прямой, и непрямой путь);
  - вспышка 100 мс (durations [907, 100]) поверх фона: вход C1 набора k — добавочный пуассон
    rate·(p_k − 1) на клетки range(k, int(0.2·n), 2) (proportion_connected 0.2, burst3_oscillations,
    default_params.py 4318–4336); сети:
      «только D1»  — вспышка в C1 и CF (input_lists[0] = C1, CF),
      «D1 и D2»    — вспышка в C1, C2 и CF; у C2 силы наборов переставлены (other_scenario: набор 0
                     получает p1 — «стоп» сопернику);
    CF — один набор, сила (p0 + p1)/2 на все клетки; CS и, в «только D1», C2 — без вспышки;
  - сетка сил p0, p1 = linspace(1, 3, 7)² (_get_input_go_nogo_p0_and_p1);
  - выбор: средняя частота SNr набора за вспышку ниже порога threshold = 14 Гц (скрипт рис. 6;
    в статье — «ниже 50% покоя SNr»). Исходы: только 0, только 1, оба, ни один.

Отличия от авторов (записаны в ISSUES):
  - фон коры — как в сети рисунка 1 (20 Гц, ±10%, Beta/fig1), а не их подстроенная амплитуда
    (amp_base 0.975 и амплитуда колебаний из прогонов eNeuro_fig_01_and_02_sim_inh — process() —
    у нас этих прогонов нет);
  - пробы — строки пачки (одна сеть, независимые пробы), а не 49 проб подряд в одном прогоне;
    перед вспышкой прогрев warm мс (у авторов — 907 мс покоя после предыдущей пробы).

Запуск (GPU, по одной сети за раз):
    PYTHONPATH=tools:. .venv/bin/python tools/bg_select.py net=d1 seed=1 out=/tmp/bgl/sel_d1_s1
    PYTHONPATH=tools:. .venv/bin/python tools/bg_select.py net=d1d2 seed=1 out=/tmp/bgl/sel_d1d2_s1
    PYTHONPATH=tools:. .venv/bin/python tools/bg_select.py report=/tmp/bgl/sel_d1_s1,/tmp/bgl/sel_d1d2_s1
"""

from __future__ import annotations

import json
import os
import sys
import time

import torch

from bg_lindahl import DT, Net

PAR = "~/.brain/lindahl/dump/fig1_Beta_0_20000.json"
RES = 7
PROP = 0.2          # proportion_connected
THRESHOLD = 14.0    # Гц, threshold скрипта рис. 6
BURST = 100.0       # мс


def go_nogo_par(par: dict, n: int = 2) -> dict:
    """Каналы по get_input_Go_NoGo (other_scenario=True); n — число каналов (у авторов 2)."""

    for p in ("M1", "M2", "GI", "SN", "C1", "C2"):
        par["node"][p]["n_sets"] = n
    for p in ("FS", "GA", "ST"):
        par["node"][p]["n_sets"] = 1
    for c in ("M1_SN_gaba", "M2_GI_gaba", "GI_SN_gaba"):
        par["conn"][c]["rule"] = "set-set"
    return par


def run(opts: dict) -> None:
    dev = opts.get("device", "cuda")
    seed = int(opts.get("seed", 1))
    kind = opts.get("net", "d1")
    warm = float(opts.get("warm", 1500.0))
    torch.manual_seed(seed)
    par = go_nogo_par(json.load(open(os.path.expanduser(opts.get("par", PAR)))))
    v = torch.linspace(1.0, 3.0, RES, dtype=torch.float64)
    yy, xx = torch.meshgrid(v, v, indexing="ij")
    p0, p1 = yy.reshape(-1), xx.reshape(-1)          # p0 — сила набора 0 (y), p1 — набора 1 (x)
    B = p0.numel()
    t0 = time.time()
    net = Net(par, seed, dev, int(opts.get("nsub", 10)), B=B)
    print(f"сборка {time.time() - t0:.0f} с: клеток {net.N}, синапсов {int(net.deg.sum())}, строк {B}", flush=True)
    a, z = warm, warm + BURST
    for inp in ("C1", "C2") if kind == "d1d2" else ("C1",):
        n = net.in_off[inp][1] - net.in_off[inp][0]
        m = int(n * PROP)
        base = float(par["node"][inp]["rate"])
        for k in range(2):
            amp = (p0, p1)[k] if inp == "C1" else (p1, p0)[k]     # C2: силы переставлены (other_scenario)
            idx = torch.arange(k, m, 2)
            for row in range(B):
                rows = torch.zeros(B, dtype=torch.bool)
                rows[row] = True
                net.add_gen(inp, idx, a, z, base * (float(amp[row]) - 1.0), rows)
    ncf = net.in_off["CF"][1] - net.in_off["CF"][0]
    for row in range(B):
        rows = torch.zeros(B, dtype=torch.bool)
        rows[row] = True
        net.add_gen("CF", torch.arange(ncf), a, z, float(par["node"]["CF"]["rate"]) * ((float(p0[row]) + float(p1[row])) / 2 - 1.0), rows)
    # запись: частоты наборов SN, M1, M2, GI по бинам 10 мс
    groups = []
    for p in ("M1", "M2", "GI", "SN"):
        s0, s1 = net.off[p]
        for k in range(2):
            groups.append((f"{p}{k}", torch.arange(s0 + k, s1, 2)))
    G = torch.zeros(net.N, len(groups), dtype=torch.float64)
    for j, (_, idx) in enumerate(groups):
        G[idx, j] = 1.0 / len(idx)
    G = G.to(dev)
    bin_ms = 10.0
    per_bin = int(round(bin_ms / DT))
    T = z + 200.0
    steps = int(round(T / DT))
    hist = torch.zeros(steps // per_bin, B, len(groups), dtype=torch.float64, device=dev)
    t1 = time.time()
    for k in range(steps):
        s = net.step().to(torch.float64)
        hist[k // per_bin] += s @ G
        if k and k % int(round(500.0 / DT)) == 0:
            print(f"  {k * DT:.0f} мс, {time.time() - t1:.0f} с", flush=True)
    hz = (hist / (bin_ms / 1000.0)).cpu()
    names = [g[0] for g in groups]
    tb = (torch.arange(hz.shape[0]) * bin_ms)
    win = (tb >= a) & (tb < z)
    rest = (tb >= a - 500.0) & (tb < a)
    sn0 = hz[win][:, :, names.index("SN0")].mean(0)
    sn1 = hz[win][:, :, names.index("SN1")].mean(0)
    print(f"счёт {time.time() - t1:.0f} с; SNr покой {float(hz[rest][:, :, names.index('SN0')].mean()):.1f} / "
          f"{float(hz[rest][:, :, names.index('SN1')].mean()):.1f} Гц", flush=True)
    out = opts["out"]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.save({"hz": hz, "names": names, "bin_ms": bin_ms, "p0": p0, "p1": p1, "sn0": sn0, "sn1": sn1,
                "burst": (a, z), "net": kind, "seed": seed, "opts": opts}, out + ".pt")
    table(out + ".pt")


def outcome(sn0: torch.Tensor, sn1: torch.Tensor) -> torch.Tensor:
    """0 — ни один, 1 — только 0, 2 — только 1, 3 — оба."""

    s0, s1 = sn0 < THRESHOLD, sn1 < THRESHOLD
    return s0.long() + 2 * s1.long()


def table(path: str) -> None:
    d = torch.load(path, weights_only=False)
    o = outcome(d["sn0"], d["sn1"]).view(RES, RES)
    sym = {0: "·", 1: "0", 2: "1", 3: "Б"}
    v = torch.linspace(1.0, 3.0, RES)
    print(f"{d['net']}, зерно {d['seed']}: строки — сила канала 0 (p0), столбцы — канала 1 (p1); "
          f"0/1 — выбран один, Б — оба, · — ни один")
    print("      " + " ".join(f"{float(x):4.2f}" for x in v))
    for i in range(RES):
        print(f"{float(v[i]):4.2f}  " + " ".join(f"   {sym[int(o[i, j])]}" for j in range(RES)))
    sn0, sn1 = d["sn0"].view(RES, RES), d["sn1"].view(RES, RES)
    print("SNr канала 0 / 1 за вспышку, Гц (диагональ p0 = p1): " +
          ", ".join(f"{float(v[i]):.2f}: {float(sn0[i, i]):.1f}/{float(sn1[i, i]):.1f}" for i in range(RES)))
    n = {k: int((o == k).sum()) for k in range(4)}
    print(f"итого из {RES * RES}: один — {n[1] + n[2]}, оба — {n[3]}, ни один — {n[0]}")


def report(paths: list[str], out: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    cm = ListedColormap(["#d9d9d9", "#2a78d6", "#e8743b", "#202020"])
    fig, axes = plt.subplots(2, len(paths), figsize=(5.2 * len(paths), 9), squeeze=False)
    for j, p in enumerate(paths):
        d = torch.load(p + ".pt", weights_only=False)
        o = outcome(d["sn0"], d["sn1"]).view(RES, RES)
        ax = axes[0, j]
        ax.imshow(o.numpy(), cmap=cm, vmin=0, vmax=3, origin="lower", extent=[0.83, 3.17, 0.83, 3.17])
        ax.set_title(f"{'только D1' if d['net'] == 'd1' else 'D1 и D2'}, зерно {d['seed']}\n"
                     "серый — ни один, синий — 0, оранжевый — 1, чёрный — оба")
        ax.set_xlabel("сила канала 1 (× фона)")
        ax.set_ylabel("сила канала 0 (× фона)")
        ax = axes[1, j]
        hz, names = d["hz"], d["names"]
        t = torch.arange(hz.shape[0]) * d["bin_ms"]
        for i_row, lab in ((RES * RES - 1, "3/3"), (RES * (RES - 1), "3/1"), (RES * 3 + 3, "2/2")):
            for ch, ls in (("SN0", "-"), ("SN1", "--")):
                ax.plot(t, hz[:, i_row, names.index(ch)], ls, label=f"{ch} при {lab}")
        ax.axvspan(*d["burst"], color="#eeeeee")
        ax.axhline(THRESHOLD, color="k", lw=0.6)
        ax.set_xlim(d["burst"][0] - 400, d["burst"][1] + 200)
        ax.set_xlabel("мс")
        ax.set_ylabel("SNr набора, Гц")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    print(out)


if __name__ == "__main__":
    o = dict(kv.split("=", 1) for kv in sys.argv[1:])
    if "report" in o:
        ps = o["report"].split(",")
        for p in ps:
            table(p + ".pt")
        report(ps, o.get("png", ps[0] + "_grid.png"))
    else:
        run(o)
