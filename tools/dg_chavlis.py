"""
Эталон зубчатой извилины Chavlis, Petrantonakis & Poirazi 2017 на torch — их сеть
целиком, как в коде авторов (CHAT_PLAN шаг 3, цикл 36).

Статья: Hippocampus 27:89, https://doi.org/10.1002/hipo.22675; код — ModelDB 206372
(https://github.com/ModelDBRepository/206372), DG_12dendrites.py, Brian 1.4 /
Python 2. Контроль авторов — модель с 12 дендритами: при входе 10% клеток EC
по 40 Гц активны ~5% гранулярных (у 3 дендритов — 20–70%).

Что перенесено (числа — из DG_12dendrites.py, строки в скобках у кода ниже):
  - 400 EC (вход), 2000 гранулярных (GC), 100 корзинчатых (BC), 80 мшистых
    (MC), 40 HIPP; GC — сома + 21 дендритный отсек (3 ветки: 4 дистальных, 2
    средних, 1 проксимальный), пассивные, AHP на соме; BC, MC, HIPP — AdEx
    (Brette–Gerstner), порог — vm > VT;
  - синапс — своё состояние у каждого синапса: AMPA dy/dt = −y/τd + α h (1 − y),
    dh/dt = −h/τr; NMDA так же (j, x) с Mg-блоком; ГАМК (z, r); спайк пре
    после задержки: h += w (у шума — h = w); проводимость — g × Σ состояний;
  - связи: EC → GC (12 дистальных отсеков) и EC → HIPP, GC → MC, MC → GC
    (3 проксимальных), HIPP → GC (12 дистальных) — матрицы авторов
    (ConnectivityMatrices_12dendrites, syn_*.txt — архивы numpy Brian 1);
    GC → BC и BC → GC — внутри кластера (20 GC ↔ своя BC, все со всеми); MC →
    все BC;
  - шум: GC — 500 пуассоновских источников по 0.1 Гц, на каждый из 21
    отсеков с долей 1/21, задержка U(0, 10) мс; BC — 20 своих по 3 Гц, MC —
    20 своих по 3.8 Гц, HIPP — 20 своих по 3 Гц;
  - протокол: 300 мс EC молчит, 500 мс 40 из 400 клеток EC — пуассон 40 Гц,
    10 мс; шаг 0.1 мс; клетки — Эйлер (Brian 1 для нелинейных), синапсы —
    экспоненциальный Эйлер (implicit=True у авторов).

Рефрактерность — как у Brian 1 при сбросе-строке и AdaptiveReset: только
запрет спайка, без удержания потенциала. Отличия от оригинала: случайные числа свои (связи — авторов из файлов;
активные клетки EC, их спайки, шум, кластеры — по тем же правилам).

Счёт — класс мозга brain/dg_chavlis.py (тот же код идёт в мозг), здесь —
протокол авторов на шаге 0.1 мс; device=cuda dtype=32 — как в мозге.

    PYTHONPATH=. python tools/dg_chavlis.py [seed=1] [active=файл.npy] [stim_hz=40] [pre_ms=300] [stim_ms=500] [device=cpu] [dtype=64] [out=файл.npy]
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import torch

from brain import circuit
from brain.dg_chavlis import SUB_DT, ChavlisDG


def main(opts: dict) -> None:
    seed = int(opts.get("seed", 1))
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(opts.get("threads", 2)))
    dev = torch.device(opts.get("device", "cpu"))
    dtype = torch.float64 if opts.get("dtype", "64") == "64" else torch.float32
    gen = torch.Generator(device=dev).manual_seed(seed)
    net = circuit.load("hippocampus/dg_chavlis.json")
    n_gc = net["groups"]["GC"]["cells"]
    n_int = sum(net["groups"][g]["cells"] for g in ("BC", "MC", "HIPP"))
    n, n_in = n_gc + n_int, int(net["ec_cells"])
    area = SimpleNamespace(E=SimpleNamespace(start=0, size=n_gc, stop=n_gc),
                           I=SimpleNamespace(start=n_gc, size=n_int, stop=n))
    c = ChavlisDG(net, area, SUB_DT, dev, gen, torch.arange(n, n + n_in), dtype=dtype)
    print(f"синапсов {c.synapses}", flush=True)

    pre_ms, stim_ms, post_ms = float(opts.get("pre_ms", 300)), float(opts.get("stim_ms", 500)), 10.0
    stim_hz = float(opts.get("stim_hz", 40))
    T = int(round((pre_ms + stim_ms + post_ms) / SUB_DT))
    t_pre, t_stim = int(round(pre_ms / SUB_DT)), int(round((pre_ms + stim_ms) / SUB_DT))

    # EC: активные 10% (40 клеток), пуассон stim_hz в окне стимула (стр. 117–122).
    if opts.get("active"):
        active = np.load(opts["active"]).astype(np.int64)
    else:
        active = np.sort(rng.choice(n_in, int(0.1 * n_in), replace=False))
    ec_rate = torch.zeros(n + n_in, device=dev, dtype=dtype)
    ec_rate[n + torch.as_tensor(active)] = stim_hz * SUB_DT / 1000.0
    zero = torch.zeros(n, device=dev)

    counts = torch.zeros(n, dtype=torch.long, device=dev)
    counts_rest = torch.zeros_like(counts)
    for t in range(T):
        spikes = (torch.rand(n + n_in, device=dev, dtype=dtype) < ec_rate) if t_pre <= t < t_stim else torch.zeros(n + n_in, dtype=torch.bool, device=dev)
        _, fired = c.step(zero, None, spikes)
        counts += fired.long()
        if t < t_pre:
            counts_rest += fired.long()
        if t % 1000 == 0:
            print(f"  {t * SUB_DT:.0f} мс: GC спайков {int(counts[:n_gc].sum())}", flush=True)

    sec_rest, sec_stim = pre_ms / 1000.0, (stim_ms + post_ms) / 1000.0
    print(f"зерно {seed}, вход {stim_hz:.0f} Гц у {len(active)} клеток EC из {n_in}")
    for i, g in enumerate(c.names):
        m = c.type_of == i
        c_all, c_rest = counts[m], counts_rest[m]
        c_stim = c_all - c_rest
        k = int(m.sum())
        print(f"{g}: покой {float(c_rest.sum()) / k / sec_rest:.2f} Гц (активных {float((c_rest > 0).float().mean()):.1%}), "
              f"стимул {float(c_stim.sum()) / k / sec_stim:.2f} Гц (активных {float((c_stim > 0).float().mean()):.1%})")
    if opts.get("out"):
        np.save(opts["out"], counts[:n_gc].cpu().numpy())


if __name__ == "__main__":
    main(dict(a.split("=", 1) for a in sys.argv[1:]))
