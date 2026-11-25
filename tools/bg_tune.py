"""
Подбор размаха колебаний коры для сети рисунка 1 Lindahl & Hellgren Kotaleski 2016 — их же
процедура (скрипты тега 1.0.0, python/scripts_inhibition):

  (а) eNeuro_fig_01_and_02_sim_inh + manager.get_striatum_inhibition_input: входы C1, C2, CF, CS
      умножаются на силу x = linspace(1, 3) ступенями по 500 мс, меряется средняя частота M2;
  (б) inhibition_gather_results.process: x(M2) интерполяцией, размах upp = x(M2 = freq) − 1,
      freq = 0.45 Гц (eNeuro_fig_defaults: freq_beta = freq_sw = 0.45; tuning_freq_amp_to = M2);
  (в) base_simulate.pert_add_oscillations: вход oscillation2 (меандр) — C1, C2, CF: фон ×
      (amp0 ± upp), CS: фон × (1 ± upp·STN_amp_mod) (CS в amp_base_skip — amp0 = 1); amp0 = 0.975,
      STN_amp_mod = 3, 20 Гц — активация (бета); amp0 = 0.8, STN_amp_mod = 1, 1 Гц — медленные волны;
      низ — max(0, amp0 − upp) (default_params 4337–4344).

Отличия: (а) у авторов — 3000 клеток и пробы подряд в одном прогоне; здесь сеть выгрузки
(10000) и силы — строки пачки после прогрева 1000 мс на фоне (окно 500 мс при силе x).

    PYTHONPATH=tools:. .venv/bin/python tools/bg_tune.py par=~/.brain/lindahl/dump/fig1_Beta_0_10000.json \\
        state=beta out=~/.brain/lindahl/dump/fig1_Beta_0_10000_tuned.json
    (state=sw для fig1_Slow_wave2_1_10000.json; dop=… — тот же размах переносится на файл без дофамина:
     apply=<размах> par=…_dop0.json state=… out=…)
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time

import torch

from bg_lindahl import DT, Net

FREQ_M2 = 0.45
STATES = {"beta": {"amp0": 0.975, "stn": 3.0, "hz": 20.0}, "sw": {"amp0": 0.8, "stn": 1.0, "hz": 1.0}}
INPUTS = ("C1", "C2", "CF", "CS")


def constant_inputs(par: dict) -> dict:
    """Фон без колебаний: каждый вход — его node.rate весь прогон."""

    par = copy.deepcopy(par)
    for p in INPUTS:
        nd = par["node"][p]
        nd["spike_setup"] = [{"idx": None, "rates": [float(nd["rate"])], "times": [1.0], "t_stop": 1e12}]
    return par


def measure(par: dict, dev: str, seed: int, amps: torch.Tensor) -> torch.Tensor:
    warm, win = 1000.0, 500.0
    p = constant_inputs(par)
    B = amps.numel()
    torch.manual_seed(seed)
    net = Net(p, seed, dev, 10, B=B)
    for inp in INPUTS:
        n = net.in_off[inp][1] - net.in_off[inp][0]
        base = float(p["node"][inp]["rate"])
        for row in range(B):
            rows = torch.zeros(B, dtype=torch.bool)
            rows[row] = True
            net.add_gen(inp, torch.arange(n), warm, warm + win, base * (float(amps[row]) - 1.0), rows)
    a, z = net.off["M2"]
    cnt = torch.zeros(B, dtype=torch.float64, device=dev)
    t1 = time.time()
    for k in range(int(round((warm + win) / DT))):
        s = net.step()
        if k * DT >= warm:
            cnt += s[:, a:z].sum(1)
    m2 = (cnt / (z - a) / (win / 1000.0)).cpu()
    print(f"счёт {time.time() - t1:.0f} с", flush=True)
    return m2


def oscillating(par: dict, upp: float, state: str) -> dict:
    """Входы рисунка 1 по pert_add_oscillations: меандр фон × (amp0 ± upp)."""

    st = STATES[state]
    par = copy.deepcopy(par)
    half = 1000.0 / st["hz"] / 2.0
    t_stop = float(par["simu"]["sim_time"])
    times = [i * half for i in range(int(t_stop / half) + 1)]
    for p in INPUTS:
        nd = par["node"][p]
        rate = float(nd["rate"])
        if p == "CS":
            amp0, u = 1.0, upp * st["stn"]
        else:
            amp0, u = st["amp0"], upp
        ru, rd = rate * (amp0 + u), rate * max(0.0, amp0 - u)
        rates = [rd if i % 2 == 0 else ru for i in range(len(times))]
        nd["spike_setup"] = [{"idx": None, "rates": rates, "times": times, "t_stop": t_stop}]
    par["tuning"] = {"upp": upp, "state": state, **st}
    return par


def main(o: dict) -> None:
    par = json.load(open(os.path.expanduser(o["par"])))
    state = o["state"]
    if "apply" in o:
        upp = float(o["apply"])
    else:
        amps = torch.linspace(1.0, 3.0, int(o.get("n", 9)), dtype=torch.float64)
        m2 = measure(par, o.get("device", "cuda"), int(o.get("seed", 1)), amps)
        print("сила x → M2, Гц: " + ", ".join(f"{float(a):.2f}: {float(r):.3f}" for a, r in zip(amps, m2)))
        if not (m2[0] <= FREQ_M2 <= m2[-1]):
            raise SystemExit(f"M2 = {FREQ_M2} Гц вне измеренного диапазона — у авторов здесь NaN")
        i = int(torch.searchsorted(m2.contiguous(), torch.tensor(FREQ_M2, dtype=torch.float64)))
        i = max(1, min(i, len(m2) - 1))
        x0, x1, y0, y1 = float(amps[i - 1]), float(amps[i]), float(m2[i - 1]), float(m2[i])
        x = x0 + (FREQ_M2 - y0) * (x1 - x0) / (y1 - y0)
        upp = x - 1.0
        print(f"x(M2 = {FREQ_M2}) = {x:.3f} → размах {upp:.3f}")
    out = os.path.expanduser(o["out"])
    json.dump(oscillating(par, upp, state), open(out, "w"))
    print(out)


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
