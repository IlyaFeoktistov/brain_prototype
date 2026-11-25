"""
Базальные ганглии Lindahl & Hellgren Kotaleski 2016 на torch — их сеть целиком, как в коде
авторов (шаг 4а, этап «б», б1 — docs/PBWM_TRANSFER.md).

Статья: eNeuro 3:e0156-16.2016, https://doi.org/10.1523/ENEURO.0156-16.2016; код —
https://github.com/mickelindahl/bgmodel, тег 1.0.0 (версия статьи), Python 2 + NEST 2.2.2 и
модуль авторов module-130701 (клетки izhik_cond_exp, my_aeif_cond_exp). Данные — крыса.

Параметры НЕ переписаны руками: их выгружает код авторов (core/network/default_params.py,
классы Inhibition / Beta / Slow_wave2 + возмущения рисунка 1 eNeuro_fig_01_and_02_pert.py) в
python2-докере с заглушкой вместо NEST — ~/.brain/lindahl/dump/dump_fig1.py → JSON. Этот файл
читает JSON и строит сеть по правилам авторов:
  - клетки (8 популяций при размере 10000): M1 (D1 MSN), M2 (D2 MSN), FS (FSN) — izhik_cond_exp;
    GA (GPe TA), GI (GPe TI), GF (третья группа GPe авторов), ST (STN), SN (SNr) —
    my_aeif_cond_exp; уравнения — module-130701/*.cpp (dynamics, update): проводимостные
    синапсы с экспонентой, NMDA с Mg-блоком (Vact, Sact), сброс izhik: V = c (+ u·kc),
    u += d; aeif: V = min(V_reset + u·slope, max_slope), u += b; спайк — проверка V ≥ V_peak
    после каждого подшага (у авторов — после каждого шага адаптивного RKF45);
  - разброс клеток (default_params.py 1055–1080): C_m ~ N(C_m, 0.2·C_m), V_th ~ N(V_th, 1 мВ),
    V_m(0) ~ U(V_th − 20, V_th);
  - входы коры и прочие (C1, C2, CF, CS, EA, EI, EF, ES) — parrot_neuron, у каждого свой
    пуассон по расписанию spike_setup (rates, times); 1-1 к своей клетке; AMPA и NMDA от
    одного входа — одни и те же спайки (my_population.py 720–850);
  - связи all-all / set-set: каждой клетке-приёмнику ровно fan_in разных источников
    (random.sample, fan_in_distribution constant — structure.py 155–175), вес и задержка —
    равномерные на синапс из conn[...]; связи с lesion=True не строятся; рецептор — из
    nest[syn].receptor_type;
  - синапсы Цодыкса (tsodyks_synapse NEST 2.2.2, models/tsodyks_connection.h 181–211:
    u ← u·Puu, x ← x + Pxy·y + Pxz·z, y ← y·Pyy, u ← u + U(1 − u), Δ = u·x, вес·Δ;
    начально x = 1, y = 0, u = 0).

Шаг: 0.1 мс (по умолчанию NEST); внутри — nsub подшагов: проводимости — точная экспонента,
V и u — Эйлер (у авторов RKF45 с точностью 1e-6 — подшаги приближают его; сходимость
проверяется nsub=10 против nsub=20). Входящие спайки прибавляются к проводимостям в конце
шага, как в update() авторов. Дофамин: в норме tata_dop узла = 0 (сеть 0.8 − 0.8), множители
дофамина = 1; другие уровни пока не поддержаны (ошибка).

Опора для сверки (частоты модели в статье — только на графиках, рис. 2–3):
  - расчётные частоты авторов (node.rate, default_params): MSN 0.1, FSN 15, TA 5,
    TI/GF 38.3, STN 10, SNr 30 Гц;
  - цели из текста статьи: MSN 0.01–2, FSN 10–20, SNr 20–35 Гц;
  - Mallet et al. 2008 (в коде авторов scripts_inhibition/base_oscillation.py, mallet2008):
    GPe все — активация 33.6, медленные волны 25.8 Гц; STN — 14.9 / 12.6 Гц.

Запуск (JSON — от dump_fig1.py; i: 0 — активация/бета, 1 — медленные волны):
    PYTHONPATH=tools:. .venv/bin/python tools/bg_lindahl.py par=~/.brain/lindahl/dump/fig1_Beta_0_10000.json \\
        seed=1 t=2000 rec=1000 nsub=10 device=cpu threads=4 out=/tmp/bgl/beta0
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import torch

REC = ["AMPA_1", "NMDA_1", "GABAA_1", "GABAA_2", "GABAA_3", "AMPA_2"]
# receptor_type в выгрузке — номера из заглушки dump.py (AMPA_1=1 … AMPA_2=6), одинаковые для обеих клеток
REC_OF_ID = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
NET = ["M1", "M2", "FS", "GA", "GI", "GF", "ST", "SN"]
DT = 0.1  # мс, разрешение NEST по умолчанию


def _get(d: dict, k: str, default: float) -> float:
    v = d.get(k, default)
    return float(v)


class Net:
    def __init__(self, par: dict, seed: int, dev: str, nsub: int):
        self.dev, self.nsub = dev, nsub
        g = torch.Generator().manual_seed(seed)
        self.g = g
        f64 = dict(dtype=torch.float64)
        node, nest, conn = par["node"], par["nest"], par["conn"]
        self.t_stop = float(par["simu"]["sim_time"])

        # --- клетки сети ---
        self.pops = [p for p in NET if p in node and node[p]["type"] == "network"]
        self.off, n0 = {}, 0
        for p in self.pops:
            self.off[p] = (n0, n0 + int(node[p]["n"]))
            n0 += int(node[p]["n"])
        N = self.N = n0
        V_th_sigma = float(par["netw"].get("V_th_sigma", 1.0))
        rand = par["netw"].get("rand_nodes", {})
        P = {k: torch.zeros(N, **f64) for k in (
            "C_m", "k", "E_L", "V_th", "V_b", "a", "b_1", "b_2", "p_1", "p_2", "c", "d", "V_peak", "u_const",
            "u_kc", "kc_1", "kc_2", "c_max", "u_max", "I_e",
            "g_L", "Delta_T", "V_reset", "V_a", "a_1", "a_2", "tau_w", "b", "rs1", "rs2", "rm1", "rm2")}
        tau = torch.ones(6, N, **f64)
        erev = torch.zeros(6, N, **f64)
        has_rec = torch.zeros(6, N, dtype=torch.bool)
        nact = torch.ones(N, **f64) * -20.0
        sact = torch.ones(N, **f64) * 16.0
        izh = torch.zeros(N, dtype=torch.bool)
        V0 = torch.zeros(N, **f64)
        for p in self.pops:
            a, z = self.off[p]
            m = nest[node[p]["model"]]
            if float(m.get("tata_dop", 0.0)) != 0.0:
                raise NotImplementedError(f"{p}: tata_dop {m['tata_dop']} — дофамин вне нормы не перенесён")
            n = z - a
            is_izh = m["type_id"] == "izhik_cond_exp"
            izh[a:z] = is_izh
            if is_izh:
                for k, dflt in (("k", 1.0), ("E_L", -65.0), ("V_b", -65.0), ("a", 0.02), ("b_1", 0.2),
                                ("b_2", 0.2), ("p_1", 1.0), ("p_2", 1.0), ("c", -65.0), ("d", 8.0),
                                ("V_peak", 30.0), ("u_const", 0.0), ("u_kc", 0.0), ("kc_1", 0.0), ("kc_2", 0.0),
                                ("c_max", 0.0), ("u_max", 100000.0), ("I_e", 0.0)):
                    P[k][a:z] = _get(m, k, dflt)
            else:
                for k, dflt in (("g_L", 30.0), ("E_L", -70.6), ("Delta_T", 2.0), ("V_reset", -60.0),
                                ("V_a", -70.6), ("a_1", 4.0), ("a_2", 4.0), ("tau_w", 144.0), ("b", 80.5),
                                ("V_peak", 0.0), ("I_e", 0.0)):
                    P[k][a:z] = _get(m, k, dflt)
                P["rs1"][a:z] = _get(m, "V_reset_slope1", 0.0)
                P["rs2"][a:z] = _get(m, "V_reset_slope2", 0.0)
                P["rm1"][a:z] = _get(m, "V_reset_max_slope1", 0.0)
                P["rm2"][a:z] = _get(m, "V_reset_max_slope2", 0.0)
            # разброс (default_params.py 1055–1080)
            cm, vth = _get(m, "C_m", 281.0), _get(m, "V_th", -50.4)
            P["C_m"][a:z] = (cm + 0.2 * cm * torch.randn(n, generator=g, **f64)) if rand.get("C_m") else cm
            P["V_th"][a:z] = (vth + V_th_sigma * torch.randn(n, generator=g, **f64)) if rand.get("V_th") else vth
            V0[a:z] = (vth - 20.0 + 20.0 * torch.rand(n, generator=g, **f64)) if rand.get("V_m") else _get(m, "E_L", -70.0)
            for r, name in enumerate(REC):
                if f"{name}_Tau_decay" in m:
                    tau[r, a:z] = float(m[f"{name}_Tau_decay"])
                    erev[r, a:z] = float(m[f"{name}_E_rev"])
                    has_rec[r, a:z] = True
            nact[a:z] = _get(m, "NMDA_1_Vact", -20.0)
            sact[a:z] = _get(m, "NMDA_1_Sact", 16.0)
        if (P["C_m"] <= 0).any():
            raise ValueError("C_m ≤ 0 после разброса")
        self.P = {k: v.to(dev) for k, v in P.items()}
        self.izh = izh.to(dev)
        self.tau, self.erev = tau.to(dev), erev.to(dev)
        self.nact, self.sact = nact.to(dev), sact.to(dev)
        self.V = V0.to(dev)
        self.u = torch.zeros(N, dtype=torch.float64, device=dev)
        self.gs = torch.zeros(6, N, dtype=torch.float64, device=dev)

        # --- входы (parrot + пуассон по расписанию) ---
        self.inputs = [p for p in node if node[p]["type"] == "input"]
        self.in_off, m0 = {}, N
        self.in_sched = []   # (a, z, times[], rates[])
        for p in sorted(self.inputs):
            n = int(node[p]["n"])
            self.in_off[p] = (m0, m0 + n)
            for s in node[p].get("spike_setup") or []:
                idx = s.get("idx") or list(range(n))
                if len(idx) != n:
                    raise NotImplementedError(f"{p}: расписание на часть входов не перенесено")
                self.in_sched.append((m0, m0 + n, [float(x) for x in s["times"]], [float(x) for x in s["rates"]]))
            m0 += n
        self.M = m0  # все источники: сеть + входы

        # --- связи ---
        src, tgt, w, dl, rc, stp_rows = [], [], [], [], [], []
        stp_par = []   # (U, tau_fac, tau_rec, tau_psc) по синапсам Цодыкса
        self.n_syn = {}
        for name in sorted(conn):
            c = conn[name]
            if c.get("lesion"):
                continue
            s_pop, t_pop = c["source"], c["target"]
            ta, tz = self.off[t_pop]
            nt = tz - ta
            sa, sz = (self.off if s_pop in self.off else self.in_off)[s_pop]
            ns = sz - sa
            syn = nest[c["syn"]]
            r = REC_OF_ID[int(syn["receptor_type"])]
            if not has_rec[r, ta]:
                raise ValueError(f"{name}: у {t_pop} нет рецептора {REC[r]}")
            if c["rule"] == "1-1":
                if ns != nt:
                    raise ValueError(f"{name}: 1-1 при {ns} → {nt}")
                s_idx = torch.arange(sa, sz)
                t_idx = torch.arange(ta, tz)
            elif c["rule"] in ("all-all", "set-set"):
                if int(node[s_pop].get("n_sets", 1)) != 1 or int(node[t_pop].get("n_sets", 1)) != 1:
                    raise NotImplementedError(f"{name}: наборы (n_sets > 1) не перенесены")
                if c.get("fan_in_distribution", node[t_pop].get("fan_in_distribution", "constant")) != "constant":
                    raise NotImplementedError(f"{name}: не constant fan_in")
                fan = int(c["fan_in"])
                if fan > ns:
                    raise NotImplementedError(f"{name}: fan_in {fan} > {ns} источников")
                pick = torch.rand(nt, ns, generator=g).argsort(1)[:, :fan]   # без повторов, как random.sample
                s_idx = (pick + sa).reshape(-1)
                t_idx = torch.arange(ta, tz).repeat_interleave(fan)
            else:
                raise NotImplementedError(f"{name}: правило {c['rule']}")
            k = s_idx.numel()
            wd, dd = c["weight"], c["delay"]
            ww = (torch.full((k,), float(wd["params"]), **f64) if wd["type"] == "constant" else
                  wd["params"]["min"] + (wd["params"]["max"] - wd["params"]["min"]) * torch.rand(k, generator=g, **f64))
            de = (torch.full((k,), float(dd["params"]), **f64) if dd["type"] == "constant" else
                  dd["params"]["min"] + (dd["params"]["max"] - dd["params"]["min"]) * torch.rand(k, generator=g, **f64))
            # NEST хранит задержку в шагах (округление к ближайшему), минимум 1 шаг
            ds = torch.clamp(torch.round(de / DT), min=1).long()
            src.append(s_idx); tgt.append(t_idx); w.append(ww); dl.append(ds); rc.append(torch.full((k,), r))
            is_stp = syn["type_id"] == "tsodyks_synapse"
            stp_rows.append(torch.full((k,), is_stp))
            if is_stp:
                stp_par.append(torch.tensor([float(syn["U"]), float(syn["tau_fac"]), float(syn["tau_rec"]),
                                             float(syn["tau_psc"])], **f64).expand(k, 4))
            elif syn["type_id"] != "static_synapse":
                raise NotImplementedError(f"{name}: синапс {syn['type_id']}")
            self.n_syn[name] = k
        src, tgt = torch.cat(src), torch.cat(tgt)
        w, dl, rc = torch.cat(w), torch.cat(dl), torch.cat(rc)
        is_stp = torch.cat(stp_rows)
        order = torch.argsort(src, stable=True)          # CSR по источнику
        src, tgt, w, dl, rc, is_stp = src[order], tgt[order], w[order], dl[order], rc[order], is_stp[order]
        stp_id = torch.full((src.numel(),), -1, dtype=torch.long)
        n_stp = int(is_stp.sum())
        stp_id[is_stp] = torch.arange(n_stp)
        # параметры Цодыкса в порядке CSR: собрать заново по исходному порядку
        if n_stp:
            all_par = torch.zeros(order.numel(), 4, **f64)
            all_par[torch.cat(stp_rows)] = torch.cat(stp_par)
            all_par = all_par[order][is_stp]
        else:
            all_par = torch.zeros(0, 4, **f64)
        deg = torch.bincount(src, minlength=self.M)
        ptr = torch.zeros(self.M + 1, dtype=torch.long)
        ptr[1:] = torch.cumsum(deg, 0)
        self.D = int(dl.max()) + 1
        to = lambda x: x.to(dev)
        self.tgt, self.w, self.dl, self.rc = to(tgt), to(w), to(dl), to(rc)
        self.stp_id, self.ptr, self.deg = to(stp_id), to(ptr), to(deg)
        self.stp = {"U": to(all_par[:, 0]), "fac": to(all_par[:, 1]), "rec": to(all_par[:, 2]), "psc": to(all_par[:, 3]),
                    "x": torch.ones(n_stp, dtype=torch.float64, device=dev),
                    "y": torch.zeros(n_stp, dtype=torch.float64, device=dev),
                    "u": torch.zeros(n_stp, dtype=torch.float64, device=dev),
                    "last": torch.zeros(n_stp, dtype=torch.float64, device=dev)}
        self.ring = torch.zeros(self.D, 6, N, dtype=torch.float64, device=dev)
        self.k = 0

    def _in_rates(self, t: float) -> torch.Tensor:
        r = torch.zeros(self.M - self.N, dtype=torch.float64)
        for a, z, times, rates in self.in_sched:
            cur = 0.0
            for tt, rr in zip(times, rates):
                if t >= tt:
                    cur = rr
            r[a - self.N:z - self.N] = cur
        return r.to(self.dev)

    def _deliver(self, counts: torch.Tensor, t_spk: float) -> None:
        act = torch.nonzero(counts).squeeze(1)
        if act.numel() == 0:
            return
        n = self.deg[act]
        tot = int(n.sum())
        if tot == 0:
            return
        start = self.ptr[act]
        first = torch.cumsum(n, 0) - n
        j = torch.repeat_interleave(start - first, n) + torch.arange(tot, device=self.dev)
        c = torch.repeat_interleave(counts[act].to(torch.float64), n)
        val = self.w[j] * c
        sid = self.stp_id[j]
        m = sid >= 0
        if m.any():
            s = sid[m]
            S = self.stp
            h = t_spk - S["last"][s]
            Puu = torch.where(S["fac"][s] == 0, torch.zeros_like(h), torch.exp(-h / S["fac"][s].clamp(min=1e-30)))
            Pyy = torch.exp(-h / S["psc"][s])
            Pzz = torch.exp(-h / S["rec"][s])
            Pxy = ((Pzz - 1.0) * S["rec"][s] - (Pyy - 1.0) * S["psc"][s]) / (S["psc"][s] - S["rec"][s])
            Pxz = 1.0 - Pzz
            x, y, u = S["x"][s], S["y"][s], S["u"][s]
            z = 1.0 - x - y
            u = u * Puu
            x = x + Pxy * y + Pxz * z
            y = y * Pyy
            u = u + S["U"][s] * (1.0 - u)
            dy = u * x
            S["x"][s], S["y"][s], S["u"][s] = x - dy, y + dy, u
            S["last"][s] = t_spk
            val[m] = self.w[j[m]] * dy
        slot = (self.k + self.dl[j]) % self.D
        flat = (slot * 6 + self.rc[j]) * self.N + self.tgt[j]
        self.ring.view(-1).index_add_(0, flat, val)

    def step(self) -> torch.Tensor:
        """Шаг 0.1 мс; возвращает спайки сети [N] (0/1)."""

        P, V, u, gs = self.P, self.V, self.u, self.gs
        h = DT / self.nsub
        dec = torch.exp(-h / self.tau)
        spk = torch.zeros(self.N, dtype=torch.bool, device=self.dev)
        iz = self.izh
        for _ in range(self.nsub):
            mg = 1.0 / (1.0 + torch.exp((self.nact - V) / self.sact))
            I_syn = -(gs[0] * (V - self.erev[0]) + gs[1] * (V - self.erev[1]) * mg + gs[2] * (V - self.erev[2])
                      + gs[3] * (V - self.erev[3]) + gs[4] * (V - self.erev[4]) + gs[5] * (V - self.erev[5]))
            # izhik_cond_exp
            dv_i = (P["k"] * (V - P["E_L"]) * (V - P["V_th"]) - u + I_syn + P["I_e"]) / P["C_m"]
            lo = V < P["V_b"]
            dvb = V - P["V_b"]
            du_i = P["a"] * (torch.where(lo, P["b_1"] * torch.pow(dvb, P["p_1"]), P["b_2"] * torch.pow(dvb.clamp(min=0), P["p_2"]))
                             - u + P["u_const"])
            # my_aeif_cond_exp (экспонента — от V не выше V_peak, как при остановке шага на пике)
            Ve = torch.minimum(V, P["V_peak"])
            dt_ = torch.where(iz, torch.ones_like(V), P["Delta_T"])
            i_sp = P["Delta_T"] * torch.exp((Ve - P["V_th"]) / dt_)
            dv_a = (-P["g_L"] * ((V - P["E_L"]) - i_sp) - u + P["I_e"] + I_syn) / P["C_m"]
            du_a = (torch.where(V < P["V_a"], P["a_1"], P["a_2"]) * (V - P["V_a"]) - u) / torch.where(iz, torch.ones_like(V), P["tau_w"])
            V = V + h * torch.where(iz, dv_i, dv_a)
            u = u + h * torch.where(iz, du_i, du_a)
            gs = gs * dec
            fire = V >= P["V_peak"]
            if fire.any():
                # izhik: V = c + u·kc (kc по u_kc), не выше c_max; u += d (дофамин 0), не выше u_max
                v_i = P["c"] + u * torch.where(u < P["u_kc"], P["kc_1"], P["kc_2"])
                v_i = torch.where(v_i > P["c_max"], P["c_max"], v_i)
                # aeif: V = min(V_reset + u·slope, max_slope) по знаку u
                v_a = torch.where(u < 0, torch.minimum(P["V_reset"] + u * P["rs1"], P["rm1"]),
                                  torch.minimum(P["V_reset"] + u * P["rs2"], P["rm2"]))
                V = torch.where(fire, torch.where(iz, v_i, v_a), V)
                u = torch.where(fire, torch.where(iz, torch.minimum(u + P["d"], P["u_max"]), u + P["b"]), u)
                spk |= fire
        slot = self.k % self.D
        gs = gs + self.ring[slot]
        self.ring[slot].zero_()
        self.V, self.u, self.gs = V, u, gs
        # спайки этого шага (сеть + входы) уходят с задержкой
        t_now = (self.k + 1) * DT
        lam = self._in_rates(self.k * DT) * (DT * 1e-3)
        cin = torch.poisson(lam, generator=self.g if self.dev == "cpu" else None)
        counts = torch.cat([spk.to(torch.float64), cin])
        self._deliver(counts, t_now)
        self.k += 1
        return spk


def main(opts: dict) -> None:
    torch.set_num_threads(int(opts.get("threads", 4)))
    par = json.load(open(os.path.expanduser(opts["par"])))
    seed = int(opts.get("seed", 1))
    dev = opts.get("device", "cpu")
    t0 = time.time()
    net = Net(par, seed, dev, int(opts.get("nsub", 10)))
    T = float(opts.get("t", net.t_stop))
    rec0 = float(opts.get("rec", 1000.0))
    print(f"сборка {time.time() - t0:.0f} с: клеток {net.N}, входов {net.M - net.N}, синапсов {int(net.deg.sum())}, "
          f"задержка до {net.D - 1} шагов", flush=True)
    steps = int(round(T / DT))
    bin_ms = 5.0
    per_bin = int(round(bin_ms / DT))
    n_bins = steps // per_bin
    pop_of = torch.zeros(net.N, dtype=torch.long)
    for i, p in enumerate(net.pops):
        a, z = net.off[p]
        pop_of[a:z] = i
    pop_of = pop_of.to(dev)
    hist = torch.zeros(n_bins, len(net.pops), dtype=torch.float64, device=dev)
    cnt = torch.zeros(net.N, dtype=torch.float64, device=dev)
    t1 = time.time()
    for k in range(steps):
        s = net.step()
        b = k // per_bin
        if b < n_bins:
            hist[b].index_add_(0, pop_of, s.to(torch.float64))
        if k * DT >= rec0:
            cnt += s
        if k and k % int(round(500.0 / DT)) == 0:
            print(f"  {k * DT:.0f} мс, {time.time() - t1:.0f} с", flush=True)
    dur = (T - rec0) / 1000.0
    print(f"счёт {time.time() - t1:.0f} с; частоты {rec0:.0f}–{T:.0f} мс (Гц на клетку):", flush=True)
    rates = {}
    for i, p in enumerate(net.pops):
        a, z = net.off[p]
        r = cnt[a:z] / dur
        rates[p] = float(r.mean())
        print(f"  {p}: {float(r.mean()):7.2f}  (медиана {float(r.median()):6.2f}, молчат {float((r == 0).double().mean()) * 100:4.0f}%)", flush=True)
    out = opts.get("out")
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        sizes = torch.tensor([net.off[p][1] - net.off[p][0] for p in net.pops], dtype=torch.float64, device=dev)
        hz = (hist / sizes / (bin_ms / 1000.0)).cpu()
        torch.save({"rates": rates, "hist": hz, "pops": net.pops, "bin_ms": bin_ms, "opts": opts}, out + ".pt")
        import plots
        t_ms = (torch.arange(n_bins) + 1) * bin_ms
        groups = [("стриатум", ["M1", "M2", "FS"]), ("GPe", ["GA", "GI", "GF"]), ("STN, SNr", ["ST", "SN"])]
        panels = [{"label": f"{lab}\nГц на клетку", "series": {p: hz[:, net.pops.index(p)] for p in ps if p in net.pops},
                   "smooth": 4} for lab, ps in groups]
        plots.timeline(out + ".png", t_ms, panels, title=f"BG Lindahl 2016 (порт): {os.path.basename(opts['par'])}, зерно {seed}")
        print(out + ".png")


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
