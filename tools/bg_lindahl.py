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

Пачка: Net(..., B) — B независимых проб на одних связях (свои клетки, синапсы Цодыкса, входы;
add_gen — свой генератор по строкам); кольцо задержек float32. B=1 на CPU совпадает с прежним
прогоном до сотых Гц; на GPU B=16 — ×9 к CPU, сеть 20000 при B=49 — 4.3 ГБ.
Наборы (каналы действий): n_sets у узла, правила set-set / all-all, пары наборов — zip, как в
structure.py (у GF 1 набор при 2 у M2 — весь GF получает только набор 0 M2, как у авторов).

Скорость (флаги, точный режим — по умолчанию): fp32=1 — состояние клеток во float32 (на игровой карте
float64 в 32–64 раза медленнее), compile=1 — подшаги слиты torch.compile. Сеть 10000, B=32, GPU:
float64 — 23 мс на шаг (233 с на 1 с модели), fp32+compile — 2.7 мс (27 с); частоты за 1–2 с
совпали с float64 в пределах разброса проб (SNr 29.1 против 30.1, прочие ≤ 2%).

Шаг: 0.1 мс (по умолчанию NEST); внутри — nsub подшагов: проводимости — точная экспонента,
V и u — Эйлер (у авторов RKF45 с точностью 1e-6 — подшаги приближают его; сходимость
проверяется nsub=10 против nsub=20). Входящие спайки прибавляются к проводимостям в конце
шага, как в update() авторов. Дофамин: tata_dop узла (сеть − норма 0.8: 0 в норме, −0.8 без дофамина) —
множители на токи рецепторов, E_L, V_b/V_a (в формуле; ветка — по исходному), k, d; дофамин в
связях (меньше коллатералей MSN и т. п.) уже посчитан в fan_in выгрузки.

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


def _pick_sources(nt: int, ns: int, fan: float, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Источники для каждого из nt приёмников из ns (structure.py 155–175, fan_in_distribution constant):
    p = fan/ns < 1 — ровно int(fan) разных (random.sample); p ≥ 1 — пока p > 0: binomial(ns, min(p, 1))
    разных, p −= 1 (источник может повториться). Возвращает (номер источника, номер приёмника).
    """

    p = fan / ns
    if p < 1:
        k = int(fan)
        pick = torch.rand(nt, ns, generator=g).argsort(1)[:, :k]
        return pick.reshape(-1), torch.arange(nt).repeat_interleave(k)
    s_out, t_out = [], []
    for t in range(nt):
        q = p
        while q > 0:
            m = int(torch.binomial(torch.tensor(float(ns)), torch.tensor(min(q, 1.0)), generator=g))
            s_out.append(torch.randperm(ns, generator=g)[:m])
            t_out.append(torch.full((m,), t))
            q -= 1
    return torch.cat(s_out), torch.cat(t_out)


def _sets(n: int, n_sets: int, kind: str) -> list[torch.Tensor]:
    """Наборы популяции по правилу авторов (structure.get_sets): all — вся; set — slice(k, n, n_sets)."""

    if kind == "all":
        return [torch.arange(n)]
    if kind == "set":
        return [torch.arange(k, n, n_sets) for k in range(n_sets)]
    raise NotImplementedError(f"правило набора {kind}")


def _get(d: dict, k: str, default: float) -> float:
    v = d.get(k, default)
    return float(v)


def _substeps(P: dict, V, u, gs, iz, E, F, nact, sact, dec, nsub: int, h: float):
    """nsub подшагов Эйлера (V, u) и точной экспоненты (проводимости) с проверкой пика после каждого.
    Возвращает V, u, gs, спайки [B, N]. Сброс — через where без ветвления (без ожидания карты)."""

    spk = torch.zeros_like(V, dtype=torch.bool)
    for _ in range(nsub):
        mg = 1.0 / (1.0 + torch.exp((nact - V) / sact))
        I_syn = -(F[0] * gs[0] * (V - E[0]) + F[1] * gs[1] * (V - E[1]) * mg + F[2] * gs[2] * (V - E[2])
                  + F[3] * gs[3] * (V - E[3]) + F[4] * gs[4] * (V - E[4]) + F[5] * gs[5] * (V - E[5]))
        # izhik_cond_exp
        dv_i = (P["k"] * (V - P["E_L"]) * (V - P["V_th"]) - u + I_syn + P["I_e"]) / P["C_m"]
        lo = V < P["V_b"]
        dvb = V - P["V_bf"]
        du_i = P["a"] * (torch.where(lo, P["b_1"] * torch.pow(dvb, P["p_1"]), P["b_2"] * torch.pow(dvb.clamp(min=0), P["p_2"]))
                         - u + P["u_const"])
        # my_aeif_cond_exp (экспонента — от V не выше V_peak, как при остановке шага на пике)
        Ve = torch.minimum(V, P["V_peak"])
        dt_ = torch.where(iz, torch.ones_like(P["Delta_T"]), P["Delta_T"])
        i_sp = P["Delta_T"] * torch.exp((Ve - P["V_th"]) / dt_)
        dv_a = (-P["g_L"] * ((V - P["E_L"]) - i_sp) - u + P["I_e"] + I_syn) / P["C_m"]
        du_a = (torch.where(V < P["V_a"], P["a_1"], P["a_2"]) * (V - P["V_af"]) - u) / torch.where(iz, torch.ones_like(P["tau_w"]), P["tau_w"])
        V = V + h * torch.where(iz, dv_i, dv_a)
        u = u + h * torch.where(iz, du_i, du_a)
        gs = gs * dec
        fire = V >= P["V_peak"]
        # izhik: V = c + u·kc (kc по u_kc), не выше c_max; u += d·(1 − beta_d·tata_dop), не выше u_max
        v_i = P["c"] + u * torch.where(u < P["u_kc"], P["kc_1"], P["kc_2"])
        v_i = torch.where(v_i > P["c_max"], P["c_max"], v_i)
        # aeif: V = min(V_reset + u·slope, max_slope) по знаку u
        v_a = torch.where(u < 0, torch.minimum(P["V_reset"] + u * P["rs1"], P["rm1"]),
                          torch.minimum(P["V_reset"] + u * P["rs2"], P["rm2"]))
        V = torch.where(fire, torch.where(iz, v_i, v_a), V)
        u = torch.where(fire, torch.where(iz, torch.minimum(u + P["d"], P["u_max"]), u + P["b"]), u)
        spk = spk | fire
    return V, u, gs, spk


_substeps_compiled = None


class Net:
    """
    Сеть авторов пачкой: B строк — независимые пробы на одних связях (свои состояния клеток,
    синапсов Цодыкса и входов). Строка = отдельный прогон NEST той же сети.
    """

    def __init__(self, par: dict, seed: int, dev: str, nsub: int, B: int = 1, ring_dtype=torch.float32,
                 compile_: bool = False, fp32: bool = False):
        self.dev, self.nsub, self.B = dev, nsub, B
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
        self.n_sets = {p: int(node[p].get("n_sets", 1)) for p in node}
        V_th_sigma = float(par["netw"].get("V_th_sigma", 1.0))
        rand = par["netw"].get("rand_nodes", {})
        P = {k: torch.zeros(N, **f64) for k in (
            "C_m", "k", "E_L", "V_th", "V_b", "V_bf", "V_af", "a", "b_1", "b_2", "p_1", "p_2", "c", "d", "V_peak", "u_const",
            "u_kc", "kc_1", "kc_2", "c_max", "u_max", "I_e",
            "g_L", "Delta_T", "V_reset", "V_a", "a_1", "a_2", "tau_w", "b", "rs1", "rs2", "rm1", "rm2")}
        tau = torch.ones(6, N, **f64)
        dopf = torch.ones(6, N, **f64)   # 1 + beta_I_<рецептор>·tata_dop
        erev = torch.zeros(6, N, **f64)
        has_rec = torch.zeros(6, N, dtype=torch.bool)
        nact = torch.ones(N, **f64) * -20.0
        sact = torch.ones(N, **f64) * 16.0
        izh = torch.zeros(N, dtype=torch.bool)
        V0 = torch.zeros(N, **f64)
        for p in self.pops:
            a, z = self.off[p]
            m = nest[node[p]["model"]]
            # дофамин узла: tata_dop = сеть − норма (0 в норме, −0.8 без дофамина); множители — как в
            # izhik_cond_exp.cpp 116–132, 654 и my_aeif_cond_exp.cpp 109–124 (module-130701)
            td = float(m.get("tata_dop", 0.0))
            n = z - a
            is_izh = m["type_id"] == "izhik_cond_exp"
            izh[a:z] = is_izh
            if is_izh:
                for k, dflt in (("k", 1.0), ("E_L", -65.0), ("V_b", -65.0), ("a", 0.02), ("b_1", 0.2),
                                ("b_2", 0.2), ("p_1", 1.0), ("p_2", 1.0), ("c", -65.0), ("d", 8.0),
                                ("V_peak", 30.0), ("u_const", 0.0), ("u_kc", 0.0), ("kc_1", 0.0), ("kc_2", 0.0),
                                ("c_max", 0.0), ("u_max", 100000.0), ("I_e", 0.0)):
                    P[k][a:z] = _get(m, k, dflt)
                P["k"][a:z] *= 1 - _get(m, "beta_k", 0.0) * td
                P["E_L"][a:z] *= 1 - _get(m, "beta_E_L", 0.0) * td
                P["V_bf"][a:z] = P["V_b"][a:z] * (1 - _get(m, "beta_V_b", 0.0) * td)   # в формуле u; ветка — по V_b
                P["d"][a:z] *= 1 - _get(m, "beta_d", 0.0) * td
            else:
                for k, dflt in (("g_L", 30.0), ("E_L", -70.6), ("Delta_T", 2.0), ("V_reset", -60.0),
                                ("V_a", -70.6), ("a_1", 4.0), ("a_2", 4.0), ("tau_w", 144.0), ("b", 80.5),
                                ("V_peak", 0.0), ("I_e", 0.0)):
                    P[k][a:z] = _get(m, k, dflt)
                P["E_L"][a:z] *= 1 - _get(m, "beta_E_L", 0.0) * td
                P["V_af"][a:z] = P["V_a"][a:z] * (1 - _get(m, "beta_V_a", 0.0) * td)   # в формуле w; ветка — по V_a
                P["rs1"][a:z] = _get(m, "V_reset_slope1", 0.0)
                P["rs2"][a:z] = _get(m, "V_reset_slope2", 0.0)
                P["rm1"][a:z] = _get(m, "V_reset_max_slope1", 0.0)
                P["rm2"][a:z] = _get(m, "V_reset_max_slope2", 0.0)
            # разброс (default_params.py 1055–1080) — одна сеть на все строки пачки
            cm, vth = _get(m, "C_m", 281.0), _get(m, "V_th", -50.4)
            P["C_m"][a:z] = (cm + 0.2 * cm * torch.randn(n, generator=g, **f64)) if rand.get("C_m") else cm
            P["V_th"][a:z] = (vth + V_th_sigma * torch.randn(n, generator=g, **f64)) if rand.get("V_th") else vth
            V0[a:z] = (vth - 20.0 + 20.0 * torch.rand(n, generator=g, **f64)) if rand.get("V_m") else _get(m, "E_L", -70.0)
            for r, name in enumerate(REC):
                if f"{name}_Tau_decay" in m:
                    tau[r, a:z] = float(m[f"{name}_Tau_decay"])
                    erev[r, a:z] = float(m[f"{name}_E_rev"])
                    has_rec[r, a:z] = True
                    dopf[r, a:z] = 1 + _get(m, f"beta_I_{name}", 0.0) * td
            nact[a:z] = _get(m, "NMDA_1_Vact", -20.0)
            sact[a:z] = _get(m, "NMDA_1_Sact", 16.0)
        if (P["C_m"] <= 0).any():
            raise ValueError("C_m ≤ 0 после разброса")
        self.P = {k: v.to(dev) for k, v in P.items()}
        self.izh = izh.to(dev)
        self.tau, self.erev = tau.to(dev)[:, None, :], erev.to(dev)[:, None, :]
        self.dopf = dopf.to(dev)[:, None, :]
        self.nact, self.sact = nact.to(dev), sact.to(dev)
        self.V = V0.to(dev)[None, :].repeat(B, 1)
        self.u = torch.zeros(B, N, dtype=torch.float64, device=dev)
        self.gs = torch.zeros(6, B, N, dtype=torch.float64, device=dev)

        # --- входы (parrot + пуассон-генераторы по расписанию; генераторы на один вход складываются) ---
        self.inputs = [p for p in node if node[p]["type"] == "input"]
        self.in_off, m0 = {}, N
        for p in sorted(self.inputs):
            n = int(node[p]["n"])
            self.in_off[p] = (m0, m0 + n)
            m0 += n
        self.M = m0  # все источники: сеть + входы
        self.gens = []   # (t0, t1, rate, индексы входов [абсолютные − N], маска строк [B] | None)
        for p in sorted(self.inputs):
            for s in node[p].get("spike_setup") or []:
                times = [float(x) for x in s["times"]]
                rates = [float(x) for x in s["rates"]]
                t_end = float(s.get("t_stop") or 1e12)
                idx = torch.tensor(s["idx"]) if s.get("idx") is not None else torch.arange(int(node[p]["n"]))
                for j, (tt, rr) in enumerate(zip(times, rates)):
                    self.add_gen(p, idx, tt, times[j + 1] if j + 1 < len(times) else t_end, rr)

        # --- связи ---
        src, tgt, w, dl, rc, stp_rows = [], [], [], [], [], []
        stp_par = []
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
                if node[t_pop].get("fan_in_distribution", "constant") != "constant":
                    raise NotImplementedError(f"{name}: не constant fan_in")
                fan = float(c["fan_in"])
                # наборы (каналы действий): приёмник — driver, источник — pool; пары наборов — zip, как в
                # structure.py 686–700 (при разном числе наборов лишние отбрасываются)
                kd, kp = c["rule"].split("-")
                t_sets = _sets(nt, self.n_sets.get(t_pop, 1), kd)
                s_sets = _sets(ns, self.n_sets.get(s_pop, 1), kp)
                s_parts, t_parts = [], []
                for t_set, s_set in zip(t_sets, s_sets):
                    s_rel, t_rel = _pick_sources(len(t_set), len(s_set), fan, g)
                    s_parts.append(s_set[s_rel] + sa)
                    t_parts.append(t_set[t_rel] + ta)
                s_idx, t_idx = torch.cat(s_parts), torch.cat(t_parts)
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
        stp_all = torch.cat(stp_rows)
        order = torch.argsort(src, stable=True)          # CSR по источнику
        src, tgt, w, dl, rc, is_stp = src[order], tgt[order], w[order], dl[order], rc[order], stp_all[order]
        stp_id = torch.full((src.numel(),), -1, dtype=torch.long)
        n_stp = self.n_stp = int(is_stp.sum())
        stp_id[is_stp] = torch.arange(n_stp)
        if n_stp:
            all_par = torch.zeros(order.numel(), 4, **f64)
            all_par[stp_all] = torch.cat(stp_par)
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
        z64 = lambda: torch.zeros(n_stp, B, dtype=torch.float64, device=dev)
        self.stp = {"U": to(all_par[:, 0]), "fac": to(all_par[:, 1]), "rec": to(all_par[:, 2]), "psc": to(all_par[:, 3]),
                    "x": z64() + 1.0, "y": z64(), "u": z64(), "last": z64()}
        # кольцо задержек [D, 6, B, N]: float32 по умолчанию (память при B > 1); проводимости — float64
        self.ring = torch.zeros(self.D, 6, B, N, dtype=ring_dtype, device=dev)
        self.k = 0
        self._rate_t = None
        self.dec = torch.exp(-(DT / nsub) / self.tau)
        # fp32 (флаг): состояние клеток во float32 — на игровых картах float64 в 32–64 раза медленнее;
        # синапсы Цодыкса и счёт частот остаются float64
        self.sdt = torch.float32 if fp32 else torch.float64
        if fp32:
            self.P = {k: v.float() for k, v in self.P.items()}
            self.V, self.u, self.gs = self.V.float(), self.u.float(), self.gs.float()
            self.tau, self.erev, self.dopf = self.tau.float(), self.erev.float(), self.dopf.float()
            self.nact, self.sact, self.dec = self.nact.float(), self.sact.float(), self.dec.float()
        self._sub = _substeps
        if compile_:
            global _substeps_compiled
            if _substeps_compiled is None:
                _substeps_compiled = torch.compile(_substeps, dynamic=False)
            self._sub = _substeps_compiled

    # ---------- входы ----------
    def add_gen(self, pop: str, idx, t0: float, t1: float, rate: float, rows=None) -> None:
        """Пуассон-генератор: входам idx популяции входа pop (номера внутри неё) — rate Гц на [t0, t1),
        в строках rows (маска [B] или None — во всех). Генераторы на один вход складываются (NEST)."""

        a, z = self.in_off[pop]
        idx = torch.as_tensor(idx, dtype=torch.long)
        if idx.numel() and int(idx.max()) >= z - a:
            raise ValueError(f"{pop}: номер входа вне популяции")
        rows_t = None if rows is None else torch.as_tensor(rows, dtype=torch.bool)
        self.gens.append((float(t0), float(t1), float(rate), idx + (a - self.N), rows_t))
        self._rate_t = None

    def _in_rates(self, t: float) -> torch.Tensor:
        """Частоты входов [B, M−N] в момент t; пересчёт только при смене расписания."""

        if self._rate_t is None:
            bounds = sorted({g[0] for g in self.gens} | {g[1] for g in self.gens})
            self._bounds = bounds
            self._rate_t = (-1.0, -1.0)
        lo, hi = self._rate_t
        if lo <= t < hi:
            return self._rate
        nxt = [b for b in self._bounds if b > t]
        prv = [b for b in self._bounds if b <= t]
        r = torch.zeros(self.B, self.M - self.N, dtype=torch.float64)
        for t0, t1, rate, idx, rows in self.gens:
            if t0 <= t < t1:
                if rows is None:
                    r[:, idx] += rate
                else:
                    r[rows.nonzero().squeeze(1)[:, None], idx[None, :]] += rate
        self._rate = r.to(self.dev)
        self._rate_t = (prv[-1] if prv else -1e12, nxt[0] if nxt else 1e12)
        return self._rate

    # ---------- доставка спайков ----------
    def _deliver(self, counts: torch.Tensor, t_spk: float) -> None:
        """counts [B, M] — спайки источников этого шага."""

        B = self.B
        nz = torch.nonzero(counts)
        if nz.numel() == 0:
            return
        b_act, s_act = nz[:, 0], nz[:, 1]
        n = self.deg[s_act]
        tot = int(n.sum())
        if tot == 0:
            return
        start = self.ptr[s_act]
        first = torch.cumsum(n, 0) - n
        j = torch.repeat_interleave(start - first, n) + torch.arange(tot, device=self.dev)
        bj = torch.repeat_interleave(b_act, n)
        c = torch.repeat_interleave(counts[b_act, s_act].to(torch.float64), n)
        val = self.w[j] * c
        sid = self.stp_id[j]
        m = sid >= 0
        if m.any():
            s, bs = sid[m], bj[m]
            S = self.stp
            fac, rec, psc, U = S["fac"][s], S["rec"][s], S["psc"][s], S["U"][s]
            h = t_spk - S["last"][s, bs]
            Puu = torch.where(fac == 0, torch.zeros_like(h), torch.exp(-h / fac.clamp(min=1e-30)))
            Pyy = torch.exp(-h / psc)
            Pzz = torch.exp(-h / rec)
            Pxy = ((Pzz - 1.0) * rec - (Pyy - 1.0) * psc) / (psc - rec)
            Pxz = 1.0 - Pzz
            x, y, u = S["x"][s, bs], S["y"][s, bs], S["u"][s, bs]
            z = 1.0 - x - y
            u = u * Puu
            x = x + Pxy * y + Pxz * z
            y = y * Pyy
            u = u + U * (1.0 - u)
            dy = u * x
            S["x"][s, bs], S["y"][s, bs], S["u"][s, bs] = x - dy, y + dy, u
            S["last"][s, bs] = t_spk
            val[m] = self.w[j[m]] * dy
        slot = (self.k + self.dl[j]) % self.D
        flat = ((slot * 6 + self.rc[j]) * B + bj) * self.N + self.tgt[j]
        self.ring.view(-1).index_add_(0, flat, val.to(self.ring.dtype))

    def step(self) -> torch.Tensor:
        """Шаг 0.1 мс; возвращает спайки сети [B, N] (bool)."""

        V, u, gs, spk = self._sub(self.P, self.V, self.u, self.gs, self.izh, self.erev, self.dopf, self.nact,
                                  self.sact, self.dec, self.nsub, DT / self.nsub)
        slot = self.k % self.D
        gs = gs + self.ring[slot].to(self.sdt)
        self.ring[slot].zero_()
        self.V, self.u, self.gs = V, u, gs
        # спайки этого шага (сеть + входы) уходят с задержкой
        t_now = (self.k + 1) * DT
        lam = self._in_rates(self.k * DT) * (DT * 1e-3)
        cin = torch.poisson(lam, generator=self.g) if self.dev == "cpu" else torch.poisson(lam)
        counts = torch.cat([spk.to(torch.float64), cin], 1)
        self._deliver(counts, t_now)
        self.k += 1
        return spk


def main(opts: dict) -> None:
    torch.set_num_threads(int(opts.get("threads", 4)))
    par = json.load(open(os.path.expanduser(opts["par"])))
    seed = int(opts.get("seed", 1))
    dev = opts.get("device", "cpu")
    torch.manual_seed(seed)
    t0 = time.time()
    B = int(opts.get("B", 1))
    net = Net(par, seed, dev, int(opts.get("nsub", 10)), B=B, compile_=bool(int(opts.get("compile", 0))),
              fp32=bool(int(opts.get("fp32", 0))))
    T = float(opts.get("t", net.t_stop))
    rec0 = float(opts.get("rec", 1000.0))
    print(f"сборка {time.time() - t0:.0f} с: клеток {net.N}, входов {net.M - net.N}, синапсов {int(net.deg.sum())}, "
          f"задержка до {net.D - 1} шагов, строк {B}", flush=True)
    steps = int(round(T / DT))
    bin_ms = 5.0
    per_bin = int(round(bin_ms / DT))
    n_bins = steps // per_bin
    pop_of = torch.zeros(net.N, dtype=torch.long)
    for i, p in enumerate(net.pops):
        a, z = net.off[p]
        pop_of[a:z] = i
    onehot = torch.nn.functional.one_hot(pop_of, len(net.pops)).to(torch.float64).to(dev)
    hist = torch.zeros(n_bins, B, len(net.pops), dtype=torch.float64, device=dev)
    cnt = torch.zeros(B, net.N, dtype=torch.float64, device=dev)
    t1 = time.time()
    for k in range(steps):
        s = net.step().to(torch.float64)
        b = k // per_bin
        if b < n_bins:
            hist[b] += s @ onehot
        if k * DT >= rec0:
            cnt += s
        if k and k % int(round(500.0 / DT)) == 0:
            print(f"  {k * DT:.0f} мс, {time.time() - t1:.0f} с", flush=True)
    dur = (T - rec0) / 1000.0
    print(f"счёт {time.time() - t1:.0f} с; частоты {rec0:.0f}–{T:.0f} мс (Гц на клетку, среднее строк):", flush=True)
    rates = {}
    for i, p in enumerate(net.pops):
        a, z = net.off[p]
        r = cnt[:, a:z] / dur
        rates[p] = float(r.mean())
        row = " ".join(f"{float(x):.2f}" for x in r.mean(1)) if B > 1 else ""
        print(f"  {p}: {float(r.mean()):7.2f}  (медиана {float(r.median()):6.2f}, молчат {float((r == 0).double().mean()) * 100:4.0f}%) {row}", flush=True)
    out = opts.get("out")
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        sizes = torch.tensor([net.off[p][1] - net.off[p][0] for p in net.pops], dtype=torch.float64, device=dev)
        hz = (hist.mean(1) / sizes / (bin_ms / 1000.0)).cpu()
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
