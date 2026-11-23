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
    10 мс; шаг 0.1 мс, Эйлер (Brian 1 по умолчанию).

Отличия от оригинала: случайные числа свои (связи — авторов из файлов;
активные клетки EC, их спайки, шум, кластеры — по тем же правилам).

    PYTHONPATH=. python tools/dg_chavlis.py net=/путь/ConnectivityMatrices_12dendrites [seed=1] [active=файл.npy] [stim_hz=40] [pre_ms=300] [stim_ms=500]
"""

from __future__ import annotations

import io
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

DT = 0.1  # мс
DT_T = torch.float64


def load_syn(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(пре, пост) из файла связей Brian 1 (save_connectivity — zip с .npy)."""

    with zipfile.ZipFile(path) as z:
        pre = np.load(io.BytesIO(z.read("presynaptic.npy")))
        post = np.load(io.BytesIO(z.read("postsynaptic.npy")))
    return pre.astype(np.int64).ravel(), post.astype(np.int64).ravel()


class Syn:
    """
    Группа синапсов одной кинетики: состояние у каждого синапса, прирост после
    задержки (шаги), сумма состояний по мишени (индекс в векторе мишеней).
    """

    def __init__(self, pre, target, delay_steps, tau_rise, tau_decay, alpha, set_mode=False, w=1.0):
        self.pre = torch.as_tensor(pre, dtype=torch.long)
        self.target = torch.as_tensor(target, dtype=torch.long)
        self.delay = torch.as_tensor(delay_steps, dtype=torch.long)
        n = len(self.pre)
        self.s = torch.zeros(n, dtype=DT_T)  # j / y / z
        self.r = torch.zeros(n, dtype=DT_T)  # x / h / r
        self.tr, self.td, self.alpha, self.set_mode, self.w = tau_rise, tau_decay, alpha, set_mode, w

    def step(self, hist: torch.Tensor, t: int) -> None:
        # Эйлер по старым значениям (как Brian 1: сначала состояние, потом события).
        ds = -self.s / self.td + self.alpha * self.r * (1.0 - self.s)
        dr = -self.r / self.tr
        self.s = self.s + DT * ds
        self.r = self.r + DT * dr
        slot = (t - self.delay) % hist.shape[0]
        arrived = hist[slot, self.pre]
        if self.set_mode:
            self.r = torch.where(arrived, torch.full_like(self.r, self.w), self.r)
        else:
            self.r = self.r + arrived.to(DT_T) * self.w

    def total(self, n: int) -> torch.Tensor:
        return torch.zeros(n, dtype=DT_T).index_add_(0, self.target, self.s)


def mg_gc(v):
    # Гранулярные: 1 / (1 + η·Mg·exp(−γ v)), η 0.2 /мМ, Mg 2 мМ, γ 0.04 /мВ (стр. 166–168).
    return 1.0 / (1.0 + 0.2 * 2.0 * torch.exp(-0.04 * v))


def mg_int(v):
    # BC, MC, HIPP: 1 / (1 + exp(−0.072 v) / 3.57) (стр. 355).
    return 1.0 / (1.0 + torch.exp(-0.072 * v) / 3.57)


def main(opts: dict) -> None:
    seed = int(opts.get("seed", 1))
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    torch.set_num_threads(int(opts.get("threads", 2)))
    net = Path(opts["net"])

    n_in, n_gc, n_bc, n_mc, n_hp = 400, 2000, 100, 80, 40
    cl = 20  # GC в кластере (стр. 276)
    pre_ms, stim_ms, post_ms = float(opts.get("pre_ms", 300)), float(opts.get("stim_ms", 500)), 10.0
    stim_hz = float(opts.get("stim_hz", 40))
    T = int(round((pre_ms + stim_ms + post_ms) / DT))

    # ---- GC: сома + 21 отсек (стр. 125–186) ----
    # Порядок отсеков: 0 сома; ветка b (0..2): дистальные b01, b02, b11, b12,
    # средние b0, b1, проксимальный b — индексы 1 + 7b + (0..6).
    um2 = 1e-8  # см²
    a_soma = math.pi * 12.0 * 18.0 * um2
    dl = 83.0
    diam = [0.80] * 4 + [0.90] * 2 + [1.00]
    a_dend = [math.pi * d * dl * um2 for d in diam]
    C = [1.0 * a_soma * 1e6] + [2.5 * a * 1e6 for a in a_dend] * 3  # пФ
    GL = [3e-5 * a_soma * 1e9] + [1e-5 * a * 1e9 for a in a_dend] * 3  # нС
    EL = [-87.0] + [-82.0] * 21
    n_comp = 22
    C_t = torch.tensor(C, dtype=DT_T)
    GL_t = torch.tensor(GL, dtype=DT_T)
    EL_t = torch.tensor(EL, dtype=DT_T)

    def ra(d):  # МОм отрезка 83 мкм диаметра d
        return 210.0 * 4.0 / (math.pi * (d * 1e-4) ** 2) * (dl * 1e-4) / 1e6

    g0, g1, g2 = 1000.0 / ra(0.80), 1000.0 / ra(0.90), 1000.0 / ra(1.00)  # нС
    edges = []
    for b in range(3):
        base = 1 + 7 * b
        d01, d02, d11, d12, m0, m1, p = (base + i for i in range(7))
        edges += [(d01, m0, g0), (d02, m0, g0), (d11, m1, g0), (d12, m1, g0), (m0, p, g1), (m1, p, g1), (p, 0, g2)]
    distal = [1 + 7 * b + i for b in range(3) for i in range(4)]  # 12, порядок dend001, 002, 011, 012, 101, …
    proximal = [1 + 7 * b + 6 for b in range(3)]
    e_a = torch.tensor([e[0] for e in edges])
    e_b = torch.tensor([e[1] for e in edges])
    e_g = torch.tensor([e[2] for e in edges], dtype=DT_T)

    v_gc = EL_t.repeat(n_gc, 1)  # [GC, отсек]
    i_ahp = torch.zeros(n_gc, dtype=DT_T)
    ref_gc = torch.zeros(n_gc, dtype=torch.long)

    # ---- AdEx (стр. 323–470): C пФ, gl нС, El, VT, ΔT мВ, τw мс, a нС, Vr, b пА, рефр. мс ----
    adex = {
        "bc": (179.3, 18.054, -52.0, -39.0, 2.0, 100.0, 0.1, -45.0, 20.5, 2.0, n_bc),
        "mc": (252.1, 4.53, -64.0, -42.0, 2.0, 180.0, 1.0, -49.0, 82.9, 2.0, n_mc),
        "hp": (58.4, 1.93, -59.0, -50.0, 2.0, 93.0, 0.82, -56.0, 15.0, 3.0, n_hp),
    }
    st = {k: {"v": torch.full((p[-1],), p[2], dtype=DT_T), "w": torch.zeros(p[-1], dtype=DT_T),
              "ref": torch.zeros(p[-1], dtype=torch.long)} for k, p in adex.items()}

    # ---- пресинаптические «популяции» в общей истории спайков ----
    # 0..399 EC, далее GC, BC, MC, HP, шумы.
    off = {}
    cur = 0
    for name, n in (("ec", n_in), ("gc", n_gc), ("bc", n_bc), ("mc", n_mc), ("hp", n_hp),
                    ("ngc", 500), ("nbc", 20 * n_bc), ("nmc", 30 * n_mc), ("nhp", 20 * n_hp)):
        off[name] = cur
        cur += n
    n_pre = cur
    rates = torch.zeros(n_pre, dtype=DT_T)
    rates[off["ngc"]: off["ngc"] + 500] = 0.1
    rates[off["nbc"]: off["nbc"] + 20 * n_bc] = 3.0
    rates[off["nmc"]: off["nmc"] + 30 * n_mc] = 3.8
    rates[off["nhp"]: off["nhp"] + 20 * n_hp] = 3.0
    hist = torch.zeros(128, n_pre, dtype=torch.bool)  # задержки ≤ 10 мс = 100 шагов

    # EC: активные 10% (40 клеток), пуассон stim_hz в окне стимула (стр. 117–122).
    if opts.get("active"):
        active = np.load(opts["active"]).astype(np.int64)
    else:
        active = np.sort(rng.choice(n_in, int(0.1 * n_in), replace=False))
    ec_active = torch.zeros(n_in, dtype=torch.bool)
    ec_active[torch.as_tensor(active)] = True

    def steps(ms):
        return int(round(ms / DT))

    # ---- синапсы ----
    # мишени: GC отсеки — плоский индекс cell*22 + comp; у AdEx — номер клетки.
    syn = {}
    # EC → GC, 12 дистальных (стр. 509–797): AMPA 0.8066 нС, NMDA ×1.08; τ AMPA 0.1/2.5, NMDA 0.33/50, α_nmda 2.
    pre_l, tgt_l = [], []
    for k in range(12):
        p, q = load_syn(net / f"syn_eg{k + 1}.txt")
        pre_l.append(p + off["ec"])
        tgt_l.append(q * n_comp + distal[k])
    p, q = np.concatenate(pre_l), np.concatenate(tgt_l)
    syn["eg_a"] = (Syn(p, q, np.full(len(p), steps(3.0)), 0.1, 2.5, 1.0), 0.8066, 0.0, None)
    syn["eg_n"] = (Syn(p, q, np.full(len(p), steps(3.0)), 0.33, 50.0, 2.0), 0.8066 * 1.08, 0.0, mg_gc)
    # MC → GC, 3 проксимальных (стр. 881–943): AMPA 0.1066, NMDA ×1.08, задержка 3 мс.
    pre_l, tgt_l = [], []
    for k in range(3):
        p, q = load_syn(net / f"syn_mg{k + 1}.txt")
        pre_l.append(p + off["mc"])
        tgt_l.append(q * n_comp + proximal[k])
    p, q = np.concatenate(pre_l), np.concatenate(tgt_l)
    syn["mg_a"] = (Syn(p, q, np.full(len(p), steps(3.0)), 0.1, 2.5, 1.0), 0.1066, 0.0, None)
    syn["mg_n"] = (Syn(p, q, np.full(len(p), steps(3.0)), 0.33, 50.0, 2.0), 0.1066 * 1.08, 0.0, mg_gc)
    # HIPP → GC, 12 дистальных (стр. 985–1130): ГАМК 0.12 нС, τ 0.9/6.8, E −86, задержка 1.6.
    pre_l, tgt_l = [], []
    for k in range(12):
        p, q = load_syn(net / f"syn_hg{k + 1}.txt")
        pre_l.append(p + off["hp"])
        tgt_l.append(q * n_comp + distal[k])
    p, q = np.concatenate(pre_l), np.concatenate(tgt_l)
    syn["hg"] = (Syn(p, q, np.full(len(p), steps(1.6)), 0.9, 6.8, 1.0), 0.12, -86.0, None)
    # BC → GC своего кластера, сома (стр. 970–983): ГАМК 14 нС, задержка 0.85.
    gcs = np.arange(n_gc)
    syn["bg"] = (Syn(gcs // cl + off["bc"], gcs * n_comp, np.full(n_gc, steps(0.85)), 0.9, 6.8, 1.0), 14.0, -86.0, None)
    # Шум GC (стр. 1134–): 500 источников × 2000 GC, доля 1/21 на отсек, задержка U(0,10) мс; 0.008 нС AMPA и NMDA, set.
    pre_l, tgt_l = [], []
    for comp in range(1, n_comp):
        mask = rng.random((500, n_gc)) < 1.0 / 21
        s_i, c_i = np.nonzero(mask)
        pre_l.append(s_i + off["ngc"])
        tgt_l.append(c_i * n_comp + comp)
    p, q = np.concatenate(pre_l), np.concatenate(tgt_l)
    d = np.round(rng.random(len(p)) * 10.0 / DT).astype(np.int64)
    syn["gn_a"] = (Syn(p, q, d, 0.1, 2.5, 1.0, set_mode=True), 0.008, 0.0, None)
    syn["gn_n"] = (Syn(p, q, d, 0.33, 50.0, 2.0, set_mode=True), 0.008, 0.0, mg_gc)
    n_gc_t = n_gc * n_comp

    # → BC: GC своего кластера (0.21 нС, NMDA ×1.5, задержка 0.8), все MC (0.35, ×1.1, 3 мс), шум 20 × 3 Гц (3.5 / 2.5 нС).
    # τ BC: AMPA 1.2/4.2, NMDA 10/130, α_nmda 0.5.
    syn_i = {"bc": [], "mc": [], "hp": []}
    syn_i["bc"] += [
        (Syn(gcs + off["gc"], gcs // cl, np.full(n_gc, steps(0.8)), 1.2, 4.2, 1.0), 0.21, None),
        (Syn(gcs + off["gc"], gcs // cl, np.full(n_gc, steps(0.8)), 10.0, 130.0, 0.5), 0.21 * 1.5, mg_int),
    ]
    mm, bb = np.meshgrid(np.arange(n_mc), np.arange(n_bc), indexing="ij")
    mm, bb = mm.ravel(), bb.ravel()
    syn_i["bc"] += [
        (Syn(mm + off["mc"], bb, np.full(len(mm), steps(3.0)), 1.2, 4.2, 1.0), 0.35, None),
        (Syn(mm + off["mc"], bb, np.full(len(mm), steps(3.0)), 10.0, 130.0, 0.5), 0.35 * 1.1, mg_int),
    ]
    nb = np.arange(20 * n_bc)
    d = np.round(rng.random(len(nb)) * 10.0 / DT).astype(np.int64)
    syn_i["bc"] += [
        (Syn(nb + off["nbc"], nb // 20, d, 1.2, 4.2, 1.0, set_mode=True), 3.5, None),
        (Syn(nb + off["nbc"], nb // 20, d, 10.0, 130.0, 0.5, set_mode=True), 2.5, mg_int),
    ]
    # → MC: GC (матрица авторов; 0.5 нС, NMDA ×1.05, 1.5 мс), шум 20 своих × 3.8 Гц (4.7 / 4.465 нС); τ 0.5/6.2, 4/100.
    p, q = load_syn(net / "syn_gm.txt")
    syn_i["mc"] += [
        (Syn(p + off["gc"], q, np.full(len(p), steps(1.5)), 0.5, 6.2, 1.0), 0.5, None),
        (Syn(p + off["gc"], q, np.full(len(p), steps(1.5)), 4.0, 100.0, 0.5), 0.5 * 1.05, mg_int),
    ]
    nm = np.arange(20 * n_mc)
    d = np.round(rng.random(len(nm)) * 10.0 / DT).astype(np.int64)
    syn_i["mc"] += [
        (Syn(nm + off["nmc"], nm // 20, d, 0.5, 6.2, 1.0, set_mode=True), 4.7, None),
        (Syn(nm + off["nmc"], nm // 20, d, 4.0, 100.0, 0.5, set_mode=True), 4.465, mg_int),
    ]
    # → HIPP: EC (матрица авторов; 0.24 нС, NMDA ×1.15, 3 мс), шум 20 × 3 Гц (0.2 / 0.2 нС);
    # τ AMPA 2/11, NMDA 4.8/110 (EC), у шума NMDA 5/100.
    p, q = load_syn(net / "syn_eh.txt")
    syn_i["hp"] += [
        (Syn(p + off["ec"], q, np.full(len(p), steps(3.0)), 2.0, 11.0, 1.0), 0.24, None),
        (Syn(p + off["ec"], q, np.full(len(p), steps(3.0)), 4.8, 110.0, 0.5), 0.24 * 1.15, mg_int),
    ]
    nh = np.arange(20 * n_hp)
    d = np.round(rng.random(len(nh)) * 10.0 / DT).astype(np.int64)
    syn_i["hp"] += [
        (Syn(nh + off["nhp"], nh // 20, d, 2.0, 11.0, 1.0, set_mode=True), 0.2, None),
        (Syn(nh + off["nhp"], nh // 20, d, 5.0, 100.0, 0.5, set_mode=True), 0.2, mg_int),
    ]
    print(f"синапсов: GC {sum(len(x[0].pre) for x in syn.values())}, BC {sum(len(x[0].pre) for x in syn_i['bc'])}, "
          f"MC {sum(len(x[0].pre) for x in syn_i['mc'])}, HIPP {sum(len(x[0].pre) for x in syn_i['hp'])}", flush=True)

    counts = {k: torch.zeros(n, dtype=torch.long) for k, n in (("gc", n_gc), ("bc", n_bc), ("mc", n_mc), ("hp", n_hp))}
    counts_rest = {k: torch.zeros_like(v) for k, v in counts.items()}
    t_pre, t_stim = steps(pre_ms), steps(pre_ms + stim_ms)
    gc_flat_soma = torch.arange(n_gc) * n_comp

    for t in range(T):
        # 1. Пре-спайки внешних источников этого шага.
        fired = torch.zeros(n_pre, dtype=torch.bool)
        r = rates.clone()
        if t_pre <= t < t_stim:
            r[off["ec"]: off["ec"] + n_in] = ec_active.to(DT_T) * stim_hz
        fired |= torch.rand(n_pre, generator=gen, dtype=DT_T) < r * DT / 1000.0

        # 2. Токи и мембраны (Эйлер; синапсы — состояние на начало шага).
        vg = v_gc.reshape(-1)
        g_e = torch.zeros(n_gc_t, dtype=DT_T)
        i_syn = torch.zeros(n_gc_t, dtype=DT_T)
        for s, g, E, mg in syn.values():
            tot = s.total(n_gc_t) * g
            i_syn += tot * (vg - E) * (mg(vg) if mg is not None else 1.0)
        i_syn = i_syn.view(n_gc, n_comp)
        i_ax = torch.zeros(n_gc, n_comp, dtype=DT_T)
        dv_ab = v_gc[:, e_b] - v_gc[:, e_a]
        i_ax.index_add_(1, e_a, dv_ab * e_g)
        i_ax.index_add_(1, e_b, -dv_ab * e_g)
        i_leak = GL_t * (EL_t - v_gc)
        i_soma_extra = torch.zeros(n_gc, n_comp, dtype=DT_T)
        i_soma_extra[:, 0] = -i_ahp
        dv = (i_leak - i_syn + i_ax + i_soma_extra) / C_t
        v_gc = v_gc + DT * dv
        i_ahp = i_ahp + DT * (2.0 * (v_gc[:, 0] - (-87.0)) - i_ahp) / 45.0
        # рефрактерность: сома в сбросе
        hold = ref_gc > 0
        v_gc[:, 0] = torch.where(hold, torch.full_like(v_gc[:, 0], -74.0), v_gc[:, 0])
        ref_gc = (ref_gc - 1).clamp(min=0)
        spk_gc = (v_gc[:, 0] > -56.0) & ~hold
        v_gc[:, 0] = torch.where(spk_gc, torch.full_like(v_gc[:, 0], -74.0), v_gc[:, 0])
        i_ahp = i_ahp + spk_gc.to(DT_T) * 45.0
        ref_gc = torch.where(spk_gc, torch.full_like(ref_gc, steps(20.0)), ref_gc)

        spk = {"gc": spk_gc}
        for k, (Cm, gl, El, VT, dT, tw, a, Vr, b, rf, n) in adex.items():
            x = st[k]
            v = x["v"]
            i_s = torch.zeros(n, dtype=DT_T)
            for s, g, mg in syn_i[k]:
                tot = s.total(n) * g
                i_s += tot * (v - 0.0) * (mg(v) if mg is not None else 1.0)
            dvv = (gl * (El - v) + gl * dT * torch.exp(((v - VT) / dT).clamp(max=20.0)) - x["w"] - i_s) / Cm
            dw = (a * (v - El) - x["w"]) / tw
            v = v + DT * dvv
            x["w"] = x["w"] + DT * dw
            hold = x["ref"] > 0
            v = torch.where(hold, torch.full_like(v, Vr), v)
            x["ref"] = (x["ref"] - 1).clamp(min=0)
            s_k = (v > VT) & ~hold
            v = torch.where(s_k, torch.full_like(v, Vr), v)
            x["w"] = x["w"] + s_k.to(DT_T) * b
            x["ref"] = torch.where(s_k, torch.full_like(x["ref"], steps(rf)), x["ref"])
            x["v"] = v
            spk[k] = s_k

        for k, s_k in spk.items():
            fired[off[k]: off[k] + len(s_k)] = s_k
            counts[k] += s_k.long()
            if t < t_pre:
                counts_rest[k] += s_k.long()
        hist[t % hist.shape[0]] = fired

        # 3. Синапсы: состояние и события (задержка от истории).
        for s, *_ in syn.values():
            s.step(hist, t)
        for k in syn_i:
            for s, *_ in syn_i[k]:
                s.step(hist, t)
        if t % 1000 == 0:
            print(f"  {t * DT:.0f} мс: GC спайков {int(counts['gc'].sum())}", flush=True)

    sec_rest, sec_stim = pre_ms / 1000.0, (stim_ms + post_ms) / 1000.0
    print(f"зерно {seed}, вход {stim_hz:.0f} Гц у {int(ec_active.sum())} клеток EC из {n_in}")
    for k, n in (("gc", n_gc), ("bc", n_bc), ("mc", n_mc), ("hp", n_hp)):
        c_all, c_rest = counts[k], counts_rest[k]
        c_stim = c_all - c_rest
        print(f"{k.upper()}: покой {float(c_rest.sum()) / n / sec_rest:.2f} Гц (активных {float((c_rest > 0).float().mean()):.1%}), "
              f"стимул {float(c_stim.sum()) / n / sec_stim:.2f} Гц (активных {float((c_stim > 0).float().mean()):.1%})")
    if opts.get("out"):
        np.save(opts["out"], counts["gc"].numpy())


if __name__ == "__main__":
    main(dict(a.split("=", 1) for a in sys.argv[1:]))
