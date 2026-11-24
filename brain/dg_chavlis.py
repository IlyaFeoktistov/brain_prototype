"""
Зубчатая извилина по Chavlis, Petrantonakis & Poirazi 2017 (Hippocampus 27:89,
https://doi.org/10.1002/hipo.22675; код — ModelDB 206372, DG_12dendrites.py,
Brian 1.4) — сеть авторов целиком, без подгонки (CHAT_PLAN шаг 3, цикл 36).
Сверено с их кодом на Brian 1 (tools/dg_chavlis.py, те же матрицы и вход EC):
активных гранулярных 4.65 ± 0.78% против 4.55%, 1.72 против 1.75 спайка на
активную.

Сеть (числа — DG_12dendrites.py, строки — в комментариях):
  - 2000 гранулярных (GC): сома + 21 пассивный дендритный отсек (3 ветки × 4
    дистальных, 2 средних, 1 проксимальный), AHP на соме, рефрактерность 20
    мс; 100 корзинчатых (BC), 80 мшистых (MC), 40 HIPP — AdEx;
  - синапс — своё состояние у каждого: подъём r (dr/dt = −r/τr), спад s (ds/dt
    = −s/τd + α r (1 − s)); спайк после задержки — r += w (шум — r = w); ток —
    g × Σs × (v − E) × Mg-блок у NMDA;
  - EC → GC (12 дистальных), EC → HIPP, GC → MC, MC → GC (3 проксимальных),
    HIPP → GC (12 дистальных) — матрицы авторов; GC ↔ BC внутри кластера (20
    GC — своя BC, все со всеми), MC → все BC;
  - шум — независимые пуассоновские источники у каждого типа: у GC 500 по 0.1
    Гц (доля 1/21 на отсек, задержка U(0, 10) мс), BC 20 своих по 3 Гц, MC 20
    по 3.8 Гц, HIPP 20 по 3 Гц.
EC → корзинчатые у авторов нет. Без входа EC гранулярные молчат, тормозные
живут на своём шуме (BC ~23 Гц, MC ~6, HIPP ~0.5).

Счёт — как у авторов: шаг 0.1 мс (на шаг мозга — подшаги), клетки — Эйлер,
синапсы — экспоненциальный Эйлер (Synapses(…, implicit=True) у Brian 1),
рефрактерность — только запрет спайка, без удержания потенциала (сброс-строка
и AdaptiveReset у Brian 1).

Стык с мозгом:
  - входные клетки EC авторов (400) — клетки EC мозга (net["ec_area"]),
    случайные; их спайки прошлого шага мозга приходят на синапсы авторов в
    первом подшаге;
  - выход — спайки гранулярных (area.E, пути мозга DG → CA3);
  - вход мозга в мВ (вещества и прочее, путей в зубчатую нет) — ток в сому:
    мВ × входная проводимость клетки в покое (пассивная клетка от того же
    тока сместилась бы на те же мВ) [своё: перевод единиц].

Шаг без синхронизаций с процессором (пишется в CUDA Graph).
"""

from __future__ import annotations

import io
import math
import zipfile
from pathlib import Path

import numpy as np
import torch

DATA = Path(__file__).resolve().parent.parent / "data" / "anatomy"

SUB_DT = 0.1  # мс, шаг авторов
HIST = 128  # подшагов истории спайков: задержки ≤ 10 мс = 100

# Каналы тока синапсов: AMPA и ГАМК без блока, NMDA — с Mg-блоком.
AMPA, NMDA, GABA = 0, 1, 2
E_GABA = -86.0  # мВ (стр. 970–1130)

# GC (стр. 125–186): сома 12 × 18 мкм, отсеки 83 мкм (дистальные 0.8, средние
# 0.9, проксимальный 1.0 мкм), Cm 1 / 2.5 мкФ/см², gL 3e-5 / 1e-5 См/см², EL
# −87 / −82 мВ, Ri 210 Ом·см; порог −56, сброс −74, AHP: τ 45 мс, +45 пА за спайк,
# 2 нС × (v − (−87)).
GC_COMP = 22
GC_VTH, GC_VRESET, GC_REF_MS = -56.0, -74.0, 20.0
AHP_TAU, AHP_JUMP, AHP_G, AHP_E = 45.0, 45.0, 2.0, -87.0

# AdEx (стр. 323–470): C пФ, gL нС, EL, VT, ΔT мВ, τw мс, a нС, Vr мВ, b пА, рефр. мс.
ADEX = {
    "BC": (179.3, 18.054, -52.0, -39.0, 2.0, 100.0, 0.1, -45.0, 20.5, 2.0),
    "MC": (252.1, 4.53, -64.0, -42.0, 2.0, 180.0, 1.0, -49.0, 82.9, 2.0),
    "HIPP": (58.4, 1.93, -59.0, -50.0, 2.0, 93.0, 0.82, -56.0, 15.0, 3.0),
}
NAMES = ("GC", "BC", "MC", "HIPP")
CLUSTER = 20  # GC на одну BC (стр. 276)


def load_syn(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(пре, пост) из файла связей Brian 1 (save_connectivity — zip с .npy)."""

    with zipfile.ZipFile(path) as z:
        pre = np.load(io.BytesIO(z.read("presynaptic.npy")))
        post = np.load(io.BytesIO(z.read("postsynaptic.npy")))
    return pre.astype(np.int64).ravel(), post.astype(np.int64).ravel()


def gc_geometry() -> dict:
    """Ёмкость (пФ), утечка (нС), EL (мВ) отсеков GC и осевые проводимости (нС)."""

    um2 = 1e-8  # см²
    a_soma = math.pi * 12.0 * 18.0 * um2
    dl = 83.0
    diam = [0.80] * 4 + [0.90] * 2 + [1.00]
    a_dend = [math.pi * d * dl * um2 for d in diam]
    C = [1.0 * a_soma * 1e6] + [2.5 * a * 1e6 for a in a_dend] * 3
    GL = [3e-5 * a_soma * 1e9] + [1e-5 * a * 1e9 for a in a_dend] * 3
    EL = [-87.0] + [-82.0] * 21

    def ra(d):  # МОм отрезка dl мкм диаметра d
        return 210.0 * 4.0 / (math.pi * (d * 1e-4) ** 2) * (dl * 1e-4) / 1e6

    g0, g1, g2 = 1000.0 / ra(0.80), 1000.0 / ra(0.90), 1000.0 / ra(1.00)
    edges = []
    for b in range(3):
        d01, d02, d11, d12, m0, m1, p = (1 + 7 * b + i for i in range(7))
        edges += [(d01, m0, g0), (d02, m0, g0), (d11, m1, g0), (d12, m1, g0), (m0, p, g1), (m1, p, g1), (p, 0, g2)]
    distal = [1 + 7 * b + i for b in range(3) for i in range(4)]  # порядок dend001, 002, 011, 012, 101, …
    proximal = [1 + 7 * b + 6 for b in range(3)]
    return {"C": C, "GL": GL, "EL": EL, "edges": edges, "distal": distal, "proximal": proximal}


def gc_input_conductance(geo: dict) -> float:
    """Входная проводимость сомы пассивной GC (нС): 1 / R_in из уравнений отсеков."""

    G = np.diag(np.array(geo["GL"], dtype=np.float64))
    for a, b, g in geo["edges"]:
        G[a, a] += g
        G[b, b] += g
        G[a, b] -= g
        G[b, a] -= g
    return 1.0 / float(np.linalg.inv(G)[0, 0])


class ChavlisDG:
    """
    Сеть зубчатой Chavlis 2017 на клетках зоны мозга: GC — area.E, BC, MC,
    HIPP — area.I подряд. step(вход мВ, спайки зоны прошлого шага, все спайки
    мозга прошлого шага) → (v сом, спайки шага).
    """

    def __init__(self, net: dict, area, dt: float, device: torch.device, generator: torch.Generator,
                 ec_cells: torch.Tensor, dtype: torch.dtype = torch.float32):
        self.area, self.device, self.dt, self.dtype = area, device, dt, dtype
        self.substeps = max(1, round(dt / SUB_DT))
        if abs(self.substeps * SUB_DT - dt) > 1e-9:
            raise ValueError(f"зубчатая Chavlis: шаг мозга {dt} мс не кратен шагу авторов {SUB_DT} мс")
        groups = net["groups"]
        self.names = list(NAMES)
        self.size = {g: int(groups[g]["cells"]) for g in NAMES}
        n_gc = self.size["GC"]
        n_int = self.size["BC"] + self.size["MC"] + self.size["HIPP"]
        if area.E.size != n_gc or area.I.size != n_int or area.I.start != area.E.stop:
            raise ValueError(f"зубчатая Chavlis: нужно {n_gc} GC и {n_int} тормозных/мшистых подряд, "
                             f"в зоне {area.E.size} и {area.I.size}")
        self.n = n = n_gc + n_int
        self.first = area.E.start
        self.n_gc, self.n_int = n_gc, n_int
        self.type_of = torch.cat([torch.full((self.size[g],), i, dtype=torch.long) for i, g in enumerate(NAMES)]).to(device)
        self.topo = False
        self.ec_cells = ec_cells.to(device=device, dtype=torch.long)
        n_ec = len(self.ec_cells)
        if n_ec != int(net["ec_cells"]):
            raise ValueError(f"зубчатая Chavlis: входных клеток EC {n_ec}, у авторов {net['ec_cells']}")
        mats = DATA / net["matrices"]

        def cpu_rand(*shape):
            return torch.rand(*shape, generator=generator, device=generator.device, dtype=torch.float64).cpu().numpy()

        def steps(ms):
            return int(round(ms / SUB_DT))

        # ---- GC ----
        geo = gc_geometry()
        f = dict(device=device, dtype=dtype)
        self.C = torch.tensor(geo["C"], **f)
        self.GL = torch.tensor(geo["GL"], **f)
        self.EL = torch.tensor(geo["EL"], **f)
        self.e_a = torch.tensor([e[0] for e in geo["edges"]], device=device)
        self.e_b = torch.tensor([e[1] for e in geo["edges"]], device=device)
        self.e_g = torch.tensor([e[2] for e in geo["edges"]], **f)
        self.gc_ref_steps = steps(GC_REF_MS) - 1

        # ---- AdEx: BC, MC, HIPP по клеткам ----
        def per_cell(i):
            return torch.cat([torch.full((self.size[g],), ADEX[g][i]) for g in NAMES[1:]]).to(**f)

        self.Cm, self.gl, self.El, self.VT, self.dT, self.tw, self.a, self.Vr, self.b = (per_cell(i) for i in range(9))
        self.int_ref_steps = torch.cat([torch.full((self.size[g],), steps(ADEX[g][9]) - 1, dtype=torch.long)
                                        for g in NAMES[1:]]).to(device)
        # Mg-блок у мишени: GC — 1/(1 + 0.2·2·e^(−0.04v)) (стр. 166–168), прочие —
        # 1/(1 + e^(−0.072v)/3.57) (стр. 355).
        self.mg_eta = torch.cat([torch.full((n_gc * GC_COMP,), 0.4), torch.full((n_int,), 1 / 3.57)]).to(**f)
        self.mg_gamma = torch.cat([torch.full((n_gc * GC_COMP,), 0.04), torch.full((n_int,), 0.072)]).to(**f)

        # Мишени тока: отсеки GC (cell·22 + отсек), затем клетки AdEx.
        self.n_tgt = n_gc * GC_COMP + n_int
        # Пресинаптические места в истории: клетки зоны, EC, шумы.
        off = {"own": 0, "ec": n}
        n_noise = {"GC": 500, "BC": 20 * self.size["BC"], "MC": 20 * self.size["MC"], "HIPP": 20 * self.size["HIPP"]}
        noise_hz = {"GC": 0.1, "BC": 3.0, "MC": 3.8, "HIPP": 3.0}
        cur = n + n_ec
        for g in NAMES:
            off["n" + g] = cur
            cur += n_noise[g]
        self.n_pre = cur
        self.n_ext = cur - n  # EC и шумы
        p = torch.zeros(cur - n - n_ec, dtype=torch.float64)
        for g in NAMES:
            lo = off["n" + g] - n - n_ec
            p[lo: lo + n_noise[g]] = noise_hz[g] * SUB_DT / 1000.0
        self.noise_p = p.to(**f)
        gc0 = 0
        bc0 = n_gc
        mc0 = bc0 + self.size["BC"]
        hp0 = mc0 + self.size["MC"]
        tgt_int = n_gc * GC_COMP - n_gc  # индекс мишени AdEx-клетки = номер в зоне + это

        parts = []

        def add(pre, tgt, delay, tr, td, alpha, g, chan, set_mode=False):
            pre = np.asarray(pre, dtype=np.int64)
            k = len(pre)
            parts.append(dict(pre=pre, tgt=np.asarray(tgt, dtype=np.int64) * 3 + chan,
                              delay=np.broadcast_to(np.asarray(delay, dtype=np.int64), (k,)).copy(),
                              tr=np.full(k, tr), td=np.full(k, td), alpha=np.full(k, alpha),
                              g=np.full(k, g), set_mode=np.full(k, set_mode)))

        def matrix(names):
            pre_l, post_l, idx = [], [], []
            for i, nm in enumerate(names):
                pp, qq = load_syn(mats / nm)
                pre_l.append(pp)
                post_l.append(qq)
                idx.append(np.full(len(pp), i))
            return np.concatenate(pre_l), np.concatenate(post_l), np.concatenate(idx)

        # EC → GC, 12 дистальных (стр. 509–797): AMPA 0.8066 нС (τ 0.1/2.5), NMDA ×1.08 (0.33/50, α 2), 3 мс.
        pp, qq, k = matrix([f"syn_eg{i + 1}.txt" for i in range(12)])
        tg = qq * GC_COMP + np.array(geo["distal"])[k]
        add(pp + off["ec"], tg, steps(3.0), 0.1, 2.5, 1.0, 0.8066, AMPA)
        add(pp + off["ec"], tg, steps(3.0), 0.33, 50.0, 2.0, 0.8066 * 1.08, NMDA)
        # MC → GC, 3 проксимальных (стр. 881–943): 0.1066 нС, NMDA ×1.08, 3 мс.
        pp, qq, k = matrix([f"syn_mg{i + 1}.txt" for i in range(3)])
        tg = qq * GC_COMP + np.array(geo["proximal"])[k]
        add(pp + mc0, tg, steps(3.0), 0.1, 2.5, 1.0, 0.1066, AMPA)
        add(pp + mc0, tg, steps(3.0), 0.33, 50.0, 2.0, 0.1066 * 1.08, NMDA)
        # HIPP → GC, 12 дистальных (стр. 985–1130): ГАМК 0.12 нС, τ 0.9/6.8, 1.6 мс.
        pp, qq, k = matrix([f"syn_hg{i + 1}.txt" for i in range(12)])
        add(pp + hp0, qq * GC_COMP + np.array(geo["distal"])[k], steps(1.6), 0.9, 6.8, 1.0, 0.12, GABA)
        # BC → GC своего кластера, сома (стр. 970–983): 14 нС, 0.85 мс.
        gcs = np.arange(n_gc)
        add(gcs // CLUSTER + bc0, gcs * GC_COMP, steps(0.85), 0.9, 6.8, 1.0, 14.0, GABA)
        # Шум GC (стр. 1134–): 500 источников, на каждый отсек 1–21 доля 1/21,
        # задержка U(0, 10) мс; 0.008 нС AMPA и NMDA, r = w.
        pre_l, tgt_l = [], []
        for comp in range(1, GC_COMP):
            s_i, c_i = np.nonzero(cpu_rand(500, n_gc) < 1.0 / 21)
            pre_l.append(s_i + off["nGC"])
            tgt_l.append(c_i * GC_COMP + comp)
        pp, tg = np.concatenate(pre_l), np.concatenate(tgt_l)
        d = np.round(cpu_rand(len(pp)) * 10.0 / SUB_DT).astype(np.int64)
        add(pp, tg, d, 0.1, 2.5, 1.0, 0.008, AMPA, True)
        add(pp, tg, d, 0.33, 50.0, 2.0, 0.008, NMDA, True)
        # → BC (τ AMPA 1.2/4.2, NMDA 10/130, α 0.5): GC кластера 0.21 нС, NMDA
        # ×1.5, 0.8 мс; все MC 0.35, ×1.1, 3 мс; шум 20 × 3 Гц 3.5 / 2.5 нС.
        bc_t = gcs // CLUSTER + bc0 + tgt_int
        add(gcs + gc0, bc_t, steps(0.8), 1.2, 4.2, 1.0, 0.21, AMPA)
        add(gcs + gc0, bc_t, steps(0.8), 10.0, 130.0, 0.5, 0.21 * 1.5, NMDA)
        mm, bb = np.meshgrid(np.arange(self.size["MC"]), np.arange(self.size["BC"]), indexing="ij")
        mm, bb = mm.ravel() + mc0, bb.ravel() + bc0 + tgt_int
        add(mm, bb, steps(3.0), 1.2, 4.2, 1.0, 0.35, AMPA)
        add(mm, bb, steps(3.0), 10.0, 130.0, 0.5, 0.35 * 1.1, NMDA)
        nb = np.arange(n_noise["BC"])
        d = np.round(cpu_rand(len(nb)) * 10.0 / SUB_DT).astype(np.int64)
        add(nb + off["nBC"], nb // 20 + bc0 + tgt_int, d, 1.2, 4.2, 1.0, 3.5, AMPA, True)
        add(nb + off["nBC"], nb // 20 + bc0 + tgt_int, d, 10.0, 130.0, 0.5, 2.5, NMDA, True)
        # → MC (τ 0.5/6.2, 4/100, α 0.5): GC по матрице 0.5 нС, ×1.05, 1.5 мс;
        # шум 20 × 3.8 Гц 4.7 / 4.465 нС.
        pp, qq = load_syn(mats / "syn_gm.txt")
        add(pp + gc0, qq + mc0 + tgt_int, steps(1.5), 0.5, 6.2, 1.0, 0.5, AMPA)
        add(pp + gc0, qq + mc0 + tgt_int, steps(1.5), 4.0, 100.0, 0.5, 0.5 * 1.05, NMDA)
        nm = np.arange(n_noise["MC"])
        d = np.round(cpu_rand(len(nm)) * 10.0 / SUB_DT).astype(np.int64)
        add(nm + off["nMC"], nm // 20 + mc0 + tgt_int, d, 0.5, 6.2, 1.0, 4.7, AMPA, True)
        add(nm + off["nMC"], nm // 20 + mc0 + tgt_int, d, 4.0, 100.0, 0.5, 4.465, NMDA, True)
        # → HIPP: EC по матрице 0.24 нС (τ 2/11), NMDA ×1.15 (4.8/110), 3 мс;
        # шум 20 × 3 Гц 0.2 / 0.2 нС (NMDA 5/100).
        pp, qq = load_syn(mats / "syn_eh.txt")
        add(pp + off["ec"], qq + hp0 + tgt_int, steps(3.0), 2.0, 11.0, 1.0, 0.24, AMPA)
        add(pp + off["ec"], qq + hp0 + tgt_int, steps(3.0), 4.8, 110.0, 0.5, 0.24 * 1.15, NMDA)
        nh = np.arange(n_noise["HIPP"])
        d = np.round(cpu_rand(len(nh)) * 10.0 / SUB_DT).astype(np.int64)
        add(nh + off["nHIPP"], nh // 20 + hp0 + tgt_int, d, 2.0, 11.0, 1.0, 0.2, AMPA, True)
        add(nh + off["nHIPP"], nh // 20 + hp0 + tgt_int, d, 5.0, 100.0, 0.5, 0.2, NMDA, True)

        def cat(key, **kw):
            return torch.as_tensor(np.concatenate([x[key] for x in parts]), **kw)

        self.pre = cat("pre", device=device)
        self.tgt = cat("tgt", device=device)
        self.delay = cat("delay", device=device)
        if int(self.delay.max()) >= HIST:
            raise ValueError("зубчатая Chavlis: задержка длиннее истории спайков")
        self.decay_r = torch.exp(-SUB_DT / cat("tr", dtype=torch.float64)).to(**f)
        self.inv_td = (1.0 / cat("td", dtype=torch.float64)).to(**f)
        self.alpha = cat("alpha", **f)
        self.g = cat("g", **f)
        self.set_mode = cat("set_mode", device=device)
        self.synapses = len(self.pre)
        self.dropped = torch.zeros(1, device=device)

        # Вход мозга (мВ) → ток в сому (пА): × входная проводимость клетки.
        g_in = torch.cat([torch.full((n_gc,), gc_input_conductance(geo)),
                          torch.cat([torch.full((self.size[g],), ADEX[g][1]) for g in NAMES[1:]])])
        self.gain = g_in.to(**f)
        self.reset_state()

    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        f = dict(device=self.device, dtype=self.dtype)
        self.v_gc = self.EL.repeat(self.n_gc, 1)
        self.i_ahp = torch.zeros(self.n_gc, **f)
        self.ref_gc = torch.zeros(self.n_gc, dtype=torch.long, device=self.device)
        self.v_int = self.El.clone()
        self.w_int = torch.zeros(self.n_int, **f)
        self.ref_int = torch.zeros(self.n_int, dtype=torch.long, device=self.device)
        self.s = torch.zeros(self.synapses, **f)
        self.r = torch.zeros(self.synapses, **f)
        self.hist = torch.zeros(HIST, self.n_pre, dtype=torch.bool, device=self.device)
        self.tick = torch.zeros(1, dtype=torch.long, device=self.device)
        self.v = torch.cat([self.v_gc[:, 0], self.v_int])

    STATE = ("v_gc", "i_ahp", "ref_gc", "v_int", "w_int", "ref_int", "s", "r", "hist", "tick", "v")

    def state_dict(self) -> dict:
        return {k: getattr(self, k).clone() for k in self.STATE}

    def load_state_dict(self, state: dict) -> None:
        for k in self.STATE:
            getattr(self, k).copy_(state[k])

    # ------------------------------------------------------------------

    def substep(self, ext: torch.Tensor, ext_pa: torch.Tensor) -> torch.Tensor:
        """
        Подшаг 0.1 мс. ext — спайки EC (первые n_ec) и шумов этого подшага
        (остальное — пуассон здесь же, если передан None-шум), ext_pa — ток в
        сому, пА. Возвращает спайки клеток зоны.
        """

        n_gc, dt = self.n_gc, SUB_DT
        # 1. Токи по состоянию синапсов на начало подшага.
        G = torch.zeros(self.n_tgt * 3, device=self.device, dtype=self.dtype).index_add_(0, self.tgt, self.s * self.g)
        G = G.view(self.n_tgt, 3)
        v_all = torch.cat([self.v_gc.reshape(-1), self.v_int])
        mg = 1.0 / (1.0 + self.mg_eta * torch.exp(-self.mg_gamma * v_all))
        i_syn = (G[:, AMPA] + G[:, NMDA] * mg) * v_all + G[:, GABA] * (v_all - E_GABA)
        i_gc = i_syn[: n_gc * GC_COMP].view(n_gc, GC_COMP)
        i_int = i_syn[n_gc * GC_COMP:]

        # 2. GC: отсеки (Эйлер), AHP, порог.
        v = self.v_gc.clone()
        dv_ab = v[:, self.e_b] - v[:, self.e_a]
        i_ax = torch.zeros_like(v)
        i_ax.index_add_(1, self.e_a, dv_ab * self.e_g)
        i_ax.index_add_(1, self.e_b, -dv_ab * self.e_g)
        i_tot = self.GL * (self.EL - v) - i_gc + i_ax
        i_tot[:, 0] += ext_pa[:n_gc] - self.i_ahp
        v = v + dt * i_tot / self.C
        ahp = self.i_ahp + dt * (AHP_G * (v[:, 0] - AHP_E) - self.i_ahp) / AHP_TAU
        # Рефрактерность Brian 1 при сбросе-строке: только запрет спайка.
        hold = self.ref_gc > 0
        ref = (self.ref_gc - 1).clamp(min=0)
        spk_gc = (v[:, 0] > GC_VTH) & ~hold
        v[:, 0] = torch.where(spk_gc, GC_VRESET, v[:, 0])
        # Состояние — на месте (copy_): шаг пишется в CUDA Graph, новый
        # тензор при повторе графа не виден.
        self.i_ahp.copy_(ahp + spk_gc.to(self.dtype) * AHP_JUMP)
        self.ref_gc.copy_(torch.where(spk_gc, self.gc_ref_steps, ref))
        self.v_gc.copy_(v)

        # 3. AdEx (Эйлер; AdaptiveReset — тоже без удержания).
        vi, w = self.v_int, self.w_int
        dvi = (self.gl * (self.El - vi) + self.gl * self.dT * torch.exp(((vi - self.VT) / self.dT).clamp(max=20.0))
               - w - i_int + ext_pa[n_gc:]) / self.Cm
        dw = (self.a * (vi - self.El) - w) / self.tw
        vi = vi + dt * dvi
        w = w + dt * dw
        hold = self.ref_int > 0
        ref = (self.ref_int - 1).clamp(min=0)
        spk_int = (vi > self.VT) & ~hold
        self.v_int.copy_(torch.where(spk_int, self.Vr, vi))
        self.w_int.copy_(w + spk_int.to(self.dtype) * self.b)
        self.ref_int.copy_(torch.where(spk_int, self.int_ref_steps, ref))

        # 4. История: спайки зоны, EC и шумов этого подшага.
        own = torch.cat([spk_gc, spk_int])
        slot = self.tick % HIST
        self.hist.index_copy_(0, slot, torch.cat([own, ext])[None])

        # 5. Синапсы: экспоненциальный Эйлер, затем приход спайков после задержки.
        a = self.inv_td + self.alpha * self.r
        s_inf = self.alpha * self.r / a
        self.s.copy_(s_inf + (self.s - s_inf) * torch.exp(-a * dt))
        r = self.r * self.decay_r
        arrived = self.hist.view(-1).index_select(0, ((self.tick - self.delay) % HIST) * self.n_pre + self.pre)
        self.r.copy_(torch.where(arrived & self.set_mode, self.g.new_ones(()), r + arrived.to(self.dtype)))
        self.tick.add_(1)
        return own

    def noise(self) -> torch.Tensor:
        """Спайки шумовых источников авторов за подшаг (пуассон)."""

        return torch.rand(len(self.noise_p), device=self.device, dtype=self.dtype) < self.noise_p

    def step(self, ext_mv: torch.Tensor, fired_prev: torch.Tensor,
             brain_spikes: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Шаг мозга: подшаги авторов; спайки EC прошлого шага мозга — в первом
        подшаге. Возвращает (v сом, спайки шага).
        """

        ec = brain_spikes.index_select(0, self.ec_cells) > 0
        quiet = torch.zeros_like(ec)
        ext_pa = ext_mv.to(self.dtype) * self.gain
        spikes = torch.zeros(self.n, dtype=torch.bool, device=self.device)

        for j in range(self.substeps):
            spikes |= self.substep(torch.cat([ec if j == 0 else quiet, self.noise()]), ext_pa)
        self.v.copy_(torch.cat([self.v_gc[:, 0], self.v_int]))
        return self.v.float(), spikes
