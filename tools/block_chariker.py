"""
Эталон блока «вход коры» — V1 L4Cα макаки по Chariker, Shapley & Young 2016
(https://doi.org/10.1523/JNEUROSCI.2603-16.2016, полный текст — принятая
версия в репозитории Duke: dukespace.lib.duke.edu, 12368.full.pdf), на
буквах вместо решёток. Не мозг — отдельная сеть на torch: проверяет, что
сам блок доносит букву от LGN до L4 (шаг 1 из DESIGN «Сборка из готовых
блоков»). Вопрос: V1 L4 различает буквы ~ как LGN (в мозге LGN 0.72, V1 L4
0.22).

Сеть (как у авторов, нормированные единицы: покой 0, порог 1, VE 14/3,
VI −2/3, время в с):
  LGN (Lin, Xing & Shapley 2012) — V' = −100 V + I + шум (пуассоновские
    толчки ±S_noise), порог 1, сброс 0; I = I0 (1 + gain · контраст), I0 =
    100; S_noise и gain подбираются здесь так, чтобы фон был ~20, а ответ на
    штрих в пиксель (контраст 1) ~100 Гц (Hawken et al. 1996) — так же
    задают их авторы («constants are chosen so that...»);
  L4Cα: 4000 клеток на гиперколонку (ГК, 0.5 × 0.5 мм), ¾ E, ¼ I (корзинчатые);
    LIF: dv/dt = −gL v − gE (v − VE) − gI (v − VI), gL E 50/с, I 66.5/с,
    рефрактерность 2 мс;
  связи по расстоянию (сила от расстояния не зависит): пресинаптические E —
    гауссиана SD 200/√2 мкм, I — 125/√2 мкм; пик вероятности E → E 13–18%
    (меньше у клеток с большим числом входов LGN), E → I, I → E, I → I 60%;
    итог ~200 E и ~100 I на клетку E, ~750 E и ~100 I на клетку I;
  силы: S_EE 0.028, S_IE 0.0095, S_EI 0.056, S_II U(0.036, 0.048),
    S_E,LGN 0.059, S_I,LGN 0.084, S6_EE 0.014, S6_IE 0.0047, у каждой
    клетки свой множитель ±10%;
  NMDA (без блока магнием): доля 0.2 на E, 0.33 на I — на синапсах E и L6
    (не LGN); спад AMPA ~3, GABA ~5, NMDA ~80 мс; отказы E → E до 20%, L6
    до 50%;
  L6 → L4: 300 клеток L6 на ГК, ⅚ контактов в круге 180 мкм, ~50 на клетку E
    (у клеток с 0–2 входами LGN вдвое больше, чем у 5–6); спайки L6 —
    пуассон 0.5–10 Гц;
  фон (нейромодуляция): пуассон 250/с, S_amb 0.01 на каждую клетку;
  вход LGN клетки E — шаблон (рис. 3): 0–6 клеток LGN в 2–3 ряда по 1–3
    вдоль ориентации своего домена, ряды разной полярности (ON/OFF) через
    1.5–2 шага решётки LGN; у 2 входов — пара ON–OFF рядом со случайной
    ориентацией (сложные клетки); число входов — по рис. 4 (⅓ клеток 0–2,
    ⅔ — 4–6); клетка I — 0–8 входов (гаусс 3 ± 2) без ориентации; шаблон —
    не дальше 400–500 мкм коры от клетки; 6 ориентаций (0°, 30°, …, 150°)
    вертушкой в каждой ГК.

Отличия от авторов (интерфейс к нашей задаче и упрощения, записать долгом,
не чинить до проверки):
  - решётка LGN — наша сетчатка: ON и OFF в каждом пикселе (у авторов OFF в
    центрах треугольников ON), поле — наша разность гауссиан (body.Eye);
    шаг решётки a px, ГК — hc_px px (у авторов ГК = 2 шага решётки M-клеток
    на 5°: 0.25° против 0.125°); hc_px = 2a — их отношение;
  - ток LGN — ступенька на показ буквы (у авторов — дрейфующая решётка);
  - L6 только фон: в эталоне L6 настроен на ориентацию и стреляет до 40 Гц,
    но его модель не дана — здесь L6 не несёт букву (консервативно);
  - глаз один (без глазодоминантных колонок), без гамма-синхронии L6,
    без шума выбора шаблона у границ доменов;
  - подъём синаптических проводимостей (у авторов «несколько мс», по Tao et
    al. 2004) — разность экспонент: AMPA 1 → 3, GABA 1 → 5, NMDA 2 → 80 мс
    [сверить с Tao 2004]; отказ — постоянная вероятность на спайк: E → E
    0.1, L6 0.25 (середина «до 20%» и «до 50%»);
  - край поля: связи за краем теряются (у авторов 9 ГК, мерили центр).

Проверка (как letters_dc): trials показов каждой буквы; на метки и на
проверку поровну; классификатор по корреляции с центром буквы и голос
клеток — для LGN, L4 E, L4 I. Показ: фон rest мс, буква show мс; счёт
спайков за показ.

    PYTHONPATH=. python tools/block_chariker.py [letters=абвгдежзик] [trials=10] [a=1] [hc_px=2] [side=16]
        [show=350] [rest=300] [dt=0.1] [batch=50] [seed=1] [device=cuda] [s_ie=0.0095] [stp=U,τмс] [lgn_wiring=ours] [lgn_psp=мВ] [cap=e:Гц,i:Гц] [l4_local=brain] [l4_size=brain] [lgn_replay=файл] [lgn_gain=k] [lgn_evoked=k] [lgn_cv=0.27] [save=файл]
"""

from __future__ import annotations

import math
import sys
import time

import torch

VE, VI = 14.0 / 3.0, -2.0 / 3.0
RHO_E, RHO_I, RHO_6 = 3000 / 500.0 ** 2, 1000 / 500.0 ** 2, 300 / 500.0 ** 2  # клеток на мкм²
SD_E, SD_I = 200 / math.sqrt(2), 125 / math.sqrt(2)  # мкм
# L6: ⅚ контактов в круге 180 мкм → SD гауссианы 180 / √(2 ln 6) мкм.
SD_6 = 180 / math.sqrt(2 * math.log(6))
# Число входов LGN у клетки E — рис. 4 на глаз: ⅓ клеток 0–2 (сложные —
# в основном 2), ⅔ — 4–6, в среднем ~4.
NLGN_P = torch.tensor([0.04, 0.04, 0.25, 0.0, 0.22, 0.25, 0.20])
# Ряды шаблона по числу входов (клеток в ряду, соседние ряды — разной полярности).
ROWS = {3: [(1, 1, 1), (2, 1)], 4: [(2, 2), (1, 2, 1)], 5: [(2, 1, 2), (2, 3), (1, 3, 1)], 6: [(3, 3), (2, 2, 2)]}
# Синапсы: (подъём, спад), с.
TAU = {"ampa": (0.001, 0.003), "gaba": (0.001, 0.005), "nmda": (0.002, 0.080)}


def psp_mix_peak(S: float, g_leak: float, parts: tuple) -> float:
    """
    Пик ПСП (норм. единицы, по модулю) от одного спайка силы S у клетки в покое:
    parts — (доля силы, рецептор, потенциал реверсии), как в cortex_step.
    """

    dt = 1e-5
    syn = [[f * S / (TAU[r][1] - TAU[r][0])] * 2 + [TAU[r], rev] for f, r, rev in parts]
    v = peak = 0.0
    for _ in range(int(0.08 / dt)):
        g_v = 0.0
        for st in syn:
            (tr, td), rev = st[2], st[3]
            st[0] *= math.exp(-dt / td)
            st[1] *= math.exp(-dt / tr)
            g_v += (st[0] - st[1]) * (v - rev)
        v += dt * (-g_leak * v - g_v)
        peak = max(peak, abs(v))
    return peak


def psp_peak(S: float, g_leak: float) -> float:
    """Пик ВПСП (норм. единицы, порог 1) от одного спайка AMPA силы S у клетки в покое."""

    dt, (tr, td) = 1e-5, TAU["ampa"]
    v = peak = 0.0
    dec = ris = S / (td - tr)
    for _ in range(int(0.05 / dt)):
        dec *= math.exp(-dt / td)
        ris *= math.exp(-dt / tr)
        v += dt * (-g_leak * v - (dec - ris) * (v - VE))
        peak = max(peak, v)
    return peak


def lattice(n_side: int, size: float, dev) -> torch.Tensor:
    """Клетки ровной решёткой n_side × n_side на квадрате size мкм: [n, 2] (y, x)."""

    c = (torch.arange(n_side, device=dev).float() + 0.5) * size / n_side
    return torch.stack(torch.meshgrid(c, c, indexing="ij"), -1).view(-1, 2)


def sample_pre(post_xy: torch.Tensor, mean: torch.Tensor, sd: float, n_side: int, size: float,
               g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Пресинаптические клетки решётки n_side² по гауссиане SD sd вокруг каждой
    постсинаптической: число — Пуассон(mean) без выбросов (± 2√mean; у
    авторов «outliers discarded»), место — гауссово смещение, клетка —
    ближайшая на решётке; за краем — теряется. Возвращает (pre, post).
    """

    dev = post_xy.device
    lam = mean.float()
    k = torch.poisson(lam.cpu(), generator=g).to(dev)
    k = torch.minimum(torch.maximum(k, lam - 2 * lam.sqrt()), lam + 2 * lam.sqrt()).round().long().clamp(min=0)
    post = torch.repeat_interleave(torch.arange(len(post_xy), device=dev), k)
    off = torch.randn(len(post), 2, generator=g).to(dev) * sd
    ij = ((post_xy[post] + off) * n_side / size - 0.5).round().long()
    ok = ((ij >= 0) & (ij < n_side)).all(1)
    return ij[ok, 0] * n_side + ij[ok, 1], post[ok]


def table(pre: torch.Tensor, post: torch.Tensor, n_pre: int, n_post: int) -> torch.Tensor:
    """Исходящие синапсы: [n_pre + 1, D], пустые — n_post (фиктивная клетка); строка n_pre — пустая."""

    order = torch.argsort(pre, stable=True)
    pre, post = pre[order], post[order]
    deg = torch.bincount(pre, minlength=n_pre)
    d = int(deg.max()) if len(pre) else 1
    start = torch.cumsum(deg, 0) - deg
    col = torch.arange(len(pre), device=pre.device) - start[pre]
    out = torch.full((n_pre + 1, max(d, 1)), n_post, dtype=torch.long, device=pre.device)
    out[pre, col] = post
    return out


class Pop:
    """Клетки LIF одного типа в B испытаниях."""

    def __init__(self, n: int, B: int, g_leak: float, dev):
        self.n, self.g_leak = n, g_leak
        z = lambda: torch.zeros(B, n, device=dev)
        self.v, self.ref = torch.rand(B, n, device=dev) * 0.5, z()
        # Разность экспонент на каждый рецептор: (спад, подъём).
        self.syn = {r: (z(), z()) for r in TAU}
        # Число пришедших спайков за шаг по источникам [B, n + 1] (+ фиктивная).
        self.inbox = {s: torch.zeros(B * (n + 1), device=dev) for s in ("lgn", "e", "i", "l6", "amb")}
        self.spk = torch.zeros(B, n, dtype=torch.bool, device=dev)
        self.count = z()


class Block:
    def __init__(self, opts: dict, B: int, dev: str):
        self.dev, self.B = dev, B
        seed = int(opts.get("seed", 1))
        g = torch.Generator().manual_seed(seed)
        self.dt = float(opts.get("dt", 0.1)) / 1000.0
        a, hc_px, side = float(opts.get("a", 1)), float(opts.get("hc_px", 2)), int(opts.get("side", 16))
        self.side = side
        um_px = 500.0 / hc_px          # мкм коры на пиксель
        size = side * um_px            # сторона куска коры, мкм
        self.size = size
        n_e, n_i, n_6 = (max(1, round(size * math.sqrt(r))) for r in (RHO_E, RHO_I, RHO_6))
        xy_e, xy_i = lattice(n_e, size, dev), lattice(n_i, size, dev)
        NE, NI, N6 = n_e ** 2, n_i ** 2, n_6 ** 2
        # LGN: ON и OFF в каждой точке решётки шага a px (наша сетчатка при a = 1).
        n_l = int(side // a)
        self.lgn_px = (torch.arange(n_l, device=dev).float() + 0.5) * a - 0.5  # центры, px
        NL = 2 * n_l ** 2
        self.n_l = n_l
        print(f"кусок коры {size / 1000:.1f} × {size / 1000:.1f} мм ({(size / 500) ** 2:.0f} ГК): E {NE}, I {NI}, "
              f"L6 {N6}, LGN {NL} (решётка {n_l} × {n_l}, шаг {a:g} px; ГК {hc_px:g} px)", flush=True)

        # Ориентация клетки E — вертушка в своей ГК, 6 доменов по 30°.
        hc = (xy_e / 500.0).floor() * 500.0 + 250.0
        ang = torch.atan2(xy_e[:, 0] - hc[:, 0], xy_e[:, 1] - hc[:, 1])
        theta = ((ang + math.pi) / (2 * math.pi) * 6).long().clamp(max=5).float() * math.pi / 6

        # Шаблоны LGN клеток E.
        n_lgn = torch.multinomial(NLGN_P, NE, replacement=True, generator=g).to(dev)
        reach = 450.0 / um_px  # px: шаблон не дальше 400–500 мкм коры
        pre_l, post_l = [], []
        rnd = lambda *s: torch.rand(*s, generator=g).to(dev)

        def to_lgn(y: torch.Tensor, x: torch.Tensor, pol: torch.Tensor) -> torch.Tensor:
            i = ((y + 0.5) / a - 0.5).round().long()
            j = ((x + 0.5) / a - 0.5).round().long()
            ok = (i >= 0) & (i < n_l) & (j >= 0) & (j < n_l)
            return torch.where(ok, pol * n_l ** 2 + i.clamp(0, n_l - 1) * n_l + j.clamp(0, n_l - 1), -1)

        centre = xy_e / um_px - 0.5
        r_off, a_off = reach * rnd(NE).sqrt(), 2 * math.pi * rnd(NE)
        centre = centre + torch.stack([r_off * torch.sin(a_off), r_off * torch.cos(a_off)], 1)
        u = torch.stack([torch.cos(theta), torch.sin(theta)], 1)    # вдоль ряда (0° — вертикаль)
        p = torch.stack([-torch.sin(theta), torch.cos(theta)], 1)   # поперёк
        first = (rnd(NE) < 0.5).long()
        for n in range(1, 7):
            cells = (n_lgn == n).nonzero()[:, 0]
            if len(cells) == 0:
                continue
            c0 = centre[cells]
            if n == 1:
                pre_l.append(to_lgn(c0[:, 0], c0[:, 1], first[cells]))
                post_l.append(cells)
                continue
            if n == 2:
                # Пара ON–OFF как можно ближе, ориентация случайная.
                phi = 2 * math.pi * rnd(len(cells))
                dy, dx = 0.5 * a * torch.sin(phi), 0.5 * a * torch.cos(phi)
                for s, pol in ((-1, first[cells]), (1, 1 - first[cells])):
                    pre_l.append(to_lgn(c0[:, 0] + s * dy, c0[:, 1] + s * dx, pol))
                    post_l.append(cells)
                continue
            variants = ROWS[n]
            pick = torch.randint(len(variants), (len(cells),), generator=g).to(dev)
            sep = a * (1.5 + 0.5 * rnd(len(cells)))  # между рядами 1.5–2 шага (3/16–1/4° при 0.125°)
            for vi, rows in enumerate(variants):
                sel = pick == vi
                cs = cells[sel]
                for r, m in enumerate(rows):
                    across = (r - (len(rows) - 1) / 2) * sep[sel]
                    pol = (first[cs] + r) % 2
                    for k in range(m):
                        along = (k - (m - 1) / 2) * a
                        y = c0[sel, 0] + across * p[cs, 0] + along * u[cs, 0]
                        x = c0[sel, 1] + across * p[cs, 1] + along * u[cs, 1]
                        pre_l.append(to_lgn(y, x, pol))
                        post_l.append(cs)
        pre_l, post_l = torch.cat(pre_l), torch.cat(post_l)
        ok = pre_l >= 0
        lgn_e = (pre_l[ok], post_l[ok])
        lgn_scale = 1.0
        if opts.get("lgn_wiring") == "ours":
            # Замена на нашу деталь: поле простой клетки как engine._wire_oriented —
            # 4 ориентации (ближайшая к домену), в своём месте полоса OFF 2 × 5
            # по центру и по бокам ON 2 × 5 (только «+» подполя, как у таламуса
            # в мозге), 20 входов у каждой клетки. Сумма сил на клетку — как у
            # авторов (их среднее число входов × S_E,LGN): меняется устройство
            # поля, не сила.
            a4 = (theta / (math.pi / 4)).round().remainder(4) * math.pi / 4
            dr, dc = -torch.sin(a4), torch.cos(a4)
            pos = xy_e / um_px - 0.5
            t = torch.arange(-2, 3, device=dev).float()
            pre_o = []
            for side_, pol in ((-0.5, 1), (0.5, 1), (-1.5, 0), (1.5, 0)):
                y = pos[:, 0:1] + t[None] * dr[:, None] + side_ * dc[:, None]
                x = pos[:, 1:2] + t[None] * dc[:, None] - side_ * dr[:, None]
                pre_o.append(to_lgn(y, x, torch.full_like(y, pol, dtype=torch.long)))
            pre_o = torch.cat(pre_o, 1)
            post_o = torch.arange(NE, device=dev)[:, None].expand_as(pre_o)
            lgn_scale = len(lgn_e[0]) / NE / 20.0
            ok = pre_o >= 0
            lgn_e = (pre_o[ok], post_o[ok])
        # LGN → I: 0–8 входов (гаусс 3 ± 2), без ориентации, в пределах reach.
        k_i = (3 + 2 * torch.randn(NI, generator=g)).round().clamp(0, 8).long().to(dev)
        if "lgn_psp" in opts:
            # Сила по мозгу (ниже): 56 входов по данным строятся все — сжатые в ~3
            # они давали бы толчки в десятки ВПСП, которых нет ни у авторов, ни в мозге.
            k_i = torch.full((NI,), 56, device=dev)
        post = torch.repeat_interleave(torch.arange(NI, device=dev), k_i)
        r_off, a_off = reach * rnd(len(post)).sqrt(), 2 * math.pi * rnd(len(post))
        c = xy_i[post] / um_px - 0.5
        pre = to_lgn(c[:, 0] + r_off * torch.sin(a_off), c[:, 1] + r_off * torch.cos(a_off),
                     (rnd(len(post)) < 0.5).long())
        lgn_i = (pre[pre >= 0], post[pre >= 0])

        # Кора: пик E → E 13–18% (больше у клеток с меньшим числом входов LGN),
        # прочие 60%; ожидаемое число = пик · плотность · 2π SD².
        area = lambda sd: 2 * math.pi * sd ** 2
        peak_ee = 0.18 - 0.05 * n_lgn.float() / 6
        boost_6 = torch.where(n_lgn <= 2, 2.0, torch.where(n_lgn >= 5, 1.0, 1.5))
        base_6 = 50.0 / float(boost_6.mean())
        ee = sample_pre(xy_e, peak_ee * RHO_E * area(SD_E), SD_E, n_e, size, g)
        ie = sample_pre(xy_e, torch.full((NE,), 0.6 * RHO_I * area(SD_I), device=dev), SD_I, n_i, size, g)
        ei = sample_pre(xy_i, torch.full((NI,), 0.6 * RHO_E * area(SD_E), device=dev), SD_E, n_e, size, g)
        ii = sample_pre(xy_i, torch.full((NI,), 0.6 * RHO_I * area(SD_I), device=dev), SD_I, n_i, size, g)
        self.l4_brain = opts.get("l4_local") == "brain"
        if self.l4_brain:
            # Замена на нашу деталь: связи L4 → L4 как в колонке мозга
            # (engine._wire_column для V1, column_kinds E и PV): входов на клетку
            # по данным (Potjans Table 5 ÷ контактов на связь; E → E — человек,
            # Campagnola 2022: p 0.007; тормозные — доля PV и отношения
            # Campagnola) и ПСП пары (config: layer_ee_psp_mv, layer_ei_psp_mv,
            # inhibitory_psp_mv). Разброс в пространстве — блока (SD авторов).
            # Числа — вывод _wire_column для L4 (проверено скриптом по config).
            k_brain = {"ee": 45.1, "ie": 116.6, "ei": 762.9, "ii": 290.4}
            full = lambda n, k: torch.full((n,), k, device=dev)
            ee = sample_pre(xy_e, full(NE, k_brain["ee"]), SD_E, n_e, size, g)
            ie = sample_pre(xy_e, full(NE, k_brain["ie"]), SD_I, n_i, size, g)
            ei = sample_pre(xy_i, full(NI, k_brain["ei"]), SD_E, n_e, size, g)
            ii = sample_pre(xy_i, full(NI, k_brain["ii"]), SD_I, n_i, size, g)
        # Замена 5 (l4_size=brain, поверх l4_local=brain): размер зоны мозга.
        # V1 L4 мозга — 5680 E и 1420 PV против колонки 1 мм² 21915 / 5479
        # (Potjans; зона 20000 клеток): engine._wire_column строит k =
        # round(k_данных · 0.259) входов, вес × √(k_данных / k), недостающее
        # среднее — ток по живой частоте популяции-источника, сглаженной
        # mean_field_tau_ms = 50 мс (engine._missing, _build_mean_field).
        # Средний вход тот же, разброс — меньше.
        self.l4_size = opts.get("l4_size") == "brain"
        self.mf_k = {"E": {"e": 0.0, "i": 0.0}, "I": {"e": 0.0, "i": 0.0}}
        if self.l4_size:
            assert self.l4_brain, "l4_size=brain — поверх l4_local=brain"
            share = 5680 / 21915
            self.mf_root = root = {}
            for key, (kind, src) in (("ee", ("E", "e")), ("ie", ("E", "i")), ("ei", ("I", "e")), ("ii", ("I", "i"))):
                k = max(1, round(k_brain[key] * share))
                root[(kind, src)] = math.sqrt(k_brain[key] / k)
                # Недостающих входов (в весах блока до × √): k_данных − k · √.
                self.mf_k[kind][src] = k_brain[key] - k * root[(kind, src)]
                print(f"размер зоны мозга: {kind} ← {src} строится {k} из {k_brain[key]:.1f}, вес × {root[(kind, src)]:.2f}, "
                      f"среднее поле {self.mf_k[kind][src]:.1f} входов", flush=True)
                k_brain[key] = k
            ee = sample_pre(xy_e, full(NE, k_brain["ee"]), SD_E, n_e, size, g)
            ie = sample_pre(xy_e, full(NE, k_brain["ie"]), SD_I, n_i, size, g)
            ei = sample_pre(xy_i, full(NI, k_brain["ei"]), SD_E, n_e, size, g)
            ii = sample_pre(xy_i, full(NI, k_brain["ii"]), SD_I, n_i, size, g)
        l6e = sample_pre(xy_e, base_6 * boost_6, SD_6, n_6, size, g)
        # L6 → I — «как внутри 4Cα»: во столько же раз больше, во сколько E → I больше E → E.
        l6i = sample_pre(xy_i, torch.full((NI,), 50.0 * 750 / 200, device=dev), SD_6, n_6, size, g)
        # Без связи клетки с самой собой.
        ee = (ee[0][ee[0] != ee[1]], ee[1][ee[0] != ee[1]])
        ii = (ii[0][ii[0] != ii[1]], ii[1][ii[0] != ii[1]])

        self.E, self.I = Pop(NE, B, 50.0, dev), Pop(NI, B, 66.5, dev)
        self.NL, self.N6 = NL, N6
        self.paths = [  # (источник, получатель, ящик, таблица, доля отказов)
            ("lgn", self.E, "lgn", table(*lgn_e, NL, NE), 0.0),
            ("lgn", self.I, "lgn", table(*lgn_i, NL, NI), 0.0),
            # Отказы E → E 0.1 — у авторов; ПСП мозга — средние с отказами.
            ("e", self.E, "e", table(*ee, NE, NE), 0.0 if self.l4_brain else 0.1),
            ("e", self.I, "e", table(*ei, NE, NI), 0.0),
            ("i", self.E, "i", table(*ie, NI, NE), 0.0),
            ("i", self.I, "i", table(*ii, NI, NI), 0.0),
            ("l6", self.E, "l6", table(*l6e, N6, NE), 0.25),
            ("l6", self.I, "l6", table(*l6i, N6, NI), 0.25),
        ]
        cnt = lambda pr, n: torch.bincount(pr[1], minlength=n).float()
        print(f"входов на клетку E: LGN {cnt(lgn_e, NE).mean():.1f}, E {cnt(ee, NE).mean():.0f}, "
              f"I {cnt(ie, NE).mean():.0f}, L6 {cnt(l6e, NE).mean():.0f}; на клетку I: LGN {cnt(lgn_i, NI).mean():.1f}, "
              f"E {cnt(ei, NI).mean():.0f}, I {cnt(ii, NI).mean():.0f}, L6 {cnt(l6i, NI).mean():.0f}", flush=True)

        # Силы с множителем клетки ±10%.
        jit = lambda n: 0.9 + 0.2 * rnd(n)
        s_lgn = {"E": 0.059 * lgn_scale, "I": 0.084}
        self.lgn_psp = "lgn_psp" in opts
        if self.lgn_psp:
            # Замена на нашу деталь: сила таламуса как в мозге (cfg.thalamic_psp_mv,
            # _thalamic_path) — ВПСП отдохнувшего синапса lgn_psp мВ (Bruno &
            # Sakmann 2006; на I ×2, Cruikshank et al. 2007), входов по данным 85
            # на E и 56 на I; все они — на построенных входах (их сила × k_данных /
            # k_построено). мВ → норм. единицы: порог − покой нашего нейрона
            # (cfg.v_threshold − cfg.v_rest = 20 мВ). Депрессия — абсолютная: на
            # фоне сила падает до x_фон (без нормировки к авторам).
            mv = float(opts["lgn_psp"])
            k_built = {"E": len(lgn_e[0]) / NE, "I": len(lgn_i[0]) / NI}
            for kind, k_data, ratio, gl in (("E", 85.0, 1.0, 50.0), ("I", 56.0, 2.0, 66.5)):
                s1 = mv * ratio / 20.0 / (psp_peak(1e-3, gl) / 1e-3)
                s_lgn[kind] = s1 * k_data / k_built[kind]
            print(f"сила LGN по мозгу: ВПСП {mv} мВ, на построенный вход E {s_lgn['E']:.3f} "
                  f"(авторы 0.059), I {s_lgn['I']:.3f} (авторы 0.084)", flush=True)
        self.S = {
            "E": {"lgn": s_lgn["E"] * jit(NE), "e": 0.028 * jit(NE), "i": 0.056 * jit(NE), "l6": 0.014 * jit(NE),
                  "amb": torch.full((NE,), 0.01, device=dev)},
            "I": {"lgn": s_lgn["I"] * jit(NI), "e": float(opts.get("s_ie", 0.0095)) * jit(NI), "i": 0.036 + 0.012 * rnd(NI), "l6": 0.0047 * jit(NI),
                  "amb": torch.full((NI,), 0.01, device=dev)},
        }
        self.nmda = {"E": 0.2, "I": 0.33}
        if self.l4_brain:
            # ПСП мозга (мВ) → сила блока: пик ПСП по форме синапсов блока
            # (AMPA + доля NMDA у E-входов, ГАМК с реверсией VI), мВ / 20.
            def strength(mv, kind, src):
                gl = 50.0 if kind == "E" else 66.5
                f = self.nmda[kind]
                parts = ((1 - f, "ampa", VE), (f, "nmda", VE)) if src == "e" else ((1.0, "gaba", VI),)
                return abs(mv) / 20.0 / (psp_mix_peak(1e-3, gl, parts) / 1e-3)
            psp_brain = {("E", "e"): 0.18, ("E", "i"): -0.21, ("I", "e"): 0.45, ("I", "i"): -0.88}
            for (kind, src), mv in psp_brain.items():
                old = float(self.S[kind][src].mean())
                new = strength(mv, kind, src)
                self.S[kind][src] = self.S[kind][src] / old * new
                print(f"L4 → L4 по мозгу: {kind} ← {src} ПСП {mv} мВ, сила {new:.4f} "
                      f"(авторы {old:.4f} — ПСП {old / strength(1.0, kind, src):.2f} мВ по модулю)", flush=True)
        # Сила синапса недостающего соседа — без × √ (среднее поле, ниже).
        self.S_mf = {kind: {src: self.S[kind][src].clone() for src in ("e", "i")} for kind in self.S}
        if self.l4_size:
            for (kind, src), r in self.mf_root.items():
                self.S[kind][src] = self.S[kind][src] * r
        # Частота популяций L4 [B], сглаженная 50 мс (engine mean_field_tau_ms).
        self.mf_rate = {"e": torch.full((B,), 1.4, device=dev), "i": torch.full((B,), 6.4, device=dev)}
        self.mf_decay = math.exp(-self.dt / 0.050)
        self.n_lgn = n_lgn
        # L6: пуассон 0.5–10 Гц, только фон.
        self.p6 = (0.5 + 9.5 * rnd(N6)) * self.dt
        self.p_amb = 250.0 * self.dt
        # LGN.
        self.lgn_v = torch.rand(B, NL, device=dev)
        self.lgn_I = torch.full((B, NL), 100.0, device=dev)
        self.lgn_spk = torch.zeros(B, NL, dtype=torch.bool, device=dev)
        self.lgn_count = torch.zeros(B, NL, device=dev)
        # Замена на нашу деталь (шаг 3 сборки): депрессия LGN → L4 по
        # Tsodyks–Markram, stp=U,τ_мс (наша — 0.45,500: cfg.nucleus_stp, Lee &
        # Sherman 2008). Ресурс x у клетки LGN (у всех её синапсов одна история
        # спайков); вес спайка — x / x_фон: отдохнувший синапс сильнее в 1/x_фон,
        # средний вход на фоне 20 Гц — как у авторов (проверяется сжатие
        # частоты, а не ослабление входа).
        self.stp = tuple(float(v) for v in opts["stp"].split(",")) if "stp" in opts else None
        self.lgn_x = torch.ones(B, NL, device=dev)
        self.lgn_w = torch.ones(B, NL, device=dev)
        if self.stp:
            U, tau = self.stp
            self.x_bg = 1.0 if self.lgn_psp else 1.0 / (1.0 + U * 20.0 * tau / 1000.0)
        self.noise_rate, self.s_noise, self.gain = 1000.0, 0.1, 1.0
        # Замена 6 (lgn_replay=файл от tools/lgn_record.py): вместо LGN блока —
        # спайки реле LGN мозга в окне 16 × 16 px (ON/OFF, решётка a = 1, как
        # у блока). Шаг мозга 0.5 мс: спайк — на случайный из его 5 шагов
        # блока. Проводка, сила и депрессия входа — прежние (замены 1–3).
        self.replay = torch.load(opts["lgn_replay"]) if "lgn_replay" in opts else None
        if self.replay is not None:
            assert a == 1 and side == 16, "запись LGN мозга — окно 16 × 16 px, a = 1"
            self.rep_t = torch.zeros((), dtype=torch.long, device=dev)
            # Диагностика (не исправление): lgn_gain=k — частота каждой клетки ×k
            # с тем же рисунком: добавочные пуассоновские спайки с частотой
            # (k − 1) · r̂(t), r̂ — своя частота клетки в этом же показе,
            # сглаженная окном 25 мс (сведений о букве из других показов нет).
            self.rep_gain = float(opts.get("lgn_gain", 1.0))
            # Диагностика: lgn_evoked=k — только ответ ×k, фон тот же: на показе
            # частота клетки r' = r_фон + k (r − r_фон). r — своя частота клетки
            # за весь показ (за 25 мс при 11 Гц — доля спайка, шум), r_фон — её
            # средняя за все паузы записи (букв там нет). Выше фона — добавочные
            # пуассоновские спайки, ниже — убираются её же спайки; пауза не
            # трогается, свои спайки клетки остаются.
            self.rep_evoked = float(opts.get("lgn_evoked", 1.0))
            # Диагностика: lgn_cv=c — те же частоты, но регулярные спайки (гамма-
            # процесс, CV c; у LGN блока 0.27). Частота клетки — её счёт в этом
            # показе на отрезках пауза / 0–50 мс показа / 50 мс–конец (ход
            # ответа: переход и устоявшаяся часть); мельче нельзя — при 11 Гц за
            # 25 мс меньше спайка. В своём времени Λ(t) интервалы — гамма со
            # средним 1, фаза первого спайка случайна.
            self.rep_cv = float(opts["lgn_cv"]) if "lgn_cv" in opts else None
            d = self.replay
            n_rest = round(d["rest"] / d["dt"])
            bg = torch.zeros(NL)
            for sp in d["spikes"]:
                bg += torch.bincount(sp[sp[:, 0] < n_rest, 1], minlength=NL).float()
            self.rep_bg = bg / (len(d["spikes"]) * d["rest"] / 1000.0)
            self.rep_raster = torch.zeros(1, B, NL, dtype=torch.bool, device=dev)
        self.drive = {"lgn": torch.zeros(B, NE, device=dev), "all": torch.zeros(B, NE, device=dev)}
        # Событий за шаг на источник (статический размер для nonzero_static):
        # с запасом на частоту до cap Гц у всех клеток; сверх — счёт потерь.
        cap = {"lgn": 300.0, "e": 100.0, "i": 300.0, "l6": 40.0}
        # cap=e:400,i:600 — больший запас (память ∝ запас): при синхронных
        # залпах за один шаг спайкует больше клеток, чем средняя частота cap.
        for kv in opts.get("cap", "").split(",") if opts.get("cap") else []:
            k, v = kv.split(":")
            cap[k] = float(v)
        n_src = {"lgn": NL, "e": NE, "i": NI, "l6": N6}
        self.K = {s: max(256, math.ceil(B * n_src[s] * cap[s] * self.dt)) for s in cap}
        # Потери по источнику (оба пути источника считают своё — число событий × 2).
        self.lost = {s: torch.zeros((), device=dev) for s in cap}
        if dev == "cuda":
            # Шаг одним куском (inductor): [замер] 1.6 мс против 10 на 64 тыс. клеток × 20 испытаний.
            self.step = torch.compile(self.step)

    # --- LGN ----------------------------------------------------------------
    def load_replay(self, trials: list[int], gen: torch.Generator) -> None:
        """Растр LGN мозга на пачку показов [шагов блока, B, NL] (пусто rest, показ show)."""

        d = self.replay
        sub = round(d["dt"] / 1000.0 / self.dt)
        steps = round((d["rest"] + d["show"]) / 1000.0 / self.dt)
        raster = torch.zeros(steps, self.B, self.NL, dtype=torch.bool)
        for b, j in enumerate(trials):
            sp = d["spikes"][j]
            t = sp[:, 0] * sub + torch.randint(sub, (len(sp),), generator=gen)
            raster[t, b, sp[:, 1]] = True
        if self.rep_cv is not None:
            raster = self.regularize(raster, gen)
        if self.rep_evoked != 1.0:
            k = self.rep_evoked
            t0 = round(d["rest"] / 1000.0 / self.dt)
            show_s = d["show"] / 1000.0
            counts = raster[t0:].sum(0).float()                      # [B, NL]
            r = counts / show_s
            dev = r - self.rep_bg[None]
            target = (self.rep_bg[None] + k * dev).clamp(min=0.0)
            add = ((target - r).clamp(min=0.0) * self.dt)            # вероятность на шаг
            keep = torch.where(r > target, target / r.clamp(min=1e-9), torch.ones_like(r))
            part = raster[t0:]
            part &= torch.rand(part.shape, generator=gen) < keep[None]
            part |= torch.rand(part.shape, generator=gen) < add[None]
            raster[t0:] = part
        if self.rep_gain > 1.0:
            w = round(0.025 / self.dt)
            x = raster.float().permute(1, 2, 0).reshape(-1, 1, steps)
            rate = torch.nn.functional.avg_pool1d(x, w, stride=1, padding=w // 2, count_include_pad=False)[..., :steps]
            p = ((self.rep_gain - 1.0) * rate).clamp(max=1.0)
            extra = torch.rand(p.shape, generator=gen) < p
            raster |= extra.reshape(self.B, self.NL, steps).permute(2, 0, 1)
        self.rep_raster = raster.to(self.dev)
        self.rep_t.zero_()

    def regularize(self, raster: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        """Регулярные спайки с той же частотой клетки по отрезкам (см. lgn_cv)."""

        d = self.replay
        steps = raster.shape[0]
        t_rest = round(d["rest"] / 1000.0 / self.dt)
        edges = [0, t_rest, t_rest + round(0.05 / self.dt), steps]
        x = raster.permute(1, 2, 0).reshape(-1, steps)                      # [клеток, шагов]
        n = torch.stack([x[:, a:b].sum(1).float() for a, b in zip(edges[:-1], edges[1:])], 1)  # счёт на отрезке
        lam = torch.cat([torch.zeros(len(x), 1), n.cumsum(1)], 1)          # Λ на границах
        kappa = 1.0 / self.rep_cv ** 2
        m = int(lam[:, -1].max()) + 8
        gam = torch.distributions.Gamma(torch.full((len(x), m), kappa), torch.full((len(x), m), kappa))
        torch.manual_seed(int(torch.randint(1 << 30, (1,), generator=gen)))
        isi = gam.sample()
        tt = isi.cumsum(1) - isi[:, :1] * torch.rand(len(x), 1, generator=gen)  # случайная фаза
        out = torch.zeros_like(x)
        for j in range(3):
            a, b = edges[j], edges[j + 1]
            lo, hi = lam[:, j:j + 1], lam[:, j + 1:j + 2]
            inside = (tt >= lo) & (tt < hi) & (hi > lo)
            frac = ((tt - lo) / (hi - lo).clamp(min=1e-9)).clamp(0, 1 - 1e-6)
            step = (a + frac * (b - a)).long()
            r, c = inside.nonzero(as_tuple=True)
            out[r, step[r, c]] = True
        return out.reshape(self.B, self.NL, steps).permute(2, 0, 1).contiguous()

    def lgn_step(self) -> None:
        if self.replay is not None:
            spk = self.rep_raster.index_select(0, self.rep_t.clamp(max=self.rep_raster.shape[0] - 1))[0]
            self.rep_t += 1
            self.lgn_spk = spk
            self.lgn_count += spk
            if self.stp:
                U, tau = self.stp
                self.lgn_w = self.lgn_x / self.x_bg
                self.lgn_x = self.lgn_x - U * self.lgn_x * spk
                self.lgn_x = 1.0 - (1.0 - self.lgn_x) * math.exp(-self.dt * 1000.0 / tau)
            return
        c, dt = 100.0, self.dt
        v_inf = self.lgn_I / c
        v = v_inf + (self.lgn_v - v_inf) * math.exp(-c * dt)
        kick = torch.rand_like(v) < self.noise_rate * dt
        sign = torch.where(torch.rand_like(v) < 0.5, -1.0, 1.0)
        v = v + kick * sign * self.s_noise
        spk = v >= 1.0
        self.lgn_v = torch.where(spk, 0.0, v)
        self.lgn_spk = spk
        self.lgn_count += spk
        if self.stp:
            U, tau = self.stp
            # Вес этого спайка — ресурс до него; затем ресурс тратится и восстанавливается.
            self.lgn_w = self.lgn_x / self.x_bg
            self.lgn_x = self.lgn_x - U * self.lgn_x * spk
            self.lgn_x = 1.0 - (1.0 - self.lgn_x) * math.exp(-dt * 1000.0 / tau)

    def calibrate_lgn(self) -> None:
        """S_noise — фон ~20 Гц при I0; gain — ~100 Гц на контраст 1 (Hawken et al. 1996)."""

        def rate(I_rel: float, ms: float = 2000.0) -> float:
            self.lgn_I.fill_(100.0 * I_rel)
            self.lgn_v.uniform_(0, 1)
            self.lgn_count.zero_()
            for _ in range(int(ms / 1000 / self.dt)):
                self.lgn_step()
            return float(self.lgn_count.mean()) / (ms / 1000)

        lo, hi = 0.0, 1.0
        for _ in range(14):
            self.s_noise = (lo + hi) / 2
            lo, hi = (self.s_noise, hi) if rate(1.0, 500) < 20.0 else (lo, self.s_noise)
        lo, hi = 0.0, 5.0
        for _ in range(14):
            self.gain = (lo + hi) / 2
            lo, hi = (self.gain, hi) if rate(1.0 + self.gain, 500) < 100.0 else (lo, self.gain)
        print(f"LGN (Lin 2012): толчки шума {self.noise_rate:.0f}/с по ±{self.s_noise:.3f}, gain {self.gain:.2f}; "
              f"фон {rate(1.0):.1f} Гц, контраст 1 — {rate(1.0 + self.gain):.1f} Гц", flush=True)

    def set_contrast(self, contrast: torch.Tensor | None) -> None:
        """contrast [B, side, side] (наша сетчатка, −1…1) или None — пусто."""

        if contrast is None:
            self.lgn_I.fill_(100.0)
            return
        # Значение в точках решётки LGN (ближайший пиксель).
        idx = self.lgn_px.round().long().clamp(0, self.side - 1)
        c = contrast[:, idx][:, :, idx].reshape(self.B, -1)
        on, off = c.clamp(-1, 1), (-c).clamp(-1, 1)
        self.lgn_I = 100.0 * (1.0 + self.gain * torch.cat([on, off], 1)).clamp(min=0.0)

    # --- кора ---------------------------------------------------------------
    def deliver(self, src: str, spk: torch.Tensor, post: Pop, box: str, tab: torch.Tensor, fail: float,
                w_pre: torch.Tensor | None = None) -> None:
        B, n_pre = spk.shape
        K = self.K[src]
        flat = torch.nonzero_static(spk.reshape(-1), size=K, fill_value=-1)[:, 0]
        self.lost[src] += (spk.sum() - K).clamp(min=0)
        valid = flat >= 0
        b = torch.where(valid, flat // n_pre, 0)
        pre = torch.where(valid, flat % n_pre, n_pre)
        rows = tab[pre]
        real = rows != post.n
        if fail > 0:
            real &= torch.rand(rows.shape, device=rows.device) >= fail
        # Пустые места — с весом 0 по разным адресам: в одну фиктивную ячейку
        # атомарные сложения шли очередью ([замер] 2 мс на путь из ~2.5).
        spread = torch.arange(rows.numel(), device=rows.device).view(rows.shape) % post.n
        dest = b[:, None] * (post.n + 1) + torch.where(real, rows, spread)
        val = real.float()
        if w_pre is not None:
            val = val * w_pre.reshape(-1)[torch.where(valid, flat, 0)][:, None]
        post.inbox[box].scatter_add_(0, dest.reshape(-1), val.reshape(-1))

    def cortex_step(self, pop: Pop, kind: str) -> None:
        dt, B, n = self.dt, self.B, pop.n
        S = self.S[kind]
        got = {s: pop.inbox[s].view(B, n + 1)[:, :n] for s in pop.inbox}
        got["amb"] = (torch.rand(B, n, device=self.dev) < self.p_amb).float()
        f = self.nmda[kind]
        if self.l4_size:
            # Недостающие соседи: спайков за шаг = входов × частота × dt (у всех
            # клеток популяции одно, своё в каждом испытании).
            for src in ("e", "i"):
                got[src] = got[src] + (self.mf_k[kind][src] * self.mf_rate[src] * dt)[:, None] * self.S_mf[kind][src] / S[src]
        w_ampa = S["lgn"] * got["lgn"] + (1 - f) * (S["e"] * got["e"] + S["l6"] * got["l6"]) + S["amb"] * got["amb"]
        w_nmda = f * (S["e"] * got["e"] + S["l6"] * got["l6"])
        w_gaba = S["i"] * got["i"]
        g = {}
        for r, w in (("ampa", w_ampa), ("nmda", w_nmda), ("gaba", w_gaba)):
            tr, td = TAU[r]
            dec, ris = pop.syn[r]
            dec.mul_(math.exp(-dt / td)).add_(w / (td - tr))
            ris.mul_(math.exp(-dt / tr)).add_(w / (td - tr))
            g[r] = dec - ris
        gE, gI = g["ampa"] + g["nmda"], g["gaba"]
        if kind == "E":
            # Доля LGN в возбуждающей проводимости (оценка: по приходу, без формы синапса).
            self.drive["lgn"] += S["lgn"] * got["lgn"]
            self.drive["all"] += S["lgn"] * got["lgn"] + S["e"] * got["e"] + S["l6"] * got["l6"] + S["amb"] * got["amb"]
        gT = pop.g_leak + gE + gI
        v_inf = (gE * VE + gI * VI) / gT
        free = pop.ref <= 0
        v = torch.where(free, v_inf + (pop.v - v_inf) * torch.exp(-gT * dt), pop.v)
        spk = (v >= 1.0) & free
        pop.v = torch.where(spk, 0.0, v)
        pop.ref = torch.where(spk, 0.002, pop.ref - dt)
        pop.spk = spk
        pop.count += spk
        for box in pop.inbox.values():
            box.zero_()

    def step(self) -> None:
        if self.l4_size:
            for src, pop in (("e", self.E), ("i", self.I)):
                inst = pop.spk.float().mean(1) / self.dt
                self.mf_rate[src] = self.mf_rate[src] * self.mf_decay + inst * (1.0 - self.mf_decay)
        self.lgn_step()
        l6 = torch.rand(self.B, self.N6, device=self.dev) < self.p6
        spikes = {"lgn": self.lgn_spk, "e": self.E.spk, "i": self.I.spk, "l6": l6}
        for src, post, box, tab, fail in self.paths:
            w = self.lgn_w if (src == "lgn" and self.stp) else None
            self.deliver(src, spikes[src], post, box, tab, fail, w)
        self.cortex_step(self.E, "E")
        self.cortex_step(self.I, "I")

    def run(self, ms: float) -> dict[str, torch.Tensor]:
        for pop in (self.E, self.I):
            pop.count.zero_()
        self.lgn_count.zero_()
        for d in self.drive.values():
            d.zero_()
        for _ in range(int(round(ms / 1000 / self.dt))):
            self.step()
        return {"LGN": self.lgn_count.clone(), "L4 E": self.E.count.clone(), "L4 I": self.I.count.clone()}


def contrast_maps(chars: list[str], side: int, gen: torch.Generator) -> torch.Tensor:
    """Буква (наш экран, шрифт 12 px) по центру поля со сдвигом ±1 px → контраст нашей сетчатки."""

    from brain.body import Eye
    from brain.config import BrainConfig
    from brain.world import Screen

    eye = Eye(BrainConfig(), 0, 1, "cpu")
    screen = Screen(side, side)
    out = []
    for ch in chars:
        x = int(round((side - screen.width(ch)) / 2)) + int(torch.randint(-1, 2, (1,), generator=gen))
        out.append(eye._contrast(screen.picture(ch, x)) / eye.scale)
    return torch.stack(out)


def main(opts: dict) -> None:
    dev = opts.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    letters = opts.get("letters", "абвгдежзик")
    trials = int(opts.get("trials", 10))
    show, rest = float(opts.get("show", 350)), float(opts.get("rest", 300))
    seed = int(opts.get("seed", 1))
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 1000)
    labels = torch.arange(len(letters)).repeat(trials)
    B = min(int(opts.get("batch", 50)), len(labels))
    t0 = time.time()
    net = Block(opts, B, dev)
    if net.replay is None:
        net.calibrate_lgn()
    else:
        d = net.replay
        assert d["letters"] == letters and len(d["labels"]) == len(labels) and (d["rest"], d["show"]) == (rest, show), \
            "запись LGN — те же буквы, показы, rest и show"
        print(f"LGN — спайки LGN мозга из {opts['lgn_replay']}", flush=True)
    print(f"сборка {time.time() - t0:.0f} с", flush=True)

    side = net.side
    res = {k: [] for k in ("LGN", "L4 E", "L4 I")}
    bg = {k: [] for k in res}
    early, late = {k: [] for k in res}, {k: [] for k in res}
    frac = []
    for s in range(0, len(labels), B):
        y = labels[s:s + B]
        if len(y) < B:  # последняя пачка добивается повтором (лишнее отбрасывается)
            y = torch.cat([y, labels[:B - len(y)]])
        t1 = time.time()
        if net.replay is not None:
            idx = torch.arange(s, s + B) % len(labels)
            net.load_replay(idx.tolist(), gen)
        net.set_contrast(None)
        net.run(rest - 100)
        r0 = net.run(100.0)
        net.set_contrast(contrast_maps([letters[k] for k in y.tolist()], side, gen).to(dev))
        # Показ — переход (первые 50 мс) и устойчивая часть: частоты и
        # различение отдельно (депрессия таламуса оставляет только переход).
        ra = net.run(50.0)
        da = [float(net.drive[d].sum()) for d in ("lgn", "all")]
        rb = net.run(show - 50.0)
        db = [float(net.drive[d].sum()) for d in ("lgn", "all")]
        r = {k: ra[k] + rb[k] for k in ra}
        frac.append((da[0] + db[0]) / (da[1] + db[1]))
        n = min(B, len(labels) - s)
        for k in res:
            res[k].append(r[k][:n].cpu())
            early[k].append(ra[k][:n].cpu())
            late[k].append(rb[k][:n].cpu())
            bg[k].append(r0[k][:n].cpu() / 0.1)
        print(f"  пачка {s // B + 1}/{math.ceil(len(labels) / B)}, {time.time() - t1:.0f} с: фон LGN "
              f"{float(r0['LGN'].mean()) / 0.1:.1f}, E {float(r0['L4 E'].mean()) / 0.1:.2f}, I {float(r0['L4 I'].mean()) / 0.1:.2f} Гц; "
              f"показ LGN {float(r['LGN'].mean()) * 1000 / show:.1f}, E {float(r['L4 E'].mean()) * 1000 / show:.2f}, "
              f"I {float(r['L4 I'].mean()) * 1000 / show:.2f} Гц; доля LGN в возбуждении E {frac[-1]:.0%}; "
              "потеряно событий " + ", ".join(f"{k} {float(v):.0f}" for k, v in net.lost.items()), flush=True)

    # Метки — первая половина показов каждой буквы, проверка — вторая.
    half = trials // 2
    lab_mask = torch.arange(len(labels)) < half * len(letters)
    y_lab, y_test = labels[lab_mask], labels[~lab_mask]
    print(f"\n{len(letters)} букв, метки {half} и проверка {trials - half} показов каждой (случайно {1 / len(letters):.2f}):")
    for k in res:
        R = torch.cat(res[k])
        r_lab, r_test = R[lab_mask], R[~lab_mask]
        mean = torch.stack([r_lab[y_lab == j].mean(0) for j in range(len(letters))])
        label = mean.argmax(0)
        active = mean.max(0).values > 0
        score = torch.stack([r_test[:, (label == j) & active].mean(1) if ((label == j) & active).any()
                             else torch.full((len(y_test),), -1.0) for j in range(len(letters))], 1)
        acc = float((score.argmax(1) == y_test).float().mean())
        z = lambda x: (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True).clamp_min(1e-9)
        acc_c = float((((z(r_test) @ z(mean).T) / mean.shape[1]).argmax(1) == y_test).float().mean())
        rate_bg = float(torch.cat(bg[k]).mean())
        rate = float(R.mean()) * 1000 / show
        print(f"  {k:6s} точность {acc:.2f} (голос клеток), {acc_c:.2f} (корреляция); фон {rate_bg:.2f}, "
              f"показ {rate:.2f} Гц; клеток с ответом {int(active.sum())}/{R.shape[1]}", flush=True)
    print(f"  по времени показа (корреляция; частота, Гц): 0–50 мс / 50–{show:.0f} мс")
    for k in res:
        acc = []
        for part in (early, late):
            R = torch.cat(part[k])
            mean = torch.stack([R[lab_mask][y_lab == j].mean(0) for j in range(len(letters))])
            zz = lambda x: (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True).clamp_min(1e-9)
            acc.append(float((((zz(R[~lab_mask]) @ zz(mean).T) / mean.shape[1]).argmax(1) == y_test).float().mean()))
        re, rl = float(torch.cat(early[k]).mean()) / 0.05, float(torch.cat(late[k]).mean()) / ((show - 50) / 1000)
        print(f"  {k:6s} точность {acc[0]:.2f} / {acc[1]:.2f}; частота {re:.2f} / {rl:.2f}", flush=True)

    # Контроль числа клеток: у эталона E в сотни раз больше, чем входов LGN, —
    # та же мера на случайных подвыборках L4 E (по 5 выборок).
    R = torch.cat(res["L4 E"])
    z = lambda x: (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True).clamp_min(1e-9)
    for m in (512, 2000, 10000):
        accs = []
        for _ in range(5):
            sub = R[:, torch.randperm(R.shape[1], generator=gen)[:m]]
            mean = torch.stack([sub[lab_mask][y_lab == j].mean(0) for j in range(len(letters))])
            accs.append(float((((z(sub[~lab_mask]) @ z(mean).T) / m).argmax(1) == y_test).float().mean()))
        print(f"  L4 E, {m} случайных клеток: точность (корреляция) {sum(accs) / len(accs):.2f} "
              f"({min(accs):.2f}–{max(accs):.2f})", flush=True)
    if "save" in opts:
        torch.save({"labels": labels, "letters": letters, **{k: torch.cat(v) for k, v in res.items()}}, opts["save"])
    print(f"доля LGN в возбуждении E на показе {sum(frac) / len(frac):.0%} (у авторов < 20% у простых)")
    print(f"всего {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
