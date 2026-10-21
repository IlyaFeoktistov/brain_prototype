"""
Кальциевое правило пластичности на стенде (без движка): повторяет ли оно
STDP коры человека и — без подгонки — фармакологию тех же опытов.

Механизм — из Verhoog et al. 2013 (https://doi.org/10.1523/JNEUROSCI.3158-13.2013):
знак изменения задаёт кальций поста — через NMDA (нужны глутамат и
деполяризация, снимающая магниевый блок) и через L-тип (спайк и следовая
деполяризация, ADP). Правило — пороги кальция (Shouval, Bear & Cooper 2002,
https://doi.org/10.1073/pnas.152343699; Graupner & Brunel 2012,
https://doi.org/10.1073/pnas.1109359109): время выше θ_d ослабляет, выше θ_p
(> θ_d) усиливает; изменение за пару — γ_p · T(c > θ_p) − γ_d · T(c > θ_d),
пары (через 7 с) складываются.

Напряжение в синапсе: покой + ВПСП + спайк + ADP (числа клетки —
anatomy.HUMAN_CORTEX_CELL). Кальций в единицах кальция одного спайка (1 —
вход спайка, из него доля L-типа — по нифедипину); NMDA — a_N · g(t) · B(V),
блок магнием 1 мМ (Jahr & Stevens 1990). Свободные (данными не заданы):
a_N, τ кальция, τ спада NMDA, θ_d, θ_p, γ_d, γ_p.

Подгонка — только по окну (anatomy.HUMAN_CORTEX_STDP), поиском поколениями
(метод перекрёстной энтропии) на видеокарте: сотни наборов параметров по
всем протоколам разом. Проверка вне подгонки — anatomy.HUMAN_CORTEX_STDP_TESTS
(нифедипин, MK801, ADP погашен) и +10 мс у Louth et al. 2021 (75 пар).

    PYTHONPATH=. .venv/bin/python tools/calcium_stdp.py [generations=80] [population=512] [seed=0]
"""

import math
import sys

import torch

from brain import anatomy

opts = {"generations": "80", "population": "512", "seed": "0"}

for kv in sys.argv[1:]:
    k, v = kv.split("=", 1)
    opts[k] = v

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(int(opts["seed"]))
cell = anatomy.HUMAN_CORTEX_CELL
DT = 0.2
T = torch.arange(-400.0, 600.0, DT, device=dev)
L = len(T)
PAIRS = anatomy.HUMAN_CORTEX_STDP_PAIRINGS
# ВПСП: нарастание ~ амплитуда / наклон (3 мс) — τ нарастания 1 мс, спад 34.2.
EPSP_RISE = 1.0
# Спайк в дендрите — 1 мс у 0 мВ [сверить: амплитуда обратного спайка в
# 100–150 мкм у человека здесь не измерена].
SPIKE_MV, SPIKE_MS = 0.0, 1.0
# Подъём проводимости NMDA, мс [крыса/мышь, сверить].
NMDA_RISE = 2.0
MG_MM = 1.0
CONDITIONS = {"контроль": {}, "нифедипин": dict(nifedipine=True), "MK801": dict(mk801=True), "без ADP": dict(no_adp=True)}


def protocol(spikes, nifedipine=False, mk801=False, no_adp=False):
    """Напряжение синапса, блок NMDA магнием и вход кальция спайков — от параметров не зависят."""

    shape = torch.where(T >= 0, torch.exp(-T / cell["epsp_decay_ms"]) - torch.exp(-T / EPSP_RISE), torch.zeros_like(T))
    v = cell["rest_mv"] + cell["epsp_mv"] * shape / shape.max()
    adp = cell["adp_mv"] * (cell["nif_adp_amplitude"] if nifedipine else 1.0)
    tau_adp = cell["adp_area"] / cell["adp_mv"] * (cell["nif_adp_area"] / cell["nif_adp_amplitude"] if nifedipine else 1.0)
    ap = torch.zeros_like(T)

    for ts in spikes:
        dt = T - ts
        v = torch.where((dt >= 0) & (dt < SPIKE_MS), torch.full_like(T, SPIKE_MV), v)
        after = dt - SPIKE_MS

        if not no_adp:
            v = v + torch.where(after >= 0, adp * torch.exp(-after.clamp(min=0) / tau_adp), torch.zeros_like(T))
        ap[int(torch.searchsorted(T, torch.tensor(ts, device=dev)))] += 1.0 - (cell["ap_ca_ltype"] if nifedipine else 0.0)

    block = 1.0 / (1.0 + MG_MM / 3.57 * torch.exp(-0.062 * v))
    return block * (0.0 if mk801 else 1.0), ap


# Протоколы: окно (по 5 интервалов в каждом бине опыта), проверки, окно модели.
DATA = anatomy.HUMAN_CORTEX_STDP
WIDTH = {d: (5.0 if abs(d) < 20 else 30.0) for d, *_ in DATA}
rows, index = [], {}


def add(key, spikes, **kw):
    index.setdefault(key, []).append(len(rows))
    rows.append((spikes, kw))


for d, *_ in DATA:
    for dt in torch.linspace(d - WIDTH[d] / 2, d + WIDTH[d] / 2, 5).tolist():
        add(("окно", d), [dt])

for what, (lo, hi), *_ in anatomy.HUMAN_CORTEX_STDP_TESTS:
    for dt in torch.linspace(lo, hi, 5).tolist():
        add(("опыт", what, lo, hi), [dt], **CONDITIONS[what])

GRID = (-250, -200, -150, -130, -100, -80, -60, -40, -20, -10, -5, 0, 5, 8, 10, 15, 20, 40, 60, 100, 150)

for d in GRID:
    add(("сетка", d), [float(d)])

BLOCK = torch.stack([protocol(s, **kw)[0] for s, kw in rows])  # [Q, L]
AP = torch.stack([protocol(s, **kw)[1] for s, kw in rows])  # [Q, L]
FIT = torch.tensor([i for d, *_ in DATA for i in index[("окно", d)]], device=dev)
NF = 2 * L

# Параметры — в логарифмах: a_N, τ кальция, τ NMDA, θ_d, θ_p / θ_d − 1, γ_d, γ_p.
NAMES = ["a_N (NMDA, ед. спайка)", "τ кальция, мс", "τ NMDA, мс", "θ_d", "θ_p", "γ_d, 1/мс", "γ_p, 1/мс"]
LOW = torch.log(torch.tensor([0.01, 5.0, 20.0, 0.02, 0.05, 1e-7, 1e-7], device=dev))
HIGH = torch.log(torch.tensor([50.0, 200.0, 150.0, 20.0, 20.0, 1e-2, 1e-2], device=dev))


def unpack(x):
    p = torch.exp(x)
    return p[:, 0], p[:, 1], p[:, 2], p[:, 3], p[:, 3] * (1.0 + p[:, 4]), p[:, 5], p[:, 6]


def changes(x, rows_idx, pairs=PAIRS, chunk=64):
    """Изменение веса [наборы, протоколы]."""

    out = []
    t = torch.arange(L, device=dev) * DT
    tt = (T - T[0]).clamp(min=0)

    for i in range(0, len(x), chunk):
        a_n, tau_ca, tau_n, th_d, th_p, g_d, g_p = unpack(x[i:i + chunk])
        g = torch.exp(-tt[None, :] / tau_n[:, None]) - torch.exp(-tt[None, :] / NMDA_RISE)
        g = g / g.amax(1, keepdim=True)
        # NMDA открывается от глутамата пре в t = 0: сдвиг на начало окна.
        shift = int(torch.searchsorted(T, torch.tensor(0.0, device=dev)))
        g = torch.nn.functional.pad(g, (shift, 0))[:, :L]
        influx = a_n[:, None, None] * g[:, None, :] * BLOCK[None, rows_idx, :] * DT + AP[None, rows_idx, :]
        kernel = torch.exp(-t[None, :] / tau_ca[:, None])
        c = torch.fft.irfft(torch.fft.rfft(influx, NF) * torch.fft.rfft(kernel, NF)[:, None, :], NF)[..., :L]
        t_d = (c > th_d[:, None, None]).sum(-1) * DT
        t_p = (c > th_p[:, None, None]).sum(-1) * DT
        out.append(pairs * (g_p[:, None] * t_p - g_d[:, None] * t_d))

    return torch.cat(out)


Y = torch.tensor([y for _, y, _ in DATA], device=dev)
# Нижняя граница SEM 0.05: SEM, снятые с рисунка 640 px у бинов с 1–3 клетками,
# ненадёжны (у +12.5 мс — ±0.02 при ±0.13–0.22 у соседей).
SEM = torch.tensor([s for *_, s in DATA], device=dev).clamp(min=0.05)


def loss(x):
    ch = changes(x, FIT).view(len(x), len(DATA), 5).mean(-1)
    return (((ch - Y) / SEM) ** 2).sum(-1)


# Поиск поколениями: выборка вокруг лучших, их среднее и разброс — следующее поколение.
P = int(opts["population"])
mean, std = (LOW + HIGH) / 2, (HIGH - LOW) / 2
best_x, best_l = None, float("inf")

for gen in range(int(opts["generations"])):
    x = (mean + std * torch.randn(P, len(NAMES), device=dev)).clamp(LOW, HIGH)
    if best_x is not None:
        x[0] = best_x
    l = loss(x)
    order = torch.argsort(l)
    elite = x[order[: P // 10]]
    mean, std = elite.mean(0), elite.std(0) * 0.9 + 0.02 * (HIGH - LOW)

    if float(l[order[0]]) < best_l:
        best_l, best_x = float(l[order[0]]), x[order[0]].clone()

    if gen % 10 == 0:
        print(f"поколение {gen}: лучшая ошибка {best_l:.2f}", flush=True)

final = (mean + std * torch.randn(4 * P, len(NAMES), device=dev)).clamp(LOW, HIGH)
final[0] = best_x
lf = loss(final)
best = final[int(torch.argmin(lf))][None]
err = float(lf.min())
print(f"\nокно коры: {len(DATA)} точек, ошибка (χ²) {err:.2f}")
values = [float(v[0]) for v in unpack(best)]

for name, value in zip(NAMES, values):
    print(f"  {name:24s} {value:.4g}")

close = final[lf <= err * 1.5 + 1.0]
print(f"\nнаборов с ошибкой ≤ 1.5 × лучшей + 1: {len(close)} — разброс (что данные определяют):")

for name, vals in zip(NAMES, unpack(close)):
    print(f"  {name:24s} {float(vals.min()):.4g} … {float(vals.max()):.4g}")

window = changes(best, torch.tensor([i for d, *_ in DATA for i in index[('окно', d)]], device=dev))[0].view(len(DATA), 5).mean(-1)
print("\nокно — опыт ± SEM / модель:")

for (d, y, s), m in zip(DATA, window.tolist()):
    print(f"  {d:+7.1f} мс: {y:+.2f} ± {s:.2f} / {m:+.2f}")

print("\nпроверка вне подгонки (Verhoog 2013) — опыт ± SEM / модель:")
hits = 0

for what, (lo, hi), y, s, n in anatomy.HUMAN_CORTEX_STDP_TESTS:
    m = float(changes(best, torch.tensor(index[("опыт", what, lo, hi)], device=dev))[0].mean())
    ok = abs(m - y) <= 2 * s
    hits += ok
    print(f"  {what:10s} {lo:+.0f}…{hi:+.0f} мс: {y:+.2f} ± {s:.2f} / {m:+.2f}  {'в пределах 2 SEM' if ok else 'НЕ СХОДИТСЯ'}")

dt, pairs, y, s, n = anatomy.HUMAN_CORTEX_STDP_DOPAMINE[0]
m = float(changes(best, torch.tensor(index[("сетка", 10)], device=dev), pairs=pairs)[0, 0])
ok = abs(m - y) <= 2 * s
hits += ok
print(f"  Louth +10 мс, 75 пар: {y:+.2f} ± {s:.2f} / {m:+.2f}  {'в пределах 2 SEM' if ok else 'НЕ СХОДИТСЯ'}")
print(f"\nсходится {hits} из {len(anatomy.HUMAN_CORTEX_STDP_TESTS) + 1}")
print("\nокно одной пары (модель), * — вне проверенного −130…+60:")
grid = changes(best, torch.tensor([index[("сетка", d)][0] for d in GRID], device=dev))[0].tolist()

for d, m in zip(GRID, grid):
    print(f"  {'*' if not -130 <= d <= 60 else ' '}{d:+5d} мс: {m:+.3f}")
