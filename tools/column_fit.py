"""
Согласование колонки с частотами in vivo (ARCHITECTURE 10, «Согласование»):
одна колонка (слуховая зона, без других зон) живёт в покое; 11 общих на тип
множителей (BrainConfig.column_gain_*: внешний вход E / PV / SOM / VIP, E →
тип, тип → E, I → I) подбираются так, чтобы частоты по слоям и типам
совпали с column_target_hz. Поиск — Нелдер–Мид по логарифмам множителей,
ошибка — средний квадрат log((r + 0.5) / (цель + 0.5)) по 16 популяциям.

    PYTHONPATH=. .venv/bin/python tools/column_fit.py [neurons=10000] [evals=300] [ms=1000]

Печатает лучшие множители — их переносят в config.py с пометкой.
"""

import math
import sys

import torch
from dataclasses import replace

from brain.config import BrainConfig, default_areas, log_path
from brain.engine import SpikingBrain

opts = {"neurons": "10000", "evals": "300", "ms": "1000", "area": "auditory", "mode": "gains", "steps": "60", "dt": "2.0",
        "batch": "24", "generations": "30", "warmup": "300", "anatomy": "0", "kick": "0", "start": ""}

for kv in sys.argv[1:]:
    k, v = kv.split("=", 1)
    opts[k] = v

from brain.engine import KINDS as TYPES


def build(gains: list[float], pop: dict | None = None) -> SpikingBrain:
    cfg = BrainConfig()
    cfg.layers = True
    cfg.eye = False
    cfg.compile_step = False
    # anatomy=1 — колонка как в анатомическом мозге: торможение
    # проводимостью у всех, E → E слоя 2/3 с весами по данным.
    cfg.anatomy = opts["anatomy"] == "1"
    cfg.size = opts["neurons"]
    cfg.dt_ms = float(opts["dt"])
    cfg.areas = [a for a in default_areas() if a.name == opts["area"]]
    cfg.pathways = []
    cfg.column_gain_external = tuple(gains[0:5])
    cfg.column_gain_from_e = tuple(gains[5:9])
    cfg.column_gain_to_e = tuple(gains[9:13])
    cfg.column_gain_ii = gains[13]
    cfg.column_gain_external_pop = tuple((l, t, g) for (l, t), g in (pop or {}).items())
    return SpikingBrain(cfg)


def measure(gains: list[float], pop: dict | None = None) -> dict:
    torch.manual_seed(0)
    brain = build(gains, pop)
    brain.plasticity = False
    brain.run(float(opts["warmup"]))
    rates = brain.observe(float(opts["ms"]))
    area = brain.areas[opts["area"]]
    return {(layer, kind): float(rates[pop.slice].mean()) for layer, group in area.layers.items() for kind, pop in group.items()}


# Тормозная проводимость к возбуждающей у клеток E в покое, бодрствование:
# 1.7 ± 1.1 (Rudolph et al. 2007, кошка) — цель для силы торможения.
GI_GE = 1.7


def conductance_ratio(brain, cells) -> float:
    """g_I/g_E клеток cells: ток ÷ движущая сила при нормировке (возбуждение
    — к −70 мВ, 70 мВ до E_AMPA; торможение — к −55, 19 мВ до E_ГАМК), со
    средним полем недостающих соседей и фоном."""

    c = brain.cfg
    nxt = brain.stp_u4 + brain.stp_U4 * (1 - brain.stp_u4)
    gain = torch.where(brain.typed_pre[:, None], nxt * brain.stp_x4 / brain.stp_U4, 1.0)
    gain = torch.where(brain.is_lamp5[:, None], gain * brain.ngf_d[:, None], gain)
    n = gain.shape[1]
    G = torch.ones(brain.mf_size.shape[0], n + 1, device=brain.device)
    G[:, :n] = torch.zeros(brain.mf_size.shape[0], n, device=brain.device).index_add_(0, brain.mf_pre_id, gain)
    G[:, :n] /= brain.mf_size[:, None]
    g = G.index_select(1, brain.mf_post_type).t()
    dc_e = brain.dc_mv + (((brain.mf_exc - brain.acetylcholine * brain.mf_ach) * g) @ brain.mf_rate).index_select(0, brain.mf_post_id)
    dc_i = brain.dc_inh_mv + ((brain.mf_inh * g) @ brain.mf_rate).index_select(0, brain.mf_post_id)
    e = (brain.i_syn + brain.i_slow + dc_e) / (c.ampa_reversal_mv - c.epsp_holding_mv)
    i = -(brain.i_inh + brain.i_ngf + brain.ngf_mf + dc_i) / (c.ipsp_holding_mv - c.gaba_reversal_mv)
    return float(i[cells].mean() / e[cells].mean().clamp(min=1e-6))


def loss(rates: dict, targets: dict) -> float:
    # E — главные клетки коры: их частота весит вчетверо (иначе поиск
    # находит «дешёвое» решение — молчащие E при верных тормозных).
    # Ключ («после», слой, тип) — замер после толчка, цель та же; ключ
    # («gI/gE»,) — отношение проводимостей E в покое (цель GI_GE, вес 4).
    weights = {key: 4.0 if key[-1] in ("E", "gI/gE") else 1.0 for key in rates}
    errors = [weights[key] * (math.log(max(r, 1e-3) / GI_GE) ** 2 if key == ("gI/gE",)
              else math.log((r + 0.5) / (targets[key[-2:]] + 0.5)) ** 2) for key, r in rates.items()]
    return sum(errors) / sum(weights.values())


cfg = BrainConfig()
targets = {(layer, kind): hz for layer, *v in cfg.column_target_hz for kind, hz in zip(TYPES, v)}
calls = [0]
best = [math.inf, None, None]


def f(x: list[float]) -> float:
    gains = [math.exp(v) for v in x]
    rates = measure(gains)
    value = loss(rates, targets)
    calls[0] += 1

    if value < best[0]:
        best[:] = [value, gains, rates]
        print(f"{calls[0]:4d} ошибка {value:.3f}  " + " ".join(f"{g:.2f}" for g in gains), flush=True)

    return value


def nelder_mead(f, x0: list[float], step: float, evals: int) -> None:
    n = len(x0)
    simplex = [list(x0)] + [[x0[j] + (step if j == i else 0.0) for j in range(n)] for i in range(n)]
    values = [f(x) for x in simplex]

    while calls[0] < evals:
        order = sorted(range(n + 1), key=lambda i: values[i])
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        centroid = [sum(x[j] for x in simplex[:-1]) / n for j in range(n)]
        worst = simplex[-1]
        reflected = [c + (c - w) for c, w in zip(centroid, worst)]
        fr = f(reflected)

        if fr < values[0]:
            expanded = [c + 2 * (c - w) for c, w in zip(centroid, worst)]
            fe = f(expanded)
            simplex[-1], values[-1] = (expanded, fe) if fe < fr else (reflected, fr)
        elif fr < values[-2]:
            simplex[-1], values[-1] = reflected, fr
        else:
            contracted = [c + 0.5 * (w - c) for c, w in zip(centroid, worst)]
            fc = f(contracted)

            if fc < values[-1]:
                simplex[-1], values[-1] = contracted, fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = [simplex[0][j] + 0.5 * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                    values[i] = f(simplex[i])


def measure_batch(population: list[list[float]]) -> list[dict]:
    """
    Поколение разом: в одной сети живут копии колонки, у каждой свои
    множители (AreaSpec.column_gains), — видеокарта считает их вместе.
    """

    torch.manual_seed(0)
    cfg = BrainConfig()
    cfg.layers = True
    cfg.eye = False
    cfg.compile_step = False
    # anatomy=1 — колонка как в анатомическом мозге: торможение
    # проводимостью у всех, E → E слоя 2/3 с весами по данным.
    cfg.anatomy = opts["anatomy"] == "1"
    cfg.dt_ms = float(opts["dt"])
    spec = next(a for a in default_areas() if a.name == opts["area"])
    copies = [replace(spec, name=spec.name if i == 0 else f"{spec.name}_{i}", column_gains=tuple(g))
              for i, g in enumerate(population)]
    cfg.areas = copies
    cfg.pathways = []
    cfg.layered_areas = tuple(a.name for a in copies)
    cfg.microcircuit_areas = cfg.layered_areas
    cfg.size = str(int(opts["neurons"]) * len(population))
    brain = SpikingBrain(cfg)
    brain.plasticity = False
    brain.run(float(opts["warmup"]))
    rest = brain.observe(float(opts["ms"]))
    # g_I/g_E клеток E каждой копии — в конце покоя (среднее 4 замеров).
    ratios = [[] for _ in copies]

    for _ in range(4):
        brain.run(25.0)

        for j, c in enumerate(copies):
            cells = torch.cat([torch.arange(g["E"].start, g["E"].stop, device=brain.device)
                               for g in brain.areas[c.name].layers.values()])
            ratios[j].append(conductance_ratio(brain, cells))
    after = None

    if opts["kick"] != "0":
        # Устойчивость: толчок L2/3 E всех копий (kick мВ, 50 мс) — кора
        # после стимула возвращается в покой (ответ проходит), горящая
        # копия проигрывает.
        mv = float(opts["kick"])
        tops = [brain.areas[c.name].layers["L23"]["E"].slice for c in copies]

        for s in tops:
            brain.bias[s] += mv

        brain.run(50.0)

        for s in tops:
            brain.bias[s] -= mv

        brain.run(200.0)
        after = brain.observe(float(opts["ms"]))

    out = []

    for copy in copies:
        area = brain.areas[copy.name]
        pops = [(layer, kind, pop) for layer, group in area.layers.items() for kind, pop in group.items()]
        item = {(layer, kind): float(rest[pop.slice].mean()) for layer, kind, pop in pops}
        item[("gI/gE",)] = sum(ratios[copies.index(copy)]) / 4

        if after is not None:
            item.update({("после", layer, kind): float(after[pop.slice].mean()) for layer, kind, pop in pops})

        out.append(item)

    return out


if opts["mode"] == "batch":
    # Эволюционная стратегия (μ/μ, λ) по логарифмам множителей: поколение из
    # batch вариантов вокруг среднего, лучшая четверть сдвигает среднее, шаг
    # уменьшается, если поколение не улучшает лучшее (Hansen 2016 — упрощённо).
    import random

    rng = random.Random(1)
    base = BrainConfig()
    mean = [math.log(g) for g in list(base.column_gain_external) + list(base.column_gain_from_e)
            + list(base.column_gain_to_e) + [base.column_gain_ii]]

    if opts["start"]:
        # Старт: 11 множителей через запятую.
        mean = [math.log(float(g)) for g in opts["start"].split(",")]
    n = len(mean)
    sigma = float(opts.get("sigma", "0.6"))
    size = int(opts["batch"])
    elite = max(2, size // 4)

    for generation in range(int(opts["generations"])):
        xs = [list(mean)] + [[m + sigma * rng.gauss(0, 1) for m in mean] for _ in range(size - 1)]
        results = measure_batch([[math.exp(v) for v in x] for x in xs])
        scored = sorted(((loss(r, targets), x, r) for x, r in zip(xs, results)), key=lambda t: t[0])
        improved = scored[0][0] < best[0]

        if improved:
            best[:] = [scored[0][0], [math.exp(v) for v in scored[0][1]], scored[0][2]]

        mean = [sum(x[j] for _, x, _ in scored[:elite]) / elite for j in range(n)]
        sigma *= 1.0 if improved else 0.85
        calls[0] += size
        print(f"поколение {generation:2d} лучшая {scored[0][0]:.3f} (всего {best[0]:.3f}) шаг {sigma:.2f}  "
              + " ".join(f"{g:.2f}" for g in best[1]), flush=True)
        r = best[2]
        print("        покой " + " ".join(f"{k[0]}{k[1]}={v:.1f}" for k, v in r.items() if len(k) == 2)
              + f"  gI/gE {r[('gI/gE',)]:.2f}  после: "
              + " ".join(f"{k[1]}{k[2]}={v:.1f}" for k, v in r.items() if len(k) == 3 and k[2] == "E"), flush=True)
elif opts["mode"] == "pop":
    # Фон по популяциям (Billeh et al. 2020): множитель внешнего входа
    # каждой популяции × (цель / частота)^0.3 — «стреляет больше нормы —
    # меньше фона»; общие множители — из config.py.
    import json

    base = BrainConfig()
    gains = list(base.column_gain_external) + list(base.column_gain_from_e) + list(base.column_gain_to_e) + [base.column_gain_ii]
    pop = {key: 1.0 for key in targets}
    # fit_from_e=1 — ещё и сила E → тип тормозных (по средней частоте типа во
    # всех слоях), как у Billeh et al. 2020: единый множитель на класс связей
    # под спонтанные частоты in vivo; тормозных гонит сеть, а не фон.
    fit_from_e = opts.get("fit_from_e", "0") == "1"
    save = log_path(opts.get("save", "colfit-best.json"))

    if opts.get("load"):
        saved = json.load(open(log_path(opts["load"]), encoding="utf-8"))
        gains = saved["gains"]
        pop = {tuple(k.split(".")): v for k, v in saved["pop"].items()}

    for step in range(int(opts["steps"])):
        rates = measure(gains, pop)
        value = loss(rates, targets)
        print(f"{step:3d} ошибка {value:.3f}  " + " ".join(f"{k[0]}{k[1]}={r:.1f}" for k, r in rates.items()), flush=True)

        if value < best[0]:
            best[:] = [value, list(gains), rates]
            best_pop = dict(pop)
            # Лучшее — сразу в файл: остановленный подбор не теряет найденное.
            json.dump({"error": value, "gains": list(gains), "pop": {".".join(k): v for k, v in pop.items()},
                       "rates": {".".join(k): r for k, r in rates.items()}}, open(save, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)

        if fit_from_e:
            for i, kind in enumerate(TYPES[1:]):
                hz = [r for (layer, k), r in rates.items() if k == kind]
                aim = [targets[(layer, k)] for (layer, k) in rates if k == kind]
                ratio = ((sum(aim) / len(aim) + 0.5) / (sum(hz) / len(hz) + 0.5)) ** 0.15
                gains[5 + i] *= min(1.2, max(1 / 1.2, ratio))

        for key, r in rates.items():
            # Мягко: у SOM, VIP, LAMP5 отклик крутой (облегчение входа от E).
            power, cap = float(opts.get("power", "0.15")), float(opts.get("cap", "1.2"))
            pop[key] *= min(cap, max(1.0 / cap, ((targets[key] + 0.5) / (r + 0.5)) ** power))

    print("column_gain_from_e =", tuple(round(g, 3) for g in best[1][5:9]))
    print("column_gain_external_pop =", tuple((l, t, round(g, 3)) for (l, t), g in best_pop.items()))
elif opts["mode"] == "gains":
    start = [0.0] * 11
    nelder_mead(f, start, 0.7, int(opts["evals"]))

value, gains, rates = best
print(f"\nлучшее: ошибка {value:.3f}")
print("column_gain_external =", tuple(round(g, 3) for g in gains[0:5]))
print("column_gain_from_e =", tuple(round(g, 3) for g in gains[5:9]))
print("column_gain_to_e =", tuple(round(g, 3) for g in gains[9:13]))
print("column_gain_ii =", round(gains[13], 3))

for layer, *_ in cfg.column_target_hz:
    print(f"  {layer}: " + ", ".join(f"{kind} {rates[(layer, kind)]:.1f} (цель {targets[(layer, kind)]})" for kind in TYPES))
    if ("после", layer, "E") in rates:
        print(f"    после толчка: " + ", ".join(f"{kind} {rates[('после', layer, kind)]:.1f}" for kind in TYPES))
