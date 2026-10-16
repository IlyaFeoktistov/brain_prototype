"""
Согласование анатомического мозга с частотами in vivo в контексте всего
мозга (ARCHITECTURE 11): фон каждого ядра (BrainConfig.nucleus_bias_mv) и
фон колонок по слою и типу (column_gain_external_pop) подстраиваются шагами
«стреляет больше нормы — меньше фона» (как веса фона по типам у Billeh et
al. 2020). Цели — anatomy.RATES_HZ и column_target_hz.

    PYTHONPATH=. .venv/bin/python data/tools/anatomy_fit.py [steps=20] [ms=500] [dt=0.5] [fresh=0] [write=1]

Стартует с чисел brain/anatomy.py (fresh=1 — фон колонок с нуля), в конце
печатает найденное и записывает его туда же (NUCLEUS_BIAS_MV,
COLUMN_EXTERNAL_POP); write=0 — только напечатать.
"""

import math
import sys

import torch

from brain import anatomy
from brain.config import BrainConfig
from brain.mind import Mind

opts = {"steps": "20", "ms": "1000", "warm": "2500", "dt": "0.5", "size": "small", "fresh": "0", "write": "1", "fresh_nuclei": "0", "max_mv": "2", "trace": "data/measure/logs/fit-steps.jsonl"}

for kv in sys.argv[1:]:
    k, v = kv.split("=", 1)
    opts[k] = v

from brain.engine import KINDS


# Сила E → тип тормозных (общая для зон) — подстраивается по средней
# частоте типа, как в column_fit fit_from_e (Billeh et al. 2020): тормозных
# гонит сеть, фоном их не опустить.
from_e = list(BrainConfig().column_gain_from_e)


def measure(bias: dict, pop: dict) -> tuple[dict, dict, dict]:
    torch.manual_seed(0)
    cfg = BrainConfig()
    cfg.column_gain_from_e = tuple(from_e)
    cfg.save_path = "/nonexistent/brain.pt"
    cfg.autosave = False
    cfg.anatomy = True
    cfg.hand = True
    cfg.size = opts["size"]
    cfg.dt_ms = float(opts["dt"])
    cfg.nucleus_bias_mv = tuple(bias.items())
    cfg.column_gain_external_pop = tuple((*key, g) for key, g in pop.items())
    mind = Mind(cfg)
    brain = mind.brain
    brain.plasticity = False
    # Разгон: после сборки кора оседает ~2–3 с (ПФК PV с ~100 до ~12 Гц на
    # ~2.5-й секунде) — мерить покой, а не переходный процесс.
    brain.run(float(opts["warm"]))
    rates = brain.observe(float(opts["ms"]))
    nuclei = {name: float(rates[area.E.slice].mean()) for name, area in brain.areas.items()
              if area.layers is None and name in anatomy.RATES_HZ}
    columns: dict = {}

    for area in brain.areas.values():
        if area.layers is None:
            continue

        for layer, group in area.layers.items():
            for kind, cells in group.items():
                columns[(area.name, layer, kind)] = float(rates[cells.slice].mean())

    # Мозг шага больше не нужен — освободить видеокарту до сборки следующего
    # (пики сборки уходят в ОЗУ; без этого WSL падал «device not ready»).
    del mind, brain, rates
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    return nuclei, columns, cfg


base = BrainConfig()
targets = {(layer, kind): hz for layer, *v in base.column_target_hz for kind, hz in zip(KINDS, v)}
# Старт — уже найденное (anatomy.py), продолжить подбор.
bias: dict = dict(anatomy.NUCLEUS_BIAS_MV)

if opts["fresh_nuclei"] == "1":
    # Фон ядер — с нуля (после изменения устройства ядер); узел сна —
    # прежний, его согласует sleep_fit.
    bias = {n: v for n, v in bias.items() if n in anatomy.SLEEP_RATES_HZ}
# Фон колонок — свой у каждой зоны (зона, слой, тип); старт — общий по
# (слой, тип) из anatomy.py.
common = {(e[0], e[1]): e[-1] for e in anatomy.COLUMN_EXTERNAL_POP if len(e) == 3}
own = {tuple(e[:3]): e[-1] for e in anatomy.COLUMN_EXTERNAL_POP if len(e) == 4}
pop: dict = {}

if opts["fresh"] == "1":
    # Фон колонок — с нуля (после изменения устройства входов).
    common, own = {}, {}

# Ошибка от шага к шагу скачет (базальные ганглии, гиппокамп) — записать
# лучший шаг, а не последний.
best = (math.inf, dict(bias), dict(pop))

for step in range(int(opts["steps"])):
    nuclei, columns, cfg = measure(bias, pop)

    if not pop:
        pop = {key: own.get(key, common.get(key[1:], 1.0)) for key in columns}
        nuclei, columns, cfg = measure(bias, pop)
    errors = [math.log((r + 0.5) / (anatomy.RATES_HZ[n] + 0.5)) ** 2 for n, r in nuclei.items()]
    # LAMP5 — вне ошибки: частота in vivo не измерена (цель — «как VIP»
    # [сверить]), а при десенситизации (Karayannis 2010) их синапсы в покое
    # почти не действуют; погоня за ней портила E → PV.
    errors += [math.log((r + 0.5) / (targets[k[1:]] + 0.5)) ** 2 for k, r in columns.items() if k[2] != "LAMP5"]

    if opts["trace"]:
        # Числа каждого шага — чтобы сломанный шаг можно было собрать заново
        # и разложить (data/tools/anatomy_inputs.py step=N).
        import json

        with open(opts["trace"], "a", encoding="utf-8") as f:
            f.write(json.dumps({"step": step, "error": sum(errors) / len(errors), "bias": bias, "from_e": from_e,
                                "pop": [[*key, g] for key, g in pop.items()]}, ensure_ascii=False) + "\n")
    print(f"шаг {step:2d} ошибка {sum(errors) / len(errors):.3f}  "
          + " ".join(f"{n}={r:.1f}" for n, r in nuclei.items()), flush=True)
    if sum(errors) / len(errors) < best[0]:
        best = (sum(errors) / len(errors), dict(bias), dict(pop))

        if opts["write"] == "1":
            # Лучшее — сразу в файл: прерванный подбор не теряет найденное.
            anatomy.write_fitted(tuple(bias.items()), tuple((*key, g) for key, g in pop.items()))

    worst = sorted([kv for kv in columns.items() if kv[0][2] != "LAMP5"], key=lambda kv: -abs(math.log((kv[1] + 0.5) / (targets[kv[0][1:]] + 0.5))))[:8]
    print("        хуже всех: " + " ".join(f"{a}.{l}{k}={r:.1f}/{targets[(l, k)]}" for (a, l, k), r in worst), flush=True)

    for name, r in nuclei.items():
        if name in anatomy.SLEEP_RATES_HZ:
            # Узел сна двустабилен — его согласует sleep_fit по двум состояниям.
            continue

        # Ток: шаг по логарифму отношения частот (1 мВ ≈ заметная доля пути
        # до порога), ограничен ±2 мВ за шаг.
        step_mv = 2.0 * math.log((anatomy.RATES_HZ[name] + 0.5) / (r + 0.5))
        limit = float(opts["max_mv"])
        bias[name] = bias.get(name, 0.0) + max(-limit, min(limit, step_mv))

    if opts.get("fit_from_e", "1") == "1":
        # Только PV и SOM: VIP гонит не местный E, а top-down пути между
        # зонами (их PV/VIP — _cortical_path), LAMP5 — вне подбора (выше).
        for i, kind in enumerate(KINDS[1:]):
            if kind not in ("PV", "SOM"):
                continue
            hz = [r for k, r in columns.items() if k[2] == kind]
            aim = [targets[k[1:]] for k in columns if k[2] == kind]
            ratio = ((sum(aim) / len(aim) + 0.5) / (sum(hz) / len(hz) + 0.5)) ** 0.15
            from_e[i] *= min(1.15, max(1 / 1.15, ratio))

        print("        E → PV/SOM/VIP/LAMP5 ×" + " ".join(f"{g:.3f}" for g in from_e), flush=True)

    for key, r in columns.items():
        if key[2] == "LAMP5":
            continue
        # Мягко (±15%, показатель 0.15): популяции колонки связаны в петлю
        # (E ↔ PV ↔ VIP), резкий шаг (±30%) раскачивал её от шага к шагу.
        pop[key] *= min(1.15, max(0.87, ((targets[key[1:]] + 0.5) / (r + 0.5)) ** 0.15))

print(f"лучший шаг — ошибка {best[0]:.3f}")
bias, pop = best[1], best[2]
print("nucleus_bias_mv =", tuple((n, round(v, 2)) for n, v in bias.items()))
print("column_gain_external_pop =", tuple((*key, round(g, 3)) for key, g in pop.items()))

if opts["write"] == "1":
    # Записать найденное в brain/anatomy.py вместо прежних чисел.
    path = anatomy.write_fitted(tuple(bias.items()), tuple((*key, g) for key, g in pop.items()))
    print(f"записано в {path}")
