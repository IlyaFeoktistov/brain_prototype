"""
Сеть зубчатой извилины → data/hippocampus/dg_dyhrfjeld.json (формат
brain/circuit.py, как CA3 Kopsick).

Четыре типа клеток функциональной модели DG Dyhrfjeld-Johnsen et al. 2007
(J Neurophysiol 97:1566, https://doi.org/10.1152/jn.00950.2006; КРЫСА):
гранулярные (GC), мшистые (MC) — возбуждающие; корзинчатые (BC, PV) и HIPP
— тормозные. Остальные типы DG авторы из функциональной модели убрали — нет
электрофизиологии; мы так же.

Откуда числа:
  - численность (крыса) и связи — табл. 1 Dyhrfjeld-Johnsen et al. 2007:
    дивергенция (на сколько клеток типа-получателя выходит одна клетка
    типа-источника, средние из диапазонов). Сверено с кодом группы (Morgan &
    Soltesz 2008, ModelDB 124513, p*c.hoc): их вероятности по источнику
    отличаются от табл. 1 одним множителем (GC ×17, MC ×3.0, BC ×12.8, HIPP
    ×4.3) — пропорции совпадают; BC → HIPP в коде 0, в таблице 0.5.
  - сила синапса (проводимость g, нС) — из той же модели: код группы для
    матрицы табл. 1 (Morgan & Soltesz 2008, ModelDB 124513, файлы
    <пре>.<пост>: вес NetCon в мкС). Проводимости Hippocampome подобраны под
    одиночную пару и с числом входов табл. 1 не сходятся: одиночный ВПСП MC →
    GC 6.8 мВ при 975 входах, сеть разгонялась (HIPP 667 Гц, MC 55 Гц,
    гранулярные в деполяризационном блоке; CHAT_PLAN шаг 3, цикл 6).
  - клетки (Izhikevich 2007, 9 параметров) и синапсы (спад τ_d,
    Tsodyks–Markram U, τ_u, τ_x, задержка) — Hippocampome.org v2,
    набор full_scale_DG_snn (МЫШЬ; generateCONFIGStateSTP.h для CARLsim,
    https://hippocampome.org/php/data/default_zipfiles/full_scale_DG_snn.zip;
    Wheeler et al. 2024, eLife, https://doi.org/10.7554/eLife.90597). Клетка
    — подтип 1 страницы типа (тот же, что в наборе).
  - HIPP — клетка AdEx модели DG Chavlis et al. 2017 (Hippocampus 27:89,
    https://doi.org/10.1002/hipo.22675; ModelDB 206372, HIPPCell.py /
    DG_3dendrites.py; параметры — по записям HIPP линии Santhakumar 2005).
    Izhikevich HIPP Hippocampome (d = −58 пА) залипает: от 200 пА за 200 мс
    клетка после снятия тока стреляет ~700 Гц бесконечно (u уходит в −13 нА)
    — в сети HIPP стояли на потолке частоты (CHAT_PLAN шаг 3, цикл 6).
  - пары без синапса в Hippocampome (BC → MC, HIPP → MC: по морфологии
    аксонов Hippocampome их не видит, у Dyhrfjeld-Johnsen они есть) — синапс
    того же источника на гранулярную [своё: нет готового].

Фона нет: у Hippocampome сеть DG без фонового тока, вход — извне (EC); у
нас — пути мозга (brain/circuit.py: без фона вход мозга идёт целиком).

    python3 tools/dg_extract.py <каталог full_scale_DG_snn>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ca3_kopsick_extract import parse  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "data" / "hippocampus" / "dg_dyhrfjeld.json"
SOURCE = "https://hippocampome.org/php/data/default_zipfiles/full_scale_DG_snn.zip"

# Тип Hippocampome → наше имя; численность у крысы (Dyhrfjeld-Johnsen 2007, табл. 1).
TYPES = {"DG_Granule": ("DG_Granule", 1_000_000), "DG_Mossy": ("DG_Mossy", 30_000),
         "DG_Basket": ("DG_Basket", 10_000), "DG_HIPP": ("DG_HIPP", 12_000)}

# HIPP — AdEx Chavlis et al. 2017 (ModelDB 206372, DG_3dendrites.py): C
# 58.4 пФ, gL 1.93 нС, EL −59, VT −50, ΔT 2 мВ, τw 93 мс, a 0.82 нС, b 15 пА,
# сброс −56 мВ, порог (EmpiricalThreshold) −50 мВ, рефрактерность 3 мс. В
# форме circuit.py: u — w, a = 1/τw, b = a_AdEx, vr = EL, c — сброс, d — b.
HIPP_ADEX = {"model": "adex", "C": 58.4, "gL": 1.93, "EL": -59.0, "VT": -50.0, "deltaT": 2.0,
             "vr": -59.0, "a": 1.0 / 93.0, "b": 0.82, "c": -56.0, "d": 15.0, "vpeak": -50.0,
             "refractory_ms": 3.0}

# Проводимость синапса, нС (Morgan & Soltesz 2008, ModelDB 124513: вес
# NetCon <пре>.<пост>, мкС × 1000). У авторов синапсы без кратковременной
# пластичности; у нас — Tsodyks–Markram Hippocampome: первый спайк даёт g,
# дальше — по U, τ_u, τ_x.
G_NS = {
    ("DG_Granule", "DG_Mossy"): 0.20, ("DG_Granule", "DG_Basket"): 0.94, ("DG_Granule", "DG_HIPP"): 0.10,
    ("DG_Mossy", "DG_Granule"): 0.30, ("DG_Mossy", "DG_Mossy"): 0.50, ("DG_Mossy", "DG_Basket"): 0.30,
    ("DG_Mossy", "DG_HIPP"): 0.20,
    ("DG_Basket", "DG_Granule"): 1.60, ("DG_Basket", "DG_Mossy"): 1.50, ("DG_Basket", "DG_Basket"): 7.60,
    # BC → HIPP: в коде нет (0), в табл. 1 — 0.5 клетки; сила — как BC → MC [своё: нет готового].
    ("DG_Basket", "DG_HIPP"): 1.50,
    ("DG_HIPP", "DG_Granule"): 0.50, ("DG_HIPP", "DG_Mossy"): 1.00, ("DG_HIPP", "DG_Basket"): 0.50,
}

# Дивергенция (Dyhrfjeld-Johnsen 2007, табл. 1): источник → {получатель:
# клеток}; средние диапазонов: GC → MC 7–12, BC 10–20, HIPP 100–120; MC → GC
# 30 000–35 000, MC 200–500, BC 5–10, HIPP 600; BC → GC 1000–1500, MC 50–100,
# BC 20–50, HIPP 0–1; HIPP → GC 1500–1600, MC 20–50, BC 400–500.
DIVERGENCE = {
    "DG_Granule": {"DG_Mossy": 9.5, "DG_Basket": 15.0, "DG_HIPP": 110.0},
    "DG_Mossy": {"DG_Granule": 32_500.0, "DG_Mossy": 350.0, "DG_Basket": 7.5, "DG_HIPP": 600.0},
    "DG_Basket": {"DG_Granule": 1250.0, "DG_Mossy": 75.0, "DG_Basket": 35.0, "DG_HIPP": 0.5},
    "DG_HIPP": {"DG_Granule": 1550.0, "DG_Mossy": 35.0, "DG_Basket": 450.0},
}


def main(project: str) -> None:
    hc_groups, hc_links = parse((Path(project) / "generateCONFIGStateSTP.h").read_text())
    groups = {}

    for hc, (name, cells) in TYPES.items():
        g = dict(hc_groups[hc])
        g["cells"] = cells
        groups[name] = g

    groups["DG_HIPP"] = {"cells": TYPES["DG_HIPP"][1], "excitatory": False, **HIPP_ADEX}

    links = []

    for pre, targets in DIVERGENCE.items():
        for post, div in targets.items():
            src = hc_links.get((pre, post))
            borrowed = src is None

            if borrowed:
                # Нет в Hippocampome — синапс того же источника на гранулярную.
                src = hc_links[(pre, "DG_Granule")]
            link = {k: v for k, v in src.items() if k not in ("p", "plastic", "weight_max")}
            # Вероятность у крысы: дивергенция / клеток получателя.
            link.update(pre=TYPES[pre][0], post=TYPES[post][0], p=div / TYPES[post][1], divergence=div,
                        g=G_NS[(pre, post)], g_hippocampome=src["g"],
                        synapse_from=f"{pre} -> DG_Granule" if borrowed else f"{pre} -> {post}")
            links.append(link)

    out = {
        "source": SOURCE,
        "paper": ("связи и численность — Dyhrfjeld-Johnsen et al. 2007, J Neurophysiol 97:1566, "
                  "https://doi.org/10.1152/jn.00950.2006 (крыса); клетки и синапсы — Hippocampome.org v2 "
                  "(мышь), https://doi.org/10.7554/eLife.90597"),
        "note": ("Izhikevich 2007 (C пФ, k нС/мВ, v мВ, a 1/мс, b нС, d пА); синапс — проводимость нС: "
                 "Δg = weight × g × u⁺x⁻/U на спайк; E_AMPA 0, E_GABAa −70 мВ; RK4 по 0.2 мс. "
                 "Фона нет — вход извне (пути мозга)."),
        "principal": "DG_Granule",
        "background_cap_pa": 0.0,
        "groups": groups,
        "links": links,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"{OUT}: групп {len(groups)}, связей {len(links)}")
    for link in links:
        print(f"   {link['pre']:11s} → {link['post']:11s} p {link['p']:.2e}  g {link['g']:.2f}  τd {link['tau_d_ms']:.1f}  "
              f"U {link['U']:.2f}  синапс {link['synapse_from']}")


if __name__ == "__main__":
    main(sys.argv[1])
