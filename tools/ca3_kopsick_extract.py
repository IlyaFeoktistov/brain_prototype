"""
Сеть CA3 Kopsick et al. (2023, 2024) из кода авторов → data/hippocampus/ca3_kopsick.json.

Источник — файл сети авторов generateCONFIGStateSTP.h (вызовы CARLsim:
createGroup, setNeuronParameters, connect, setSTP), репозиторий
github.com/jkopsick/simulate_formation_and_retrieval, коммит 1989764
(formation/project/ca3_full_net_lognormalI_09_05_23_assem_275). Числа не
переписываются руками: разбор вызовов, чтобы перенос совпадал с кодом,
на котором авторы получили устойчивый покой (docs/CA3_TRANSFER.md).

Фоновый ток (lognormal μ, σ по типу × U(0, 1), потолок 10 000 пА) — из
src/main_*.cpp того же каталога, покой (0–500 мс): PC 4.0/1.5 и т.д.

    python3 tools/ca3_kopsick_extract.py <каталог проекта авторов>
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "hippocampus" / "ca3_kopsick.json"
SOURCE = ("https://github.com/jkopsick/simulate_formation_and_retrieval/blob/"
          "1989764942516bf110dafc29ad08d815fd699f4c/formation/project/"
          "ca3_full_net_lognormalI_09_05_23_assem_275/generateCONFIGStateSTP.h")

NUM = r"([-+0-9.eE*]+)f?"


def number(text: str) -> float:
    """Число из C++: «0.5f», «1.4*0.19f» (авторы так поправили U у PC → PC)."""

    value = 1.0
    for part in text.rstrip("f").split("*"):
        value *= float(part.rstrip("f"))
    return value


def strip_comments(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return re.sub(r"//[^\n]*", "", src)


def parse(src: str) -> tuple[dict, dict]:
    """
    Группы и связи из файла сети CARLsim (generateCONFIGStateSTP.h, формат
    Hippocampome.org): createGroup, setNeuronParameters, connect, setSTP.
    Возвращает (группы, связи по паре (пре, пост)).
    """

    src = strip_comments(src)
    groups = {}
    for name, count, kind in re.findall(r'createGroup\("(\w+)",\s*(\d+),\s*(\w+)', src):
        groups[name] = {"cells": int(count), "excitatory": kind == "EXCITATORY_NEURON"}

    # setNeuronParameters 9-параметровая: C, k, vr, vt, a, b, vpeak, c, d (у
    # каждого — среднее и разброс; разброс у авторов 0), рефрактерность.
    keys = ("C", "k", "vr", "vt", "a", "b", "vpeak", "c", "d")
    for m in re.finditer(r"setNeuronParameters\((\w+),([^;]*)\);", src):
        values = [number(x.strip()) for x in m.group(2).split(",")]
        if len(values) == 10:
            # Набор Hippocampome: 9 параметров без разброса + рефрактерность.
            values = [x for v in values[:9] for x in (v, 0.0)] + values[9:]
        assert len(values) == 19, m.group(0)
        assert all(v == 0.0 for v in values[1:18:2]), f"разброс параметров у {m.group(1)}"
        groups[m.group(1)].update({k: values[2 * i] for i, k in enumerate(keys)}, refractory_ms=values[18])

    links = {}
    pat = (r"connect\((\w+),\s*(\w+),\s*\"random\",\s*RangeWeight\(" + NUM + r",\s*" + NUM + r",\s*" + NUM
           + r"\),\s*" + NUM + r",\s*RangeDelay\((\d+)\),\s*RadiusRF\(([-0-9.]+)\),\s*(\w+),\s*" + NUM + r",\s*" + NUM + r"\)")
    for m in re.finditer(pat, src):
        pre, post, wmin, w, wmax, p, delay, radius, plastic, fast, slow = m.groups()
        assert float(radius) == -1.0 and number(slow) == 0.0
        links[(pre, post)] = {"pre": pre, "post": post, "p": number(p), "weight": number(w),
                              "weight_max": number(wmax), "delay_ms": int(delay), "g": number(fast),
                              "plastic": plastic == "SYN_PLASTIC"}

    stp = r"setSTP\((\w+),\s*(\w+),\s*true,((?:\s*STP\w+\([^)]*\),?)+)\);"
    for m in re.finditer(stp, src):
        fields = dict(re.findall(r"(STP\w+)\(" + NUM + r",\s*[-0-9.]+f?\)", m.group(3)))
        link = links[(m.group(1), m.group(2))]
        excitatory = groups[m.group(1)]["excitatory"]
        link.update(U=number(fields["STPu"]), tau_u_ms=number(fields["STPtauU"]), tau_x_ms=number(fields["STPtauX"]),
                    tau_d_ms=number(fields["STPtdAMPA" if excitatory else "STPtdGABAa"]))

    assert all("U" in link for link in links.values()), "связь без setSTP"
    return groups, links


def main(project: str) -> None:
    root = Path(project)
    main_cpp = strip_comments(next((root / "src").glob("main_*.cpp")).read_text())

    groups, links = parse((root / "generateCONFIGStateSTP.h").read_text())

    # Фон покоя: std::lognormal_distribution<float> <тип>_distribution(μ, σ);
    # имя распределения → группа (как в main: pyr → PC и т.д.).
    lognormal = dict((n, (float(a), float(b))) for n, a, b in
                     re.findall(r"lognormal_distribution<float>\s+(\w+)_distribution\(([-0-9.]+),\s*([-0-9.]+)\)", main_cpp))
    names = {"pyr": "CA3_Pyramidal", "aac": "CA3_Axo_Axonic", "bc": "CA3_Basket", "cck": "CA3_BC_CCK",
             "bic": "CA3_Bistratified", "ivy": "CA3_Ivy", "mfao": "CA3_MFA_ORDEN", "quadD": "CA3_QuadD_LM"}
    for short, group in names.items():
        groups[group]["background_mu"], groups[group]["background_sigma"] = lognormal[short]

    out = {
        "source": SOURCE,
        "paper": "Kopsick et al. 2024, J Comput Neurosci, https://doi.org/10.1007/s10827-024-00881-3",
        "note": ("Izhikevich 2007 (C пФ, k нС/мВ, v мВ, a 1/мс, b нС, d пА); синапс — проводимость нС: "
                 "Δg = weight × g × u⁺x⁻/U на спайк, спад g ×(1 − 1/τ_d) за 1 мс; E_AMPA 0, E_GABAa −70 мВ; "
                 "фон каждую 1 мс: lognormal(μ, σ) × U(0, 1) пА, потолок 10 000; RK4 по 0.2 мс."),
        "background_cap_pa": 10000.0,
        "groups": groups,
        "links": list(links.values()),
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    cells = sum(g["cells"] for g in groups.values())
    print(f"{OUT}: групп {len(groups)}, клеток {cells}, связей {len(links)}")


if __name__ == "__main__":
    main(sys.argv[1])
