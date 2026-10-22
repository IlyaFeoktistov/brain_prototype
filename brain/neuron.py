"""
Инспектор клетки и синапса (веб, только чтение): всё, что есть в мозге о
нейроне — модель и параметры, состояние сейчас, вещества, фон и ток
недостающих соседей, ветки дендрита, входящие и исходящие синапсы по путям
и самые сильные поштучно; о синапсе — вес, врождённая и выученная части,
след STDP, третий фактор, ветка, задержка.

Вес синапса показывается и в мВ — пик ВПСП/ТПСП у клетки-получателя в покое
(вес × пик PSP единичного веса, engine._psp_peak); у проводимостей настоящий
ток зависит ещё от напряжения.
"""

from __future__ import annotations

from functools import lru_cache

import torch

from brain.engine import KINDS, SpikingBrain
from brain.wiring import ProceduralProjection


def _locate(brain: SpikingBrain, index: int):
    """Зона, метка популяции, популяция."""

    for area in brain.areas.values():
        for label, pop in area.populations:
            if pop.start <= index < pop.stop:
                return area, label, pop

    raise IndexError(f"нет клетки {index}")


def _pop_name(brain: SpikingBrain, index: int) -> str:
    area, label, _ = _locate(brain, index)
    return f"{area.name}.{label}"


@lru_cache(maxsize=4096)
def _peak(brain_id: int, start: int, size: int) -> float:
    brain = _BRAINS[brain_id]
    from brain.engine import Population
    return brain._psp_peak(Population("", start, size))


_BRAINS: dict[int, SpikingBrain] = {}


def _mv(brain: SpikingBrain, pop, weight: torch.Tensor) -> torch.Tensor:
    _BRAINS[id(brain)] = brain
    return weight.float() * _peak(id(brain), pop.start, pop.size)


def _path_index(brain: SpikingBrain, synapse: int) -> int:
    return int(torch.searchsorted(brain.path_start, torch.tensor([synapse], device=brain.device), right=True)[0]) - 1


def _synapse_info(brain: SpikingBrain, proj, row: int, col: int) -> dict:
    """Один синапс сети: всё, что о нём хранится."""

    sid = proj.offset + row * proj.pre.shape[1] + col
    pre = int(proj.pre[row, col])
    post = proj.post.start + row
    w = proj.weight[row, col]
    out = {
        "id": sid,
        "path": proj.name,
        "pre": pre,
        "preName": _pop_name(brain, pre),
        "post": post,
        "postName": _pop_name(brain, post),
        "weight": float(w),
        "psp": float(_mv(brain, proj.post, w)),
        "plastic": bool(proj.plastic),
        "modulator": proj.modulator if proj.plastic else None,
        "achSuppressed": bool(getattr(proj, "ach_suppressed", False)),
        "dendrite": bool(getattr(proj, "dendrite", False)),
        "branch": int(proj.branch[row, col]) if hasattr(proj, "branch") else None,
    }

    if brain.path_delay is not None and brain.ring_slots > 1:
        out["delayMs"] = (int(brain.path_delay[_path_index(brain, sid)]) + 1) * brain.cfg.dt_ms
    else:
        out["delayMs"] = brain.cfg.dt_ms

    if proj.plastic:
        birth = proj.birth[row, col].float()
        out.update(
            birthPsp=float(_mv(brain, proj.post, birth)),
            learnedPsp=float(_mv(brain, proj.post, w - birth)),
            earlyPsp=float(_mv(brain, proj.post, proj.early[row, col].float())),
            eligibility=float(proj.eligibility[row, col].float() * brain.eligibility_scale),
            capPsp=float(_mv(brain, proj.post, torch.tensor(proj.cap))),
        )

    return out


def _group(brain, proj, rows, cols, by_post: bool, source: str | None = None) -> dict:
    """Сводка синапсов одного пути: число, сила, выученное, след."""

    w = proj.weight[rows, cols].float()
    psp = _mv(brain, proj.post, w)
    out = {
        "path": proj.name,
        "population": source,
        "count": int(len(w)),
        "pspMean": float(psp.mean()) if len(w) else 0.0,
        "pspMin": float(psp.min()) if len(w) else 0.0,
        "pspMax": float(psp.max()) if len(w) else 0.0,
        "plastic": bool(proj.plastic),
        "modulator": proj.modulator if proj.plastic else None,
        "dendrite": bool(getattr(proj, "dendrite", False)),
        "achSuppressed": bool(getattr(proj, "ach_suppressed", False)),
    }

    if proj.plastic and len(w):
        birth = proj.birth[rows, cols].float()
        change = w - birth
        out.update(
            learnedPsp=float(_mv(brain, proj.post, change).mean()),
            stronger=int((change > 1e-6).sum()),
            weaker=int((change < -1e-6).sum()),
            eligibility=float(proj.eligibility[rows, cols].float().abs().mean() * brain.eligibility_scale),
        )

    return out


def describe(brain: SpikingBrain, index: int, top: int = 24) -> dict:
    cfg = brain.cfg
    area, label, pop = _locate(brain, index)
    kind = label.split(".")[-1]
    kind = "PV" if kind == "I" and area.layers is not None else kind
    i = index
    gap = cfg.v_threshold - cfg.v_rest
    adex = bool(brain.is_adex[i])
    state = {
        "v": float(brain.v[i]),
        "iSyn": float(brain.i_syn[i]),
        "iInh": float(brain.i_inh[i]),
        "iNgf": float(brain.i_ngf[i]),
        "iNmda": float(brain.i_nmda[i]) if brain.nmda_on else None,
        "adapt": float(brain.adapt[i]),
        "refractory": max(0.0, float(brain.refractory[i])),
        "bias": float(brain.bias[i]),
        "tracePre": float(brain.trace_pre[i]),
        "tracePost": float(brain.trace_post[i]),
        "resource": float(brain.stp_x[i]),
        "inhGain": float(brain.inh_gain[i]),
        "tGate": float(brain.t_h[i]) if bool(brain.t_mask[i]) else None,
        "spikesInWindow": float(brain.counts[i]),
    }
    model = {
        "type": "AdEx" if adex else "LIF",
        "tauM": float(brain.tau_m[i]),
        "tauSyn": float(brain.tau_syn[i]),
        "deltaT": float(brain.delta_t[i]),
        "adaptB": float(brain.adapt_b[i]),
        "vRest": cfg.v_rest,
        "vThreshold": cfg.v_threshold,
        "vReset": cfg.v_reset,
        # Ток покоя сдвигает клетку к порогу: до порога столько мВ.
        "gapMv": gap - float(brain.bias[i]),
        "refractoryMs": cfg.refractory_ms,
        "conductance": True,
        "tCurrent": bool(brain.t_mask[i]),
    }
    # Кратковременная пластичность исходящих синапсов (клетки колонок — к
    # каждому типу получателя).
    if bool(brain.typed_pre[i]):
        model["stp"] = [
            {"to": k, "U": float(brain.stp_U4[i, j]), "u": float(brain.stp_u4[i, j]), "x": float(brain.stp_x4[i, j])}
            for j, k in enumerate(KINDS) if float(brain.stp_rec4[i, j]) > 0
        ]

    # Вещества: занятость рецепторов × ток клетки при полной занятости.
    occupancy = (brain.chemistry / (brain.chemistry + brain.chemistry_half)).float()
    receptors = [
        {"substance": name, "occupancy": float(occupancy[j]), "mvFull": float(brain.receptor_mv[j, i]),
         "mvNow": float(occupancy[j] * brain.receptor_mv[j, i])}
        for j, name in enumerate(brain.chemistry_names) if float(brain.receptor_mv[j, i]) != 0.0
    ]
    tau = float(brain.tau_m[i])
    background = {
        "inputsPerStep": float(brain.noise_mean[i]),
        "gain": float(brain.noise_gain[i]),
        # Средний вклад фона, мВ: входов за мс × вес × τm.
        "mvMean": float(brain.noise_mean[i] / cfg.dt_ms * brain.noise_gain[i] * tau),
    }
    # Ток недостающих соседей (среднее поле) по источникам, мВ сейчас.
    mean_field = []

    for post, pre, coeff, rate, ach, slow in brain.mean_field:
        if post.start <= i < post.stop:
            q = int(brain.mf_pre_id[pre.start])
            now = float(brain.mf_rate[q]) if q < len(brain.mf_rate) else rate
            mean_field.append({"source": pre.name, "mv": coeff * now, "slow": bool(slow), "ach": bool(ach)})

    mean_field.sort(key=lambda m: -abs(m["mv"]))
    branches = max(1, cfg.dendrite_branches)
    current = brain.i_dend.view(brain.n_slots, branches)[i]
    plateau = torch.sigmoid((current - cfg.dendrite_threshold_mv) / cfg.dendrite_slope_mv)
    dendrite = {
        "branches": [{"current": float(c), "plateau": float(p)} for c, p in zip(current, plateau)],
        "thresholdMv": cfg.dendrite_threshold_mv,
        "slopeMv": cfg.dendrite_slope_mv,
        "gain": cfg.dendrite_gain,
        "somaMv": float(brain.dendrite_soma[i]),
    }
    incoming, outgoing, top_in, top_out = [], [], [], []

    for proj in brain.projections:
        if isinstance(proj, ProceduralProjection):
            if proj.post.start <= i < proj.post.stop:
                pre = proj.incoming(i - proj.post.start)
                incoming.append({"path": proj.name, "population": proj.pre_pop.name, "count": int(len(pre)),
                                 "pspMean": float(_mv(brain, proj.post, torch.tensor(proj.value))),
                                 "plastic": False, "procedural": True})
            continue

        if proj.source == "sensory":
            if proj.post.start <= i < proj.post.stop:
                row = i - proj.post.start
                w = proj.weight[row]
                psp = _mv(brain, proj.post, w)
                incoming.append({"path": proj.name, "population": "рецепторы", "count": int(len(w)),
                                 "pspMean": float(psp.mean()), "pspMin": float(psp.min()), "pspMax": float(psp.max()),
                                 "plastic": False, "sensory": True})
            continue

        if proj.post.start <= i < proj.post.stop:
            row = i - proj.post.start
            k = proj.pre.shape[1]
            rows = torch.full((k,), row, device=brain.device)
            cols = torch.arange(k, device=brain.device)
            group = _group(brain, proj, rows, cols, True, getattr(proj, "pre_pop", None) and proj.pre_pop.name)
            incoming.append(group)
            strongest = torch.argsort(proj.weight[row].float().abs(), descending=True)[:top]
            top_in += [_synapse_info(brain, proj, row, int(c)) for c in strongest]

        hit = torch.nonzero(proj.pre == i)

        if len(hit):
            rows, cols = hit[:, 0], hit[:, 1]
            outgoing.append(_group(brain, proj, rows, cols, False, proj.post.name))
            strongest = torch.argsort(proj.weight[rows, cols].float().abs(), descending=True)[:top]
            top_out += [_synapse_info(brain, proj, int(rows[s]), int(cols[s])) for s in strongest]

    top_in.sort(key=lambda s: -abs(s["psp"]))
    top_out.sort(key=lambda s: -abs(s["psp"]))

    return {
        "index": index,
        "area": area.name,
        "areaTitle": area.title,
        "population": f"{area.name}.{label}",
        "layer": label.split(".")[0] if "." in label else ("L23" if area.layers is not None else None),
        "kind": kind,
        "inhibitory": kind != "E",
        "local": index - pop.start,
        "model": model,
        "state": state,
        "receptors": receptors,
        "background": background,
        "meanField": mean_field[:16],
        "dendrite": dendrite,
        "incoming": sorted(incoming, key=lambda g: -g["count"]),
        "outgoing": sorted(outgoing, key=lambda g: -g["count"]),
        "topIncoming": top_in[:top],
        "topOutgoing": top_out[:top],
    }


def synapse(brain: SpikingBrain, sid: int) -> dict:
    """Синапс сети по номеру."""

    for proj in brain.projections:
        if proj.source != "net" or isinstance(proj, ProceduralProjection):
            continue

        n = proj.weight.numel()

        if proj.offset <= sid < proj.offset + n:
            row, col = divmod(sid - proj.offset, proj.pre.shape[1])
            return _synapse_info(brain, proj, row, col)

    raise IndexError(f"нет синапса {sid}")


def summary(brain: SpikingBrain) -> dict:
    """Сводка по зонам: клеток, частота за последнее окно, синапсов, выученного."""

    window = max(brain.cfg.dt_ms, 1.0)
    areas = []

    for area in brain.areas.values():
        cells = area.cells
        areas.append({"name": area.name, "title": area.title, "cells": cells.stop - cells.start,
                      "spikes": float(brain.counts[cells].sum())})

    paths = []

    for p in brain.plastic:
        change = p.weight.float() - p.birth.float()
        # Зоны-концы пути (схема путей в вебе сводит проекции к паре зон).
        src = _locate(brain, int(p.pre.flatten()[0]))[0].name
        paths.append({"path": p.name, "src": src, "dst": _locate(brain, p.post.start)[0].name,
                      "synapses": p.weight.numel(), "modulator": p.modulator,
                      "stronger": int((change > 1e-6).sum()), "weaker": int((change < -1e-6).sum()),
                      "learnedPsp": float(_mv(brain, p.post, change.abs().mean()))})

    return {"neurons": brain.n, "synapses": brain.synapse_count(), "plastic": brain.n_plastic,
            "timeMs": brain.time_ms, "areas": areas, "paths": paths, "windowMs": window}
