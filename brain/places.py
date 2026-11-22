"""
Места клеток для карты нейронов в вебе (web/src/NeuronMap.tsx): где клетка
лежит в своей зоне, только если место задано анатомией модели — не
придуманное.

  - лист коры (engine.sheet_pos — места клеток колонки на квадрате листа, мм;
    по ним строятся местные связи по расстоянию): каждый слой — своя плитка
    (на листе все слои на одном квадрате);
  - зубчатая (сеть с топографией, brain/circuit.py): место по длине зоны
    (100 мест по 60 мкм, Morgan & Soltesz 2008), тип клеток — своя плитка;
  - остальное (ядра, CA3, CA1) — места нет: карта кладёт клетки плотной
    группой по номеру и подписывает «без места».

Ответ по зоне: mode (sheet | line | none), tiles (подписи плиток), tile —
плитка клетки (uint8), xy — место в плитке в долях 0…1 (uint16 × 2), base64.
"""

from __future__ import annotations

import base64

import numpy as np


def _b64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode()


def places(brain) -> dict:
    sheet = getattr(brain, "sheet_pos", {})
    circuits = getattr(brain, "circuits", {})
    out = []

    for name, area in brain.areas.items():
        size = area.cells.stop - area.cells.start
        tile = np.zeros(size, dtype=np.uint8)
        xy = np.zeros((size, 2), dtype=np.float64)
        pops = [(label, p) for label, p in area.populations]

        if area.layers is not None and all(p.name in sheet for _, p in pops):
            layers = []
            side = max(float(sheet[p.name].max()) for _, p in pops) or 1.0

            for label, p in pops:
                layer = label.split(".")[0] if "." in label else "L23"
                if layer not in layers:
                    layers.append(layer)
                local = slice(p.start - area.cells.start, p.stop - area.cells.start)
                tile[local] = layers.index(layer)
                xy[local] = sheet[p.name].detach().cpu().numpy() / side
            mode, tiles = "sheet", [{"L23": "L2/3"}.get(x, x) for x in layers]
        elif name in circuits and getattr(circuits[name], "topo", False):
            c = circuits[name]
            tiles = list(c.names)
            mode = "line"
            offset = 0

            for k, g in enumerate(c.names):
                bins = c._bin_of[g].cpu().numpy()
                n = len(bins)
                rank = np.zeros(n)
                count = np.bincount(bins, minlength=c._bins)
                seen = np.zeros(c._bins, dtype=np.int64)
                for i, b in enumerate(bins):
                    rank[i] = seen[b]
                    seen[b] += 1
                # Клетки одного места — столбиком на своём месте по длине.
                per = np.maximum(count[bins], 1)
                cols = np.ceil(np.sqrt(per / 4.0))
                col = rank % cols
                row = rank // cols
                xy[offset: offset + n, 0] = (bins + (col + 0.5) / cols) / c._bins
                xy[offset: offset + n, 1] = (row + 0.5) / np.ceil(per / cols)
                tile[offset: offset + n] = k
                offset += n
        else:
            mode, tiles = "none", []

        out.append({
            "name": name,
            "mode": mode,
            "tiles": tiles,
            "tile": _b64(tile),
            "xy": _b64(np.clip(xy * 65535, 0, 65535).astype(np.uint16)),
        })

    return {"areas": out}
