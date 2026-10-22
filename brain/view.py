"""
Куда смотрит глаз и куда тычет рука — для показа в вебе (только чтение,
мозгу не нужно). Обратные преобразования по проводке самого мозга:

    картинка   строка на экране вокруг точки взгляда (ямка и боковое поле)
    сетчатка   частоты ганглиозных клеток: ON − OFF в каждом пикселе
    LGN        спайки реле → пиксель своего рецептора (клетка i ← рецептор
               i % размер глаза, engine: реле таламуса)
    V1 L4      спайки простых клеток → пиксели их «+» подполей (входы
               lgn → visual.L4.E, _wire_oriented): ON-входы светлее, OFF —
               темнее, так видно, какой штрих клетка «увидела»
    рука       доля клеток движения каждой клавиши, выстреливших с прошлого
               показа (нажатие решает окно в 10 мс, Hand.pressed; в один
               снимок из 20 окон оно почти никогда не попадает)

Спайки копятся между показами (engine.on_chunk, ~20 Гц) и отдаются
снимком ~5 Гц, пока открыта хоть одна вкладка. Карты LGN и V1 — скользящая
сумма: за показ старое гаснет вдвое (память ~0.4 с) — за один снимок в
0.2 с фоновые спайки реле (~10 Гц) заслоняют слово.
"""

from __future__ import annotations

import torch


class Sight:
    def __init__(self, session):
        self.session = s = session
        brain, cfg = s.brain, s.cfg
        self.rows, self.cols = cfg.retina_rows, cfg.retina_cols
        pixels = self.rows * self.cols
        self.lgn = brain.areas["lgn"].E
        l4 = brain.areas["visual"].layers["L4"]["E"]
        self.l4 = l4
        # Рецептор каждой клетки реле: слой (0 — ON, 1 — OFF) и пиксель.
        receptor = torch.arange(self.lgn.size) % (2 * pixels)
        self.lgn_pixel, self.lgn_sign = receptor % pixels, 1.0 - 2.0 * (receptor // pixels).float()
        # Входы таламуса простых клеток L4: номера клеток реле → рецепторы.
        proj = next(p for p in brain.projections if p.name == f"lgn -> {l4.name}")
        pre = (proj.pre.long().cpu() - self.lgn.start) % (2 * pixels)
        self.l4_pixel, self.l4_sign = pre % pixels, 1.0 - 2.0 * (pre // pixels).float()
        self.hand = s.H
        self.keys = len(s.keyboard)
        self.lgn_map = torch.zeros(pixels)
        self.l4_map = torch.zeros(pixels)
        self.hand_seen = torch.zeros(self.keys * s.hand.cells, dtype=torch.bool)

    def add(self, fired: torch.Tensor) -> None:
        """Спайки с прошлого показа (номера клеток мозга) — в карты пикселей."""

        cells = fired[(fired >= self.lgn.start) & (fired < self.lgn.stop)] - self.lgn.start
        self.lgn_map.index_add_(0, self.lgn_pixel[cells], self.lgn_sign[cells])
        cells = fired[(fired >= self.l4.start) & (fired < self.l4.stop)] - self.l4.start
        self.l4_map.index_add_(0, self.l4_pixel[cells].flatten(), self.l4_sign[cells].flatten())
        cells = fired[(fired >= self.hand.start) & (fired < self.hand.start + len(self.hand_seen))] - self.hand.start
        self.hand_seen[cells] = True

    def snapshot(self) -> dict:
        s = self.session
        picture = s._picture(s.screen, s.cfg.fovea_col)
        side = s._picture(s.wide, s.field / 2)
        rates = s.eye.rates(picture)
        half = rates.numel() // 2
        retina = (rates[:half] - rates[half:]) / s.cfg.retina_rate_hz
        out = {
            "type": "view",
            "rows": self.rows,
            "cols": self.cols,
            "sideCols": side.shape[1],
            # Яркость 0–1 (1 — фон); сетчатка, LGN, V1 — со знаком: + светлое
            # (ON), − тёмное (OFF).
            "picture": _round(picture.flatten()),
            "side": _round(side.flatten()),
            "retina": _round(retina),
            "lgn": _round(self.lgn_map),
            "v1": _round(self.l4_map),
            "keys": _round(self.hand_seen.float().view(self.keys, -1).mean(1)),
            "held": s._held,
        }
        self.lgn_map *= 0.5
        self.l4_map *= 0.5
        self.hand_seen.zero_()
        return out


def _round(values: torch.Tensor) -> list[float]:
    return [round(v, 3) for v in values.tolist()]
