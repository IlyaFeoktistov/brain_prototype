"""
Тело: органы мозга с общим разъёмом (ROADMAP чат-минимум 7.1).

Орган-датчик переводит сигнал мира в спайки своих рецепторов — свой
участок общего массива (engine.sensor_offsets); куда рецепторы входят в
кору, задаёт его SensorSpec. Новый орган — новый класс здесь и его
SensorSpec в BrainConfig.sensors().
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _gaussian(sigma: float) -> torch.Tensor:
    radius = max(1, int(round(3 * sigma)))
    x = torch.arange(-radius, radius + 1).float()
    g = torch.exp(-(x[:, None] ** 2 + x[None, :] ** 2) / (2 * sigma ** 2))
    return g / g.sum()


class Eye:
    """
    Глаз: картинка яркости → ганглиозные клетки сетчатки → растр спайков.

    Ганглиозная клетка сравнивает центр с окружением (Kuffler 1953): ON —
    центр светлее окружения, OFF — темнее. Ровный фон не даёт ничего,
    отвечают края и штрихи. Отклик — разность гауссиан (центр уже
    окружения); частота — доля от отклика на штрих толщиной в пиксель
    (тонкий тёмный штрих — самый сильный стимул OFF-клеток здесь).
    Спайки — пуассоновские, пока длится фиксация.
    """

    def __init__(self, cfg, offset: int, size: int, device):
        self.cfg = cfg
        self.offset = offset
        self.size = size
        self.device = device
        self.rows, self.cols = cfg.retina_rows, cfg.retina_cols
        self.center = _gaussian(cfg.retina_center_px)
        self.surround = _gaussian(cfg.retina_surround_px)
        line = torch.ones(15, 15)
        line[:, 7] = 0.0
        self.scale = float(self._contrast(line)[7, 7].neg()) or 1.0

    def _contrast(self, picture: torch.Tensor) -> torch.Tensor:
        """Центр минус окружение в каждой точке (фон за краем — как край)."""

        img = picture[None, None]

        def blur(kernel):
            pad = kernel.shape[-1] // 2
            return F.conv2d(F.pad(img, (pad, pad, pad, pad), mode="replicate"), kernel[None, None])

        return (blur(self.center) - blur(self.surround))[0, 0]

    def rates(self, picture: torch.Tensor) -> torch.Tensor:
        """Частота каждой ганглиозной клетки, Гц: слой ON, затем OFF."""

        contrast = self._contrast(picture.cpu()) / self.scale
        on = contrast.clamp(0.0, 1.0)
        off = (-contrast).clamp(0.0, 1.0)
        return torch.cat([on.flatten(), off.flatten()]) * self.cfg.retina_rate_hz

    def look(self, picture: torch.Tensor, ms: float | None = None) -> torch.Tensor:
        """Растр спайков фиксации [steps, offset + size]: участок глаза заполнен."""

        ms = self.cfg.fixation_ms if ms is None else ms
        steps = int(round(ms / self.cfg.dt_ms))
        p = (self.rates(picture) * self.cfg.dt_ms / 1000.0).to(self.device)
        raster = torch.zeros(steps, self.offset + self.size, device=self.device)
        raster[:, self.offset:] = (torch.rand(steps, self.size, device=self.device) < p).float()
        return raster


class Hand:
    """
    Рука (орган-исполнитель): у каждой клавиши своё врождённое движение —
    свой набор клеток моторной коры руки подряд (соматотопия пальцев,
    Penfield; движение — популяцией, Georgopoulos 1986). Это тело, а не
    знание: какая буква на клавише, мозг выучивает глазом. Нажата клавиша,
    чей набор выстрелил сильнее всех и хотя бы наполовину.
    """

    def __init__(self, keys: int, cells: int):
        self.keys = keys
        self.cells = cells

    def code(self, key: int) -> torch.Tensor:
        return torch.arange(key * self.cells, (key + 1) * self.cells)

    def pressed(self, fired: torch.Tensor, share: float = 0.5) -> int | None:
        """fired — спайки клеток руки за окно; номер нажатой клавиши или None."""

        blocks = (fired[: self.keys * self.cells] > 0).float().view(self.keys, self.cells).mean(1)
        key = int(blocks.argmax())
        return key if float(blocks[key]) >= share else None
