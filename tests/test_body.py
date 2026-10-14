"""Тело: органы-датчики через общий разъём — глаз видит штрихи."""

import math

import torch

from brain.body import Eye
from brain.config import BrainConfig
from brain.mind import Mind


def test_eye_v1_prefers_bar_orientation():
    # Новый мозг small с глазом (на tiny V1 мала для поля 16 × 96): полоса
    # будит прежде всего клетки своей ориентации, пустой экран — никого.
    cfg = BrainConfig()
    cfg.size = "small"
    cfg.eye = True
    cfg.autosave = False
    mind = Mind(cfg)
    brain = mind.brain
    i = [s.name for s in brain.sensors].index("eye")
    eye = Eye(cfg, brain.sensor_offsets[i], brain.sensors[i].size, brain.device)
    V = brain.areas["visual"].E
    rows, cols = cfg.retina_rows, cfg.retina_cols
    rr, cc = torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing="ij")
    rr, cc = rr.float() - rows / 2, cc.float() - cols / 2

    def v1(picture):
        with brain.frozen():
            brain.run(100.0)
            return brain.observe(cfg.fixation_ms, eye.look(picture))[V.slice]

    assert float(v1(torch.ones(rows, cols)).sum()) == 0.0

    for o, angle in enumerate((0, 45, 90, 135)):
        a = math.radians(angle)
        picture = torch.ones(rows, cols)
        picture[(rr * math.cos(a) + cc * math.sin(a)).abs() < 0.6] = 0.0
        rates = v1(picture)
        by = torch.stack([rates[k::4].sum() for k in range(4)])
        assert int(by.argmax()) == o, (angle, by.tolist())
