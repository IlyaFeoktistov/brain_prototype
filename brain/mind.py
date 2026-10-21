from __future__ import annotations

from pathlib import Path
import time

import torch

from brain import anatomy
from brain.config import BrainConfig, brain_path
from brain.engine import SpikingBrain


FORMAT_VERSION = 2


class Mind:
    """
    Мозг целиком: конфигурация + синапсы + состояние нейронов. Всё
    сохраняется в один файл, поэтому прогресс обучения не теряется между
    запусками. Мозг — анатомический (ARCHITECTURE 11): колонки коры, ядра и
    пути по данным; поведение — только из цепей, без решений кода.
    """

    def __init__(self, cfg: BrainConfig):
        self.cfg = cfg

        # Новый мозг: зоны и пути по данным (загруженный их уже имеет).
        if not any(a.name == "trn" for a in cfg.areas):
            anatomy.apply(cfg)

        self.brain = SpikingBrain(cfg)

    # ==============================================================
    # SAVE / LOAD
    # ==============================================================

    def save(self, path: str | None = None) -> Path:
        target = brain_path(path or self.cfg.save_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        tmp = target.with_suffix(target.suffix + ".tmp")

        torch.save(
            {
                "version": FORMAT_VERSION,
                "saved_at": time.time(),
                "config": self.cfg.to_dict(),
                "brain": self.brain.state_dict(),
            },
            tmp,
        )
        # Атомарно: старое сохранение не портится,
        # если процесс упадёт посреди записи.
        tmp.replace(target)

        return target

    @classmethod
    def load(cls, path: str, device: str | None = None) -> "Mind":
        path = brain_path(path)
        state = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        if state.get("version") != FORMAT_VERSION:
            raise ValueError(
                f"Неизвестная версия сохранения: {state.get('version')}"
            )

        cfg = BrainConfig.from_dict(state["config"])
        cfg.save_path = str(path)

        if device:
            cfg.device = device

        mind = cls(cfg)
        mind.brain.load_state_dict(state["brain"])

        return mind

    @classmethod
    def open(cls, cfg: BrainConfig) -> tuple["Mind", bool]:
        """Загрузить сохранённый мозг, если он есть, иначе создать новый."""

        if brain_path(cfg.save_path).exists():
            return cls.load(cfg.save_path, device=cfg.device), True

        return cls(cfg), False
