from __future__ import annotations

from pathlib import Path
import time

import torch

from brain.config import BrainConfig
from brain.engine import SpikingBrain
from brain.language import LanguageInterface


FORMAT_VERSION = 1


class Mind:
    """
    Мозг целиком: конфигурация + синапсы + состояние нейронов
    + словарь форм слов. Всё сохраняется в один файл,
    поэтому прогресс обучения не теряется между запусками.
    """

    def __init__(self, cfg: BrainConfig):
        self.cfg = cfg
        self.brain = SpikingBrain(cfg)
        self.language = LanguageInterface(self.brain, cfg)

    def calibrate(self) -> None:
        """
        «Пренатальная» калибровка: мозг слышит случайные звуки,
        синапсы не учатся, гомеостаз находит рабочую точку
        возбудимости каждой зоны.
        """

        import random

        rng = random.Random(self.cfg.seed)
        language = self.language
        plastic_weights = self.brain.net_weight[: self.brain.n_plastic].clone()

        for _ in range(self.cfg.calibration_tokens):
            sound = "".join(rng.choice("абвгдежзиклмнопрстуфхцчшыэюя") for _ in range(6))
            language.perceive(sound, learn_forms=False)

        # Калибровка — не опыт: веса и следы как при рождении.
        self.brain.net_weight[: self.brain.n_plastic].copy_(plastic_weights)
        self.brain.eligibility.zero_()
        self.brain.run(self.cfg.rest_ms)

    # ==============================================================
    # SAVE / LOAD
    # ==============================================================

    def save(self, path: str | None = None) -> Path:
        target = Path(path or self.cfg.save_path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)

        tmp = target.with_suffix(target.suffix + ".tmp")

        torch.save(
            {
                "version": FORMAT_VERSION,
                "saved_at": time.time(),
                "config": self.cfg.to_dict(),
                "brain": self.brain.state_dict(),
                "language": self.language.state_dict(),
            },
            tmp,
        )
        # Атомарно: старое сохранение не портится,
        # если процесс упадёт посреди записи.
        tmp.replace(target)

        return target

    @classmethod
    def load(cls, path: str, device: str | None = None) -> "Mind":
        state = torch.load(
            Path(path).expanduser(),
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
        mind.language.load_state_dict(state["language"])

        return mind

    @classmethod
    def open(cls, cfg: BrainConfig) -> tuple["Mind", bool]:
        """Загрузить сохранённый мозг, если он есть, иначе создать новый."""

        if Path(cfg.save_path).expanduser().exists():
            return cls.load(cfg.save_path, device=cfg.device), True

        mind = cls(cfg)
        mind.calibrate()

        return mind, False
