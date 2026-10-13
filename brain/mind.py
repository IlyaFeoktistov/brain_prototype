from __future__ import annotations

from pathlib import Path
import time

import torch

from brain.config import (
    FEATURES, BrainConfig, hippocampus_areas, hippocampus_pathways, learned_mouth, letter_mouth, sequence_areas,
    sequence_pathways,
)
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

        # Новый мозг с гиппокампом из нейронов: его зоны — часть анатомии
        # (сохраняются в cfg.areas, загруженный мозг их уже имеет).
        if cfg.neural_hippocampus and not any(a.name == "ca3" for a in cfg.areas):
            cfg.areas = cfg.areas + hippocampus_areas()
            cfg.pathways = cfg.pathways + hippocampus_pathways()

        if cfg.sequence_enabled and not any(a.name == "sequence" for a in cfg.areas):
            cfg.areas = cfg.areas + sequence_areas()
            cfg.pathways = cfg.pathways + sequence_pathways()

        if cfg.mouth_letters and not any(p.src == "auditory" and p.dst == "motor" for p in cfg.pathways):
            cfg.areas, cfg.pathways = letter_mouth(
                cfg.areas, cfg.pathways, cfg.articulator_cells, cfg.timing_cells, cfg.timing_link,
            )

        # Рот учится: только новому мозгу (у загруженного рот уже есть).
        if cfg.mouth_learns and not any(p.dst == "motor" and p.plastic for p in cfg.pathways):
            cfg.areas, cfg.pathways = learned_mouth(cfg.areas, cfg.pathways)

        self.brain = SpikingBrain(cfg)
        self.language = LanguageInterface(self.brain, cfg)

    def features(self) -> dict[str, bool]:
        return {name: bool(getattr(self.cfg, name)) for name in FEATURES}

    def set_feature(self, name: str, enabled: bool) -> None:
        if name not in FEATURES:
            raise KeyError(name)

        setattr(self.cfg, name, enabled)

        # Флаг читается при сборке шага — собрать заново.
        if FEATURES[name][1]:
            self.brain.recompile()

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
        self.brain.developing = True

        for _ in range(self.cfg.calibration_tokens):
            sound = "".join(rng.choice("абвгдежзиклмнопрстуфхцчшыэюя") for _ in range(6))
            # Звук будит своё место в смысловой коре, как слово: иначе
            # смысл молчит, и гомеостаз зон за ним (рот, память)
            # выкручивает возбудимость под вход, которого не будет.
            language.concepts.get(sound, allocate=True)
            language.memory.get(sound, allocate=True)
            language.perceive(sound, learn_forms=False)

        # Места калибровочных звуков — не знание: стереть.
        language.concepts.reset()
        language.memory.reset()

        # Калибровка — не опыт: веса и следы как при рождении.
        self.brain.net_weight[: self.brain.n_plastic].copy_(plastic_weights)
        self.brain.eligibility.zero_()
        self.brain.developing = False
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
