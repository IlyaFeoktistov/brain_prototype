from __future__ import annotations

from pathlib import Path
import time

import json

import torch

from brain import anatomy
from brain.config import BrainConfig, brain_path
from brain.engine import SpikingBrain


FORMAT_VERSION = 2


def meta(path: Path) -> Path:
    """Файл описания мозга: рядом с файлом мозга, «имя.pt.json»."""

    return Path(path).with_suffix(Path(path).suffix + ".json")


def describe_saved(path: Path) -> dict:
    """Что известно о сохранённом мозге без его загрузки; old — другой формат."""

    path = Path(path)
    info = {"name": path.name, "megabytes": round(path.stat().st_size / 2**20), "modified": path.stat().st_mtime}

    try:
        data = json.loads(meta(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {**info, "old": True}

    return {**info, **data, "old": data.get("version") != FORMAT_VERSION}


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
        # Описание рядом (формат, размер) — список мозгов не читает гигабайты.
        meta(target).write_text(json.dumps({
            "version": FORMAT_VERSION, "size": self.cfg.size, "neurons": self.brain.n,
            "synapses": self.brain.synapse_count(), "timeMs": self.brain.time_ms, "saved": time.time(),
        }, ensure_ascii=False), encoding="utf-8")

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

