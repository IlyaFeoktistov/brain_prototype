"""
Пресеты мозга: готовые сборки, одни на всех — веб, опыты, тесты.

«рабочий» — мозг, с которым идут опыты (DESIGN 2026-11-12…14): фон LGN по
подгонке lgn-a, порог реле −59 мВ, TRN → LGN ×0.1, сдвиги гомеостаза после
развития (lgn-d-pop-tcind, своё зерно), учёба с учителем на пути V1 → буквы
и выученные веса этого пути (letters_us, у ямки 0.58 на 10 буквах).
Подгонка и гомеостаз посчитаны на мозге размера small (наименьший), поэтому
размер пресета фиксирован.

Файлы пресета ищутся сначала в ~/.brain (BRAIN_HOME — там их пишут опыты),
потом в репозитории data/presets (копия для зерна 1 — чтобы веб из свежего
клона создавал тот же мозг, что у нас; ~13 МБ).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from brain import anatomy
from brain.config import BRAIN_HOME, BrainConfig
from brain.mind import Mind

FITS = BRAIN_HOME / "fits"
REPO_PRESETS = Path(__file__).resolve().parent.parent / "data" / "presets"


def preset_file(rel: str) -> Path:
    """Файл пресета по пути относительно BRAIN_HOME: из ~/.brain, иначе копия в data/presets."""

    home = BRAIN_HOME / rel
    return home if home.exists() or not (REPO_PRESETS / rel).exists() else REPO_PRESETS / rel
TRN_TO_LGN = "trn -> lgn"
LETTERS_PATH = "visual -> letters"


@dataclass(frozen=True)
class Preset:
    """Сборка мозга: конфиг до рождения и правки после (веса, гомеостаз)."""

    name: str
    title: str
    size: str = "small"
    fit: str | None = None                      # фон: ~/.brain/fits/<fit>.json (anatomy.use_fit)
    homeo: str | None = None                    # сдвиги гомеостаза: fits/<homeo>-s<seed>-homeo.pt
    relay_v_t_mv: float | None = None           # порог реле таламуса
    trn_scale: float = 1.0                      # множитель весов TRN → LGN
    us_paths: tuple[str, ...] = ()              # пути с учёбой с учителем (cfg.us_paths)
    weights: dict[str, str] = field(default_factory=dict)   # путь → файл весов ~/.brain/<шаблон с {seed}>

    def config(self, seed: int = 1) -> BrainConfig:
        cfg = BrainConfig()
        cfg.size = self.size
        cfg.seed = seed
        if self.relay_v_t_mv is not None:
            cfg.relay_v_t_mv = self.relay_v_t_mv
        if self.fit:
            anatomy.use_fit(cfg, self.fit)
        if self.us_paths:
            cfg.us_paths = self.us_paths
        return cfg

    def missing(self, seed: int = 1) -> list[Path]:
        """Файлы пресета, которых нет на диске."""

        need = [preset_file(f.format(seed=seed)) for f in self.weights.values()]
        if self.fit:
            need.append(preset_file(f"fits/{self.fit}.json"))
        if self.homeo:
            need.append(self.homeo_file(seed))
        return [p for p in need if not p.exists()]

    def homeo_file(self, seed: int) -> Path:
        return preset_file(f"fits/{self.homeo}-s{seed}-homeo.pt")

    def finish(self, mind: Mind, seed: int = 1) -> None:
        """Правки после сборки: TRN, гомеостаз, выученные веса."""

        b = mind.brain
        if self.trn_scale != 1.0:
            next(p for p in b.projections if p.name == TRN_TO_LGN).weight.mul_(self.trn_scale)
        if self.homeo:
            b.bias[: b.n] += torch.load(self.homeo_file(seed), map_location=b.device)
        for path, f in self.weights.items():
            w = torch.load(preset_file(f.format(seed=seed)), map_location=b.device)["weight"]
            next(p for p in b.projections if p.name == path).weight.copy_(w)

    def build(self, seed: int = 1, cfg: BrainConfig | None = None) -> Mind:
        """Собрать мозг пресета (cfg — уже настроенный self.config(seed) с правками вызывающего)."""

        lost = self.missing(seed)
        if lost:
            raise FileNotFoundError("нет файлов пресета «%s»: %s" % (self.name, ", ".join(map(str, lost))))
        mind = Mind(cfg or self.config(seed))
        self.finish(mind, seed)
        return mind


PRESETS: dict[str, Preset] = {
    p.name: p
    for p in (
        Preset(
            "working", "рабочий мозг: подогнанный зрительный путь, учит буквы",
            fit="lgn-a", homeo="lgn-d-pop-tcind", relay_v_t_mv=-59.0, trn_scale=0.1,
            us_paths=(LETTERS_PATH,), weights={LETTERS_PATH: "letters_us_phi60_s{seed}.pt"},
        ),
    )
}
WORKING = PRESETS["working"]
