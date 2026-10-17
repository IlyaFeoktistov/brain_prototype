from __future__ import annotations

from dataclasses import replace

from pathlib import Path
import time

import torch

from brain.config import (
    FEATURES, BrainConfig, brain_path, eye_areas, eye_pathways, hand_areas, hand_pathways, hippocampus_areas, hippocampus_pathways, learned_mouth, letter_mouth, sequence_areas,
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

        if getattr(cfg, "anatomy", False):
            # Анатомический мозг: колонки коры и гиппокамп из нейронов; поведение
            # — только из цепей: без формул и решений кода (ARCHITECTURE 11).
            cfg.layers = True
            # Колонка крупнее — меньше доля «тока недостающих соседей» (в 5 тыс.
            # он ~94% входа и раскачивал зоны с сильным явным входом).
            cfg.min_column_neurons = max(cfg.min_column_neurons, 20000)
            cfg.neural_hippocampus = True
            cfg.neural_speech = True  # слово ответа — само выстрелившее место
            cfg.neural_surprise = True  # дофамин — из готовности нейронов
            cfg.emotions_enabled = False  # чувство-формула
            cfg.thinking_enabled = False  # проверяющий кодом
            cfg.live_mode = False  # таймеры
            cfg.initiative_enabled = False
            cfg.ask_about_unknown = False  # фраза-заготовка
            cfg.grow_on_surprise = False  # куда расти — решал код
            cfg.memory_places = False  # места слов в ПФК — кодом
            cfg.concepts_enabled = False  # места слов в смысловой коре — таблицей
            cfg.goal_enabled = False  # вопрос в «цель» кладёт и гасит код
            # Веса меняются сами (engine.chemical_plasticity); след для
            # дофамина живёт ~1 с: дофамин действует, если пришёл в 0.3–2 с
            # после пары спайков (Yagishita et al. 2014) — было 200 мс,
            # подобрано под чтение.
            cfg.eligibility_tau = 1000.0

        # Новый мозг с гиппокампом из нейронов: его зоны — часть анатомии
        # (сохраняются в cfg.areas, загруженный мозг их уже имеет).
        if cfg.neural_hippocampus and not any(a.name == "ca3" for a in cfg.areas):
            cfg.areas = cfg.areas + hippocampus_areas()
            cfg.pathways = cfg.pathways + hippocampus_pathways()

        if cfg.sequence_enabled and not any(a.name == "sequence" for a in cfg.areas):
            cfg.areas = cfg.areas + sequence_areas()
            cfg.pathways = cfg.pathways + sequence_pathways()

        if cfg.eye and not any(a.name == "visual" for a in cfg.areas):
            cfg.areas = cfg.areas + eye_areas()
            cfg.pathways = cfg.pathways + eye_pathways()

        if cfg.hand and not any(a.name == "hand" for a in cfg.areas):
            cfg.areas = cfg.areas + hand_areas()
            cfg.pathways = cfg.pathways + hand_pathways()

        if getattr(cfg, "anatomy", False) and not any(a.name == "trn" for a in cfg.areas):
            from brain import anatomy

            anatomy.apply(cfg)
            # Колонки-аттракторы цели (40 мВ на связь, подобраны «держать
            # вопрос») — костыль: рабочая память держится на рекуррентном
            # NMDA со скромными весами (Wang 2001); в колонке по данным они
            # взрывали зону (anatomy_bench: E 50–100 Гц, PV 400).
            cfg.areas = [replace(a, columns=0, column_weight=0.0) if a.name == "goal" else a for a in cfg.areas]
            # Пул торможения (k-WTA кодом) — его работу делают PV колонок и
            # тормозные клетки гранулярного слоя (Гольджи); «торможение» букв
            # путём E → E с минусом — заглушка: правило Vogels переходит на
            # настоящие синапсы PV → E (vogels_areas; D'amour & Froemke 2015).
            cfg.areas = [replace(a, pool_mv=0.0) for a in cfg.areas]
            cfg.pathways = [p for p in cfg.pathways if not getattr(p, "vogels", False) or p.src != p.dst]
            cfg.vogels_areas = ("letters",)
            # «Ворота говори» — VA/VL → смысловая кора — костыль: моторный
            # таламус в височную кору не идёт (её таламус — подушка и MD;
            # Jones 2007). Речь открывают базальные ганглии через VA/VL → Брока.
            cfg.pathways = [p for p in cfg.pathways if (p.src, p.dst) != ("thalamus", "association")]
            # «Рабочая память» особым типом клеток (LIF без адаптации +
            # медленный ток только в ПФК) — тоже костыль: у всех пирамидных
            # клеток коры есть NMDA (долг — NMDA у всех E колонок по данным).
            # ПФК и цель — обычные колонки.
            cfg.areas = [
                replace(a, kind="cortex", ampa_ratio=None, slow_max=None) if a.kind == "memory" and a.name in cfg.layered_areas else a
                for a in cfg.areas
            ]

        if cfg.layers and any(a.name == "complex" for a in cfg.areas):
            # Колонка V1 (ARCHITECTURE 10): сложные клетки — её слой 2/3, а не
            # отдельная зона; пути из «complex» идут из V1.
            complex_share = next(a.share for a in cfg.areas if a.name == "complex")
            cfg.areas = [
                replace(a, share=a.share + complex_share) if a.name == "visual" else a
                for a in cfg.areas if a.name != "complex"
            ]
            cfg.pathways = [
                replace(p, src="visual" if p.src == "complex" else p.src)
                for p in cfg.pathways if (p.src, p.dst) != ("visual", "complex")
            ]

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
        target = brain_path(path or self.cfg.save_path)
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
        mind.language.load_state_dict(state["language"])

        return mind

    @classmethod
    def open(cls, cfg: BrainConfig) -> tuple["Mind", bool]:
        """Загрузить сохранённый мозг, если он есть, иначе создать новый."""

        if brain_path(cfg.save_path).exists():
            return cls.load(cfg.save_path, device=cfg.device), True

        mind = cls(cfg)
        mind.calibrate()

        return mind, False
