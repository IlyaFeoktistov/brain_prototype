from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import random
import re
import time
from typing import Callable

import torch

from brain.concepts import Concepts
from brain.engine import SpikingBrain
from brain.hippocampus import Hippocampus
from brain.limbic import FACES, Emotions


# Конец фразы — короткий звук паузы со своим местом в смысле: мозг
# учит, после чего мысль кончается, как любой другой переход; в ответе
# победил «конец» — замолкает. Без правил языка: это просто пауза.
END = "."


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    na = a.norm()
    nb = b.norm()

    if not na or not nb:
        return 0.0

    return float((a @ b) / (na * nb))


# ==================================================================
# УХО: текст -> sensory spike pattern
# ==================================================================


class Ear:
    """
    Токен -> растр sensory-спайков [steps, sensory_size].

    Одинаковый токен всегда даёт один и тот же набор каналов
    и одинаковые времена спайков (одинаковый звук ->
    одинаковая активация улитки). Какие нейроны коры
    ответят — решает сама сеть.
    """

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.cache: dict[str, torch.Tensor] = {}

    def encode(self, token: str) -> torch.Tensor:
        channels, pattern = self._code(token)

        raster = torch.zeros(
            pattern.shape[0], self.cfg.sensory_size, device=self.device
        )
        # Сложение, а не присваивание: повторная буква повторяет каналы.
        raster.index_add_(1, channels, pattern).clamp_(max=1.0)

        return raster

    def _code(self, token: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Компактный код звучания: 32 канала + узор спайков [steps, 32].
        Полный растр [steps, 2048] весит 0.5 МБ — на словаре в десятки
        тысяч слов кэш полных растров не влезал в видеокарту.
        """

        code = self.cache.get(token)

        if code is not None:
            return code

        cfg = self.cfg
        steps = int(round(cfg.stimulus_ms / cfg.dt_ms))
        p = cfg.sensory_rate_hz * cfg.dt_ms / 1000.0

        if cfg.ear_letters:
            channels, pattern = self._letters(token, steps, p)
        else:
            rng = torch.Generator().manual_seed(self._seed(token))
            channels = torch.randperm(cfg.sensory_size, generator=rng)[
                : cfg.sensory_channels_per_token
            ]
            pattern = (torch.rand(steps, len(channels), generator=rng) < p).float()

        code = (channels.to(self.device), pattern.to(self.device))

        if len(self.cache) > 200_000:
            self.cache.clear()

        self.cache[token] = code

        return code


    @staticmethod
    def _seed(text: str) -> int:
        return int.from_bytes(
            hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(),
            "little",
        ) & 0x7FFFFFFFFFFFFFFF

    def _letters(self, token: str, steps: int, p: float) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Слово звучит по буквам: время слова делится поровну между
        символами, каждый символ — свои каналы и свой узор спайков, где бы
        он ни стоял. Похожие формы («франция», «франции») звучат похоже
        сами, на любом языке; узнавать их мозг учится сам.
        """

        cfg = self.cfg
        per_char = cfg.sensory_channels_per_token
        chars = list(token) or [" "]
        bounds = [round(i * steps / len(chars)) for i in range(len(chars) + 1)]
        channels: list[torch.Tensor] = []
        pattern = torch.zeros(steps, per_char * len(chars))

        for i, char in enumerate(chars):
            rng = torch.Generator().manual_seed(self._seed("символ:" + char))
            channels.append(torch.randperm(cfg.sensory_size, generator=rng)[:per_char])
            spikes = (torch.rand(steps, per_char, generator=rng) < p).float()
            start, stop = bounds[i], bounds[i + 1]
            pattern[start:stop, i * per_char: (i + 1) * per_char] = spikes[: stop - start]

        return torch.cat(channels), pattern


# ==================================================================
# READOUT: состояние зоны -> токен
# ==================================================================


class Readout:
    """
    «Какому известному слову соответствует это состояние зоны?»

    Для каждого токена хранится усреднённый прототип состояния
    (top-k самых активных нейронов). Это словарь форм,
    а НЕ память: переходы между словами здесь не хранятся.
    """

    def __init__(self, size: int, top_k: int, rate: float, device):
        self.size = size
        self.top_k = top_k
        self.rate = rate
        self.device = device

        self.tokens: list[str] = []
        self.index: dict[str, int] = {}
        self.observations: list[int] = []

        self.idx = torch.zeros(0, top_k, dtype=torch.long, device=device)
        self.val = torch.zeros(0, top_k, device=device)

    def __contains__(self, token: str) -> bool:
        return token in self.index

    def __len__(self) -> int:
        return len(self.tokens)

    def _grow(self) -> None:
        capacity = max(64, self.idx.shape[0] * 2)
        idx = torch.zeros(capacity, self.top_k, dtype=torch.long, device=self.device)
        val = torch.zeros(capacity, self.top_k, device=self.device)
        idx[: len(self.tokens)] = self.idx[: len(self.tokens)]
        val[: len(self.tokens)] = self.val[: len(self.tokens)]
        self.idx, self.val = idx, val

    def observe(self, token: str, state: torch.Tensor) -> None:
        if not bool(state.any()):
            return

        row = self.index.get(token)
        dense = torch.zeros(self.size, device=self.device)

        if row is None:
            if len(self.tokens) == self.idx.shape[0]:
                self._grow()

            row = len(self.tokens)
            self.tokens.append(token)
            self.index[token] = row
            self.observations.append(0)
            dense += state
        else:
            dense[self.idx[row]] = self.val[row] * (1.0 - self.rate)
            dense += self.rate * state

        val, idx = dense.topk(min(self.top_k, self.size))
        self.idx[row, : len(idx)] = idx
        self.val[row, : len(val)] = val
        self.observations[row] += 1

    def prototype(self, token: str) -> torch.Tensor:
        row = self.index[token]
        dense = torch.zeros(self.size, device=self.device)
        dense[self.idx[row]] = self.val[row]
        return dense

    def decode(
        self,
        state: torch.Tensor,
        exclude: set[str] | None = None,
        top: int = 5,
    ) -> list[tuple[str, float]]:
        n = len(self.tokens)

        if not n or not bool(state.any()):
            return []

        idx, val = self.idx[:n], self.val[:n]
        dot = (state[idx] * val).sum(1)
        norm = val.norm(dim=1) * state.norm()
        scores = dot / norm.clamp(min=1e-9)

        for token in exclude or ():
            row = self.index.get(token)

            if row is not None:
                scores[row] = -1.0

        values, rows = scores.topk(min(top, n))

        return [
            (self.tokens[r], float(v))
            for r, v in zip(rows.tolist(), values.tolist())
            if v > 0
        ]

    def assembly(self, token: str, count: int) -> torch.Tensor:
        row = self.index[token]
        return self.idx[row, :count]

    def state_dict(self) -> dict:
        n = len(self.tokens)
        return {
            "tokens": self.tokens,
            "observations": self.observations,
            "idx": self.idx[:n].cpu(),
            "val": self.val[:n].cpu(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.tokens = list(state["tokens"])
        self.observations = list(state["observations"])
        self.index = {t: i for i, t in enumerate(self.tokens)}
        self.idx = state["idx"].to(self.device)
        self.val = state["val"].to(self.device)


# ==================================================================
# УВЕРЕННОСТЬ (метапознание)
# ==================================================================


class Confidence:
    """
    Насколько мозг уверен в своём ожидании — выучено из опыта.

    Резкость ожидания (margin) = во сколько раз лучший кандидат
    сильнее среднего из 20 лучших. Сама по себе это просто число.
    Пока мозг читает, он каждый раз видит, сбылось ли ожидание,
    и копит статистику «при такой резкости я прав в X% случаев».
    Эту долю он и называет своей уверенностью: хорошо выученное
    («столица германии -> берлин») идёт с высокой, слабо
    выученное — с низкой.
    """

    # [замер] с местами слов ожидания стали в разы резче (8–19), и при
    # старой шкале до 3.0 90% проверок падали в одну корзину —
    # уверенность у всех слов была одна (0.67). Шкала продлена до 17
    # (резкость не больше 20: лучший из 20 при прочих нулях).
    EDGES = (1.1, 1.2, 1.3, 1.5, 1.8, 2.2, 3.0, 4.5, 6.5, 9.0, 12.0, 14.5, 17.0)
    MIN_SAMPLES = 20

    def __init__(self):
        self.hits = [0] * (len(self.EDGES) + 1)
        self.total = [0] * (len(self.EDGES) + 1)

    @staticmethod
    def margin(ranked: list[tuple[str, float]]) -> float:
        if not ranked:
            return 0.0

        mean = sum(score for _, score in ranked) / len(ranked)

        return ranked[0][1] / mean if mean > 0 else 0.0

    def _bin(self, margin: float) -> int:
        return sum(margin >= edge for edge in self.EDGES)

    def update(self, margin: float, hit: bool) -> None:
        index = self._bin(margin)
        self.total[index] += 1
        self.hits[index] += hit

    @property
    def experience(self) -> int:
        return sum(self.total)

    def __call__(self, margin: float) -> float:
        """Доля сбывшихся ожиданий при такой резкости (0..1)."""

        # Сглаживание (Лаплас) + монотонность: более резкое
        # ожидание не может быть менее надёжным, чем мягкое. В максимум
        # идут только корзины с опытом от MIN_SAMPLES проверок.
        # [замер] без порога одна удачная проверка в корзине (1 из 1 →
        # 0.67) поднимала уверенность всех слов выше неё до 0.67.
        overall = (sum(self.hits) + 1) / (sum(self.total) + 2)
        probability = None

        for index in range(self._bin(margin) + 1):
            if self.total[index] < self.MIN_SAMPLES:
                continue

            estimate = (self.hits[index] + 1) / (self.total[index] + 2)
            probability = estimate if probability is None else max(probability, estimate)

        return overall if probability is None else probability

    def state_dict(self) -> dict:
        return {"hits": self.hits, "total": self.total}

    def load_state_dict(self, state: dict) -> None:
        # Другая шкала (старое сохранение) — опыт собирается заново.
        if len(state["hits"]) != len(self.hits):
            return

        self.hits = list(state["hits"])
        self.total = list(state["total"])


# ==================================================================
# ИНТЕРФЕЙС
# ==================================================================


@dataclass
class Step:
    """Один такт восприятия: что пришло, что мозг ожидал, ошибка."""

    token: str
    expected: str | None = None
    expected_score: float = 0.0
    error: float | None = None
    dopamine: float = 0.0
    margin: float = 0.0


@dataclass
class Reply:
    """Ответ в диалоге + скорость (как у LLM: prompt и генерация)."""

    text: str
    report: "ReadReport"
    prompt_seconds: float = 0.0
    generated: int = 0
    generate_seconds: float = 0.0
    # Уверенность каждого слова ответа (0..1), выученная из опыта.
    confidence: list[float] = field(default_factory=list)
    # Слова реплики, которых мозг раньше не слышал.
    unknown: list[str] = field(default_factory=list)
    # Ход мыслей: [{thought, verdict: ok|no|stop, score}].
    thoughts: list[dict] = field(default_factory=list)


@dataclass
class ReadReport:
    tokens: int = 0
    sentences: int = 0
    errors: list[float] = field(default_factory=list)
    hits: int = 0
    dopamine: float = 0.0
    # Сколько раз засыпал посреди чтения.
    naps: int = 0

    def mean_error(self) -> float:
        return sum(self.errors) / len(self.errors) if self.errors else 0.0

    def merge(self, other: "ReadReport") -> None:
        self.tokens += other.tokens
        self.sentences += other.sentences
        self.errors += other.errors
        self.hits += other.hits
        self.dopamine += other.dopamine
        self.naps += other.naps


TEXT_SUFFIX_SKIP = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico", ".pdf",
    ".zip", ".gz", ".tar", ".7z", ".rar", ".exe", ".dll", ".so", ".bin",
    ".pt", ".pth", ".pyc", ".mp3", ".mp4", ".wav", ".avi", ".mov",
}


class LanguageInterface:
    """
    Ухо и рот мозга.

        текст -> tokenizer -> Ear -> auditory -> association ⇄ prefrontal
                                                    │
                                 ожидание = выученные синапсы
                                 (association E->E + prefrontal -> association)
                                                    │
                                 prediction error -> dopamine -> закрепление
                                                    │
                                            association -> motor -> Mouth -> текст

    Ответов и пар «вопрос -> ответ» нигде нет.
    Последовательности хранятся только в синапсах.
    """

    def __init__(self, brain: SpikingBrain, cfg):
        self.brain = brain
        self.cfg = cfg
        self.device = brain.device

        self.ear = Ear(cfg, self.device)

        self.meaning = self.brain.areas["association"]
        self.context = self.brain.areas["prefrontal"]
        self.motor = self.brain.areas["motor"]

        self.semantic = Readout(
            self.meaning.E.size, cfg.readout_top_k, cfg.readout_rate, self.device
        )
        # Рот: какой моторный паттерн произносит какое слово.
        # Выучивается, пока мозг слушает (моторный резонанс).
        self.mouth = Readout(
            self.motor.E.size, cfg.readout_top_k, cfg.readout_rate, self.device
        )

        self.concepts = Concepts(
            self.meaning.E.size, cfg.concept_cells, cfg.seed, self.device
        )
        # Места слов в рабочей памяти (ПФК).
        self.memory = Concepts(
            self.context.E.size, cfg.concept_cells, cfg.seed + 1, self.device
        )

        self.confidence = Confidence()
        self.emotions = Emotions()
        self.hippocampus = Hippocampus(
            cfg.hippocampus_capacity, cfg.replay_priority, seed=cfg.seed
        )
        self.sleeping = False
        # Блуждание мыслей (живой режим): мысли видны, но не запоминаются.
        self.musing = False

        self.last_rates: torch.Tensor | None = None
        self.last_token: str | None = None
        # Слова текущей фразы (с последней паузы).
        self.heard_in_phrase: list[str] = []
        # Слова, про которые мозг сам спрашивал «что такое…».
        self.asked: set[str] = set()
        # Незнакомые слова из чтения — повод спросить самому.
        self.wonder: list[str] = []
        self._feelings: dict[str, torch.Tensor] = {}
        # Выбор слова по вероятности (свой генератор — воспроизводимо).
        self._random = random.Random(cfg.seed)
        # Чувствительность клеток ядер нейромедиаторов.
        self._gains: dict[str, torch.Tensor] = {}
        # Неожиданные переходы: (проекция, клетки-источник, место B) —
        # здесь вырастут синапсы.
        self._growth: list[tuple[str, torch.Tensor, torch.Tensor]] = []
        # Укрепление, причитающееся ещё не выросшим синапсам контекста.
        self._owed: list[tuple[torch.Tensor, torch.Tensor, float]] = []
        self._prev_place: torch.Tensor | None = None
        self._place: torch.Tensor | None = None
        self._held: torch.Tensor | None = None
        # Медленная тема: места слов прошлых фраз (тают по фразам) и
        # куда они толкают смысл через выученные связи.
        self._sentence_places: list[torch.Tensor] = []
        self._topic: torch.Tensor | None = None
        self._topic_drive: torch.Tensor | None = None
        # Чувство, зафиксированное на время текущей реплики.
        self.held_mood: str | None = None
        # След вспомненного разговора (прайминг ответа).
        self.recall_drive: torch.Tensor | None = None

        self.experienced_tokens = 0
        self.sentences_lived = 0
        # Наблюдатель (веб-интерфейс): вызывается после каждого слова.
        self.on_step: Callable[[Step, bool], None] | None = None
        # Стрим ответа: вызывается, как только мозг выбрал слово.
        self.on_speak: Callable[[str, float], None] | None = None
        # Ход мыслей при размышлении: каждая мысль и вердикт проверки.
        self.on_think: Callable[[dict], None] | None = None
        # Микроскоп: где прорастают синапсы — [(путь, клетки-источник,
        # клетки-цель)] в глобальных индексах.
        self.on_grow: Callable[[list[tuple[str, list[int], list[int]]]], None] | None = None
        # Выбор слова при речи: слово и кандидаты с вероятностями.
        self.on_choice: Callable[[str, list[tuple[str, float]]], None] | None = None
        self.dialog_turns = 0
        # Журнал переписки: внешняя «записка», которую можно перечитать.
        # Сам разговор мозг помнит синапсами и гиппокампом, а не журналом.
        self.dialog: list[dict] = []

    # ==============================================================
    # TEXT
    # ==============================================================

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return re.findall(
            r"[^\W_]+(?:[-'/_.][^\W_]+)*",
            text.lower(),
            flags=re.UNICODE,
        )

    @staticmethod
    def sentences(text: str) -> list[str]:
        """
        Фразы, между которыми мозг делает паузу.

        «?» паузой не считается: «какая столица франции? париж» — одна
        мысль, и переход «вопрос -> ответ» должен учиться так же, как
        «франции -> париж». После паузы следующий переход не учится.
        """

        return [
            part
            for part in re.split(r"[.!;\n]+(?:\s|$)|\n", text)
            if part.strip()
        ]

    def _area(self, rates: torch.Tensor, area) -> torch.Tensor:
        return rates[area.E.slice]

    # ==============================================================
    # PERCEPTION
    # ==============================================================

    def perceive(self, token: str, learn_forms: bool = True) -> torch.Tensor:
        """Услышать один токен; вернуть частоты всех нейронов."""

        self._feel_inside()

        # Слово будит своё место в смысловой коре (новое — получает
        # место, пока мозг учится; калибровочные звуки — нет). Новое
        # встаёт на карте рядом со словом, после которого прозвучало.
        place = (
            self.concepts.get(token, allocate=learn_forms, near=self._place)
            if self.cfg.concepts_enabled else None
        )

        held = (
            self.memory.get(token, allocate=learn_forms, near=self._held)
            if self.cfg.memory_places else None
        )
        self._held = held

        if place is not None:
            self.brain.feel("association", place, self.cfg.concept_mv)

            if self.brain.plasticity:
                self._sentence_places.append(place)
            # Прайминг: недавно звучавшее слово легче всплывает снова.
            self._priming_trace()[place] = 1.0

        if held is not None:
            # Устойчивая активность ПФК: прошлые места тают, новое — полное.
            trace = self._memory_trace()
            trace.mul_(self.cfg.memory_decay)
            trace[held] = self.cfg.memory_mv
            self._hold_memory()

        rates = self.brain.observe(
            self.cfg.stimulus_ms,
            self.ear.encode(token),
        )

        if place is not None:
            self.brain.feel("association", None, 0.0)

            # Миндалина узнаёт слово — предвкушение хорошего или плохого
            # (наблюдение в замороженном мозге настроение не меняет).
            value = self.value(rates) if self.brain.plasticity else 0.0

            if value:
                self._feel("anticipate", self.cfg.value_mood * value)


        self._prev_place, self._place = self._place, place

        self.brain.run(self.cfg.gap_ms)
        self.brain.homeostasis(rates)

        if learn_forms and place is not None:
            self._learn_form(rates)

        if learn_forms:
            meaning = self._area(rates, self.meaning)

            # Портрет слова — его место, а не всё, что ещё не погасло от
            # прошлых слов (иначе «конец фразы» похож на что угодно).
            if place is not None:
                own = torch.zeros_like(meaning)
                own[place] = meaning[place]
                meaning = own

            self.semantic.observe(token, meaning)
            self.mouth.observe(token, self._area(rates, self.motor))
            self.experienced_tokens += 1

        return rates

    def _memory_trace(self) -> torch.Tensor:
        if getattr(self, "_trace", None) is None:
            self._trace = torch.zeros(self.context.E.size, device=self.device)

        return self._trace

    def _without_current(self, rates: torch.Tensor) -> torch.Tensor:
        """
        Контекст — то, что было ДО текущего слова: место текущего слова в
        рабочей памяти не участвует (за него уже отвечает прямая связь
        слово → слово). Иначе частое слово («является») своими связями из
        памяти повторяет то же, что говорит само.
        """

        if not self.cfg.memory_places or self.last_token is None:
            return rates

        held = self.memory.get(self.last_token, allocate=False)

        if held is None:
            return rates

        rates = rates.clone()
        rates[self.context.E.start + held] = 0.0

        return rates

    def _update_topic(self) -> None:
        """
        Тема — медленный след: места слов прошлых фраз, тают ×topic_decay
        за фразу. Тема предлагает кандидатов: куда её слова толкают смысл
        по выученным связям («говорили о Франции» → Париж, Европа).
        """

        if self.cfg.topic_gain <= 0 or not self.brain.plasticity:
            self._sentence_places = []
            return

        if self._topic is None:
            self._topic = torch.zeros(self.meaning.E.size, device=self.device)

        self._topic.mul_(self.cfg.topic_decay)

        for place in self._sentence_places:
            self._topic[place] = 1.0

        self._sentence_places = []

        if not self._topic.any():
            self._topic_drive = None
            return

        drive = self.brain.learned_drive(self._full(self._topic), "association")
        norm = drive.norm()
        self._topic_drive = drive / norm if norm > 0 else None

    def _with_topic(self, drive: torch.Tensor) -> torch.Tensor:
        """Ожидание + предложение темы (доля topic_gain силы ожидания)."""

        if self._topic_drive is None or self.cfg.topic_gain <= 0:
            return drive

        if not drive.any():
            # Начало мысли: предлагает одна тема.
            return self._topic_drive.clone()

        if self.cfg.topic_everywhere:
            return drive + self.cfg.topic_gain * drive.norm() * self._topic_drive

        return drive

    def _priming_trace(self) -> torch.Tensor:
        if getattr(self, "_priming", None) is None:
            self._priming = torch.zeros(self.meaning.E.size, device=self.device)

        return self._priming

    def _hold_memory(self) -> None:
        cells = torch.arange(self.context.E.size, device=self.device)
        self.brain.feel("prefrontal", cells, self._memory_trace())

    def _learn_context(self, context: torch.Tensor, place: torch.Tensor, dopamine: float) -> None:
        """
        Путь «память → смысл» учится по совпадению под дофамином:
        связи от активных клеток контекста к месту нового слова крепнут
        пропорционально дофамину. STDP тут не работает: клетки памяти
        стреляют непрерывно, и сбалансированное окно даёт LTP ≈ LTD.
        """

        rate = self.cfg.context_hebb_rate

        if rate <= 0 or not self.brain.plasticity:
            return

        proj = next((p for p in self.brain.plastic if p.name == "prefrontal -> association"), None)

        if proj is None:
            return

        source = self.context.E.start + context
        rows = place
        hit = torch.isin(proj.pre[rows], source.to(proj.pre.dtype)).float()
        level = dopamine / self.cfg.surprise_dopamine_gain
        delta = rate * level * hit * (proj.cap - proj.weight[rows])
        proj.weight[rows] += delta
        # Как всё выученное — сначала свежее (не поспал — забыл).
        proj.early[rows] += delta

    # ==============================================================
    # МИНДАЛИНА: какие слова к добру, какие к худу
    # ==============================================================

    def _show_growth(self, links: list[tuple[str, torch.Tensor, torch.Tensor]]) -> None:
        """Сообщить микроскопу, где прорастают синапсы (глобальные индексы)."""

        if not self.on_grow:
            return

        shown = []

        for name, pre, post in links[-24:]:
            proj = next((p for p in self.brain.projections if p.name == name), None)

            if proj is not None:
                shown.append((
                    name,
                    (pre[:16] + proj.pre_pop.start).tolist(),
                    (post[:16] + proj.post.start).tolist(),
                ))

        if shown:
            self.on_grow(shown)

    def _value_groups(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Клетки «хорошо» и «плохо» (локальные индексы E миндалины)."""

        area = self.brain.areas.get("amygdala")

        if area is None:
            return None

        if getattr(self, "_groups", None) is None:
            order = torch.randperm(area.E.size, generator=torch.Generator().manual_seed(0))
            half = area.E.size // 2
            self._groups = (order[:half].to(self.device), order[half:].to(self.device))

        return self._groups

    def _value_path(self):
        return next((p for p in self.brain.projections if p.name == "association -> amygdala"), None)

    def _learn_value(self, amplitude: float) -> None:
        """
        Похвала/ругань: места слов последнего обмена репликами прорастают
        и крепнут к группе «хорошо»/«плохо» (по совпадению под наградой).
        """

        groups, path = self._value_groups(), self._value_path()

        if groups is None or path is None or not self.cfg.concepts_enabled:
            return

        words = [
            word
            for turn in self.dialog[-2:]
            for word in self.tokenize(turn.get("answer", turn["text"]) if turn["who"] == "мозг" else turn["text"])
        ]
        places = [p for p in (self.concepts.get(w, allocate=False) for w in dict.fromkeys(words)) if p is not None]

        if not places:
            return

        group = groups[0] if amplitude > 0 else groups[1]
        links = [("association -> amygdala", place, group) for place in places]
        self._show_growth(links)
        self.brain.grow(links, self.cfg.value_grow)

        source = self.meaning.E.start + torch.cat(places)
        hit = torch.isin(path.pre[group], source.to(path.pre.dtype)).float()
        level = min(1.0, abs(amplitude) / self.cfg.surprise_dopamine_gain)
        path.weight[group] += self.cfg.value_rate * level * hit * (self.cfg.value_cap - path.weight[group])

    def value(self, rates: torch.Tensor) -> float:
        """
        Предвкушение: насколько услышанное к добру (+) или к худу (−) —
        вход по выученным синапсам в группы «хорошо» и «плохо».
        """

        groups, path = self._value_groups(), self._value_path()

        if groups is None or path is None:
            return 0.0

        heard = rates.index_select(0, path.pre.flatten().long()).view_as(path.weight) > 0
        drive = (heard * path.weight).sum(1)
        good, bad = drive[groups[0]].mean(), drive[groups[1]].mean()

        return float((good - bad) / self.cfg.value_cap)

    def _form_path(self):
        return next(
            (p for p in self.brain.projections if p.name == "auditory -> association"), None
        )

    def _learn_form(self, rates: torch.Tensor) -> None:
        """
        Узнавание по звучанию (instar): клетки смысла, которые стреляли,
        пока звучало слово, тянут свои входы от слуха к узору звучания.
        """

        path = self._form_path()

        if path is None or not self.brain.plasticity:
            return

        heard = (rates.index_select(0, path.pre.flatten().long()).view_as(path.weight) > 0).float()
        active = (rates[path.post.slice] > 0).float()[:, None]
        path.weight.add_(self.cfg.form_rate * active * (self.cfg.form_cap * heard - path.weight))

    def recognize(self, rates: torch.Tensor) -> torch.Tensor:
        """Куда звучание (активность слуха) толкает смысл по выученному пути."""

        path = self._form_path()
        drive = torch.zeros(self.meaning.E.size, device=self.device)

        if path is not None:
            heard = rates.index_select(0, path.pre.flatten().long()).view_as(path.weight)
            drive = (heard * path.weight).sum(1)

        return drive

    def rest(self, connected: bool = False) -> None:
        """
        Тишина между фразами: гомеостаз возвращает синапсы
        в рабочий диапазон. Рабочая память затухает сама.

        connected=True — пауза внутри связного текста: мысль не
        обрывается, рабочая память лишь слабеет, и первое слово следующей
        фразы предсказывается и учится из контекста прошлой. Реплики
        разговора и сны разделяются полностью.
        """

        self.brain.run(self.cfg.rest_ms)
        self.brain.forget()

        self.sentences_lived += 1

        if not connected:
            # Новая мысль — переход от прошлой не учится (см. converse).
            self._place = None

        if self.cfg.memory_places:
            decay = self.cfg.text_memory_decay if connected else self.cfg.memory_rest_decay
            self._memory_trace().mul_(decay)
            self._hold_memory()

        self._priming_trace().mul_(self.cfg.priming_decay)
        self._update_topic()

        if self._growth and self.sentences_lived % self.cfg.grow_every_sentences == 0:
            self._show_growth(self._growth)
            self.brain.grow(self._growth, self.cfg.grow_synapses)
            self._growth = []

            # Выросшие синапсы контекста сразу получают укрепление, которое
            # им причиталось в момент неожиданности: иначе при следующей
            # встрече неожиданности уже нет, слабые новые синапсы сжимает
            # масштабирование, и перестройка их убивает («мадрид → испании»
            # оставался с 0 синапсов).
            for held, place, dopamine in self._owed:
                self._learn_context(held, place, dopamine)

        if len(self._owed) and not self._growth:
            self._owed = []

        if (
            self.cfg.synaptic_scaling_enabled
            and self.sentences_lived % self.cfg.synaptic_scaling_every == 0
        ):
            self.brain.synaptic_scaling()

        if (
            self.cfg.structural_plasticity_enabled
            and self.sentences_lived % self.cfg.rewire_every_sentences == 0
        ):
            self.brain.rewire()

        if not connected:
            self.last_rates = None
            self.last_token = None

        self.heard_in_phrase = []

    # ==============================================================
    # PREDICTION
    # ==============================================================

    def expectation(self, rates: torch.Tensor) -> torch.Tensor:
        """
        Ожидаемое следующее состояние ассоциативной коры:
        возбуждение, которое текущее слово И контекст
        рабочей памяти посылают через выученные синапсы
        в нейроны, которые сейчас молчат.
        """

        drive = self.brain.learned_drive(self._without_current(rates), "association")
        active = self._area(rates, self.meaning) > 0
        drive = drive.masked_fill(active, 0.0)

        # Прайминг темы: умножает, а не добавляет — тема побеждает среди
        # кандидатов текущего слова, но не вставляет слова из ниоткуда.
        if self.cfg.priming_gain > 0:
            drive = drive * (1.0 + self.cfg.priming_gain * self._priming_trace())

        drive = self._with_topic(drive)

        # Выстрелили бы только самые возбуждённые: торможение зоны
        # оставляет ~target_activity нейронов (k-WTA).
        fraction = self.cfg.expectation_top_fraction

        if fraction > 0:
            k = max(1, int(drive.numel() * fraction))
            top = drive.topk(k)
            drive = torch.zeros_like(drive).scatter_(0, top.indices, top.values)

        return drive

    @staticmethod
    def _winners(drive: torch.Tensor, fraction: float) -> torch.Tensor:
        """k-WTA: остаются только самые возбуждённые нейроны."""

        k = max(1, int(drive.numel() * fraction))
        top = drive.topk(k)

        return torch.zeros_like(drive).scatter_(0, top.indices, top.values)

    def speak(self, expected: torch.Tensor) -> torch.Tensor:
        """
        Ожидание смысла -> ожидаемый моторный паттерн (рот).

        В моторную зону уходят только нейроны смысла, которые
        реально выстрелили бы, и в ней самой побеждают самые
        возбуждённые — как при настоящем произнесении слова.
        """

        sparse = self.cfg.mouth_sparse

        if sparse:
            expected = self._winners(expected, self._target("association"))

        full = torch.zeros(self.brain.n, device=self.device)
        full[self.meaning.E.slice] = expected
        proj = self.brain.projection("association -> motor")

        got = full.index_select(0, proj.pre.flatten()).view_as(proj.weight)
        motor = (got * proj.weight).sum(1)

        return self._winners(motor, self._target("motor")) if sparse else motor

    def _target(self, area: str) -> float:
        return next(a.target_activity for a in self.cfg.areas if a.name == area)

    def predict(
        self,
        rates: torch.Tensor,
        exclude: set[str] | None = None,
        top: int = 5,
    ) -> list[tuple[str, float]]:
        return self.semantic.decode(self.expectation(rates), exclude, top)

    # ==============================================================
    # DOPAMINE
    # ==============================================================

    def surprise_dopamine(self, error: float, hit: bool) -> float:
        """
        Сбывшееся предсказание -> 0.
        Мелкая неожиданность (ниже порога) -> 0.
        Сильная ошибка -> квадратично больше.
        """

        if hit:
            return 0.0

        threshold = self.cfg.surprise_threshold
        level = max(0.0, error - threshold) / (1.0 - threshold)

        return self.cfg.surprise_dopamine_gain * level * level

    def _has_nuclei(self) -> bool:
        return self.cfg.nuclei_enabled and "vta" in self.brain.areas and "lc" in self.brain.areas

    def _nucleus_gains(self, area: str) -> torch.Tensor:
        """Разная чувствительность клеток ядра: low..1, вперемешку."""

        cached = self._gains.get(area)

        if cached is None:
            size = self.brain.areas[area].E.size
            rng = torch.Generator().manual_seed(sorted(self.brain.areas).index(area))
            gains = torch.linspace(self.cfg.nucleus_gain_low, 1.0, size)
            cached = gains[torch.randperm(size, generator=rng)].to(self.device)
            self._gains[area] = cached

        return cached

    def _nucleus_burst(self, level: float, lc: bool = True) -> tuple[float, float]:
        """
        Залп ядер на событие силы level: VTA -> дофамин, голубое пятно ->
        норадреналин. Возвращает (дофамин, доля LC).

        У VTA тонический фон: часть клеток стреляет и без событий.
        Дофамин — отклонение от фона: награда/неожиданность (level > 0) —
        залп выше фона, ругань (level < 0) — пауза ниже фона (торможение).
        LC отвечает только на значимое (level > 0) и только если lc.
        """

        cfg = self.cfg
        cells = {"vta": self.brain.areas["vta"].E}
        base = self._vta_base()

        gains = self._nucleus_gains("vta")
        drive = (cfg.vta_tonic_mv + cfg.nucleus_mv * level) * gains
        self.brain.feel("vta", torch.arange(cells["vta"].size, device=self.device), drive)

        if lc and level > 0:
            E = cells["lc"] = self.brain.areas["lc"].E
            amplitude = cfg.nucleus_mv * level * self._nucleus_gains("lc")
            self.brain.feel("lc", torch.arange(E.size, device=self.device), amplitude)

        rates = self.brain.observe(cfg.nucleus_ms)

        for area in cells:
            self.brain.feel(area, None, 0.0)

        fired = {area: float((rates[E.slice] > 0).float().mean()) for area, E in cells.items()}
        delta = fired["vta"] - base
        # Отклонение в долях возможного: все клетки — +1, тишина — −1.
        scale = (1.0 - base) if delta >= 0 else base

        return cfg.surprise_dopamine_gain * delta / max(scale, 1e-3), fired.get("lc", 0.0)

    def _vta_base(self) -> float:
        """Доля клеток VTA, стреляющих на одном тоническом фоне (замер один раз)."""

        if getattr(self, "_base", None) is None:
            self._base = 0.0

            if self.cfg.vta_tonic_mv > 0:
                E = self.brain.areas["vta"].E
                drive = self.cfg.vta_tonic_mv * self._nucleus_gains("vta")
                self.brain.feel("vta", torch.arange(E.size, device=self.device), drive)
                rates = self.brain.observe(self.cfg.nucleus_ms)
                self.brain.feel("vta", None, 0.0)
                self._base = float((rates[E.slice] > 0).float().mean())

        return self._base

    def _dopamine(self, amplitude: float) -> float:
        """Внешняя награда (похвала, ругань, любопытство) — через VTA, если оно есть."""

        if not self._has_nuclei():
            return amplitude

        dopamine, _ = self._nucleus_burst(amplitude / self.cfg.surprise_dopamine_gain, lc=False)

        return dopamine

    # ==============================================================
    # EXPERIENCE
    # ==============================================================

    def _step(self, token: str, learn: bool) -> Step:
        expected_state = None
        step = Step(token=token)

        if self.last_rates is not None:
            expected_state = self.expectation(self.last_rates)
            ranked = self.semantic.decode(
                expected_state,
                exclude={self.last_token} if self.last_token else None,
                top=20,
            )

            if ranked:
                step.expected, step.expected_score = ranked[0]
                step.margin = Confidence.margin(ranked)

        # Контекст до нового слова: активные клетки рабочей памяти.
        context = None

        if self.last_rates is not None:
            memory = self._area(self._without_current(self.last_rates), self.context)
            context = memory.topk(min(self.cfg.context_grow_cells, memory.numel())).indices
            context = context[memory[context] > 0]

        rates = self.perceive(token, learn_forms=learn)

        if expected_state is not None and token in self.semantic:
            step.error = 1.0 - _cosine(
                expected_state, self.semantic.prototype(token)
            )

            if learn and step.expected is not None:
                # Мозг сверяет: при такой резкости ожидания сбылось ли?
                self.confidence.update(step.margin, step.expected == token)

            if learn:
                step.dopamine = self.surprise_dopamine(
                    step.error, hit=step.expected == token
                )
                noradrenaline = None

                # Неожиданность приходит в ядра нейромедиаторов; сколько
                # их клеток выстрелит — столько дофамина и норадреналина.
                if step.dopamine > 0 and self._has_nuclei():
                    step.dopamine, noradrenaline = self._nucleus_burst(
                        step.dopamine / self.cfg.surprise_dopamine_gain
                    )

                # Неожиданный переход: связи для него нужны — вырастут.
                # От места прошлого слова и от контекста рабочей памяти.
                if step.dopamine > 0 and self.cfg.grow_on_surprise and self._place is not None:
                    if self._prev_place is not None:
                        self._growth.append(("association: E->E", self._prev_place, self._place))

                    if context is not None and len(context):
                        # Рост — от каждого слова контекста своими синапсами:
                        # из общего пула клеток конкретному слову («мадрид»)
                        # могло не достаться ни одного.
                        for word in dict.fromkeys(self.heard_in_phrase[:-1]):
                            held = self.memory.get(word, allocate=False)

                            if held is not None and bool(torch.isin(held, context).any()):
                                self._growth.append(("prefrontal -> association", held, self._place))
                                self._owed.append((held, self._place, step.dopamine))

                        self._learn_context(context, self._place, step.dopamine)

                # Неожиданный переход закрепляется: dopamine
                # превращает eligibility traces только что
                # прошедших пар в изменение весов.
                self.brain.reward(step.dopamine)
                self._feel("heard",
                    step.dopamine,
                    step.expected == token if step.expected is not None else None,
                    noradrenaline,
                )

        self.last_rates = rates
        self.last_token = token
        self.heard_in_phrase.append(token)

        if self.on_step:
            self.on_step(step, learn)

        return step

    def experience(self, tokens: list[str], learn: bool = True) -> list[Step]:
        return [self._step(token, learn) for token in tokens]

    # ==============================================================
    # ЧТЕНИЕ ДАННЫХ
    # ==============================================================

    def read_text(
        self,
        text: str,
        progress: Callable[[ReadReport], None] | None = None,
        stop=None,
        nap: bool = True,
    ) -> ReadReport:
        """
        Мозг живёт внутри потока текста. Пластичность
        регулируется prediction error. nap=False — не засыпать
        посреди текста (сервер спит сам между порциями, чтобы будиться).
        """

        report = ReadReport()

        for sentence in self.sentences(text):
            if stop is not None and stop.is_set():
                break

            tokens = self.tokenize(sentence)

            if not tokens:
                continue

            salience = 0.0
            new_before = [t for t in tokens if not self.familiar(t)]
            self.held_mood = self.emotions.mood.name

            for step in self.experience(self._ended(tokens)):
                report.tokens += step.token != END
                report.dopamine += step.dopamine
                salience += step.dopamine

                if step.error is not None:
                    report.errors.append(step.error)
                    report.hits += step.expected == step.token

            report.sentences += 1
            self.held_mood = None
            self._remember(tokens, salience)
            self._note_new_words(tokens, new_before)
            self.rest(connected=self.cfg.connected_text)

            if progress and report.sentences % 20 == 0:
                progress(report)

            # Устал читать — поспать и продолжить (см. nap_while_reading).
            if nap and self.cfg.nap_while_reading and self.sleep_pressure >= self.cfg.nap_after_episodes:
                self.sleep_until_rested(stop)
                report.naps += 1

        return report

    @staticmethod
    def text_files(path: str) -> list[Path]:
        target = Path(path).expanduser()

        if target.is_file():
            return [target]

        return sorted(
            p
            for p in target.rglob("*")
            if p.is_file()
            and p.suffix.lower() not in TEXT_SUFFIX_SKIP
            and not any(part.startswith(".") for part in p.relative_to(target).parts)
        )

    @staticmethod
    def load_text(path: Path) -> str | None:
        """Любой текстовый файл; бинарные пропускаются."""

        data = path.read_bytes()

        if b"\x00" in data[:4096]:
            return None

        return data.decode("utf-8", errors="ignore")

    # ==============================================================
    # СОН: гиппокамп проигрывает эпизоды коре
    # ==============================================================

    def _remember(self, tokens: list[str], salience: float) -> None:
        """Гиппокамп запоминает фразу с одного раза (но не свои сны)."""

        if not self.sleeping:
            if self.cfg.emotions_enabled:
                salience *= 1.0 + self.cfg.arousal_salience * self.emotions.arousal
            self.hippocampus.store(tokens, salience)
            self.emotions.sleepy = self.sleep_pressure >= self.cfg.sleepy_after_episodes

    def sleep(
        self,
        replays: int,
        progress: Callable[[ReadReport], None] | None = None,
        stop=None,
    ) -> ReadReport:
        """
        Сон: гиппокамп проигрывает коре запомненные фразы, чаще —
        самые неожиданные. Кора учится на них обычным правилом
        (STDP + дофамин от ошибки предсказания). Выученная фраза почти
        не даёт дофамина — её важность падает, и гиппокамп её забывает.
        """

        report = ReadReport()
        self.sleeping = True
        self.emotions.sleeping = True
        self.brain.modulation = 1.0
        self.brain.consolidating = True

        try:
            for _ in range(replays):
                if stop is not None and stop.is_set():
                    break

                episode = self.hippocampus.sample()

                if episode is None:
                    break

                salience = 0.0

                for step in self.experience(self._ended(episode.tokens)):
                    report.tokens += step.token != END
                    report.dopamine += step.dopamine
                    salience += step.dopamine

                    if step.error is not None:
                        report.errors.append(step.error)
                        report.hits += step.expected == step.token

                episode.salience = salience
                episode.replays += 1
                report.sentences += 1
                self.rest()

                if progress and report.sentences % 20 == 0:
                    progress(report)
        finally:
            self.sleeping = False
            self.emotions.sleeping = False
            self.brain.consolidating = False

        self.hippocampus.forget_consolidated(self.cfg.consolidated_salience)
        self.emotions.sleepy = self.sleep_pressure >= self.cfg.sleepy_after_episodes

        return report

    @property
    def sleep_pressure(self) -> int:
        """Сколько новых впечатлений ещё ни разу не проиграно во сне."""

        return sum(1 for episode in self.hippocampus.episodes if episode.replays == 0)

    def sleep_until_rested(self, stop=None, progress=None) -> ReadReport:
        """Проспать столько, чтобы каждое новое впечатление проигралось N раз."""

        replays = max(self.cfg.sleep_batch, self.sleep_pressure * self.cfg.replays_per_episode)

        return self.sleep(replays, progress, stop)

    # ==============================================================
    # ДИАЛОГ: учится, пока разговаривает
    # ==============================================================

    def feedback(self, text: str) -> float | None:
        """Врождённый канал поощрения/наказания."""

        words = self.tokenize(text) or [text.strip()]

        if len(words) != 1:
            return None

        word = words[0]

        if word in self.cfg.praise_words or text.strip() in self.cfg.praise_words:
            return self.cfg.feedback_dopamine

        if word in self.cfg.blame_words or text.strip() in self.cfg.blame_words:
            return -self.cfg.feedback_dopamine

        return None

    def converse(self, text: str) -> Reply:
        """
        Реплика человека — это опыт: мозг слушает её
        с обучением от prediction error. Затем отвечает,
        продолжая собственное состояние.

        Ответ мозга не обучает сам себя, но оставляет
        eligibility traces: следующее «+» / «-» человека
        закрепит или ослабит именно его.
        """

        self.dialog_turns += 1
        user_text = text

        amplitude = self.feedback(text)

        if amplitude is not None:
            self.brain.reward(self._dopamine(amplitude))
            self._feel("feedback", amplitude)
            self._learn_value(amplitude)
            self._rate_last_turn(amplitude)
            self.brain.run(self.cfg.rest_ms)
            word = "закрепил" if amplitude > 0 else "ослабил"
            return Reply(f"[dopamine {amplitude:+.0f}: {word} последний ответ]", ReadReport())

        # Новая реплика — новая мысль: пауза, иначе мозг выучит переход
        # от конца прошлой реплики к началу этой («германии -> квантовая»).
        if self.last_rates is not None:
            self.rest()

        self.held_mood = self.emotions.mood.name
        report = ReadReport()
        sentences = [s for s in self.sentences(text) if self.tokenize(s)]
        # Незнакомые слова — до того, как мозг их услышит в этой реплике.
        unknown = list(dict.fromkeys(
            token
            for sentence in sentences
            for token in self.tokenize(sentence)
            if not self.familiar(token)
        ))
        start = time.perf_counter()

        for i, sentence in enumerate(sentences):
            salience = 0.0

            # Последняя фраза человека не кончается: от неё мозг отвечает.
            tokens = self.tokenize(sentence)
            last = i == len(sentences) - 1

            for step in self.experience(tokens if last else self._ended(tokens)):
                report.tokens += step.token != END
                report.dopamine += step.dopamine
                salience += step.dopamine

                if step.error is not None:
                    report.errors.append(step.error)
                    report.hits += step.expected == step.token

            report.sentences += 1
            self._remember(self.tokenize(sentence), salience)

            # Между фразами — пауза, но после последней мозг
            # сразу продолжает мысль.
            if i < len(sentences) - 1:
                self.rest()

        self._sync()
        prompt_seconds = time.perf_counter() - start

        start = time.perf_counter()
        last = self.last_token
        # Продолжать мысль от незнакомого слова не от чего.
        # Ответ на его собственный вопрос «что такое X?»: любопытство
        # удовлетворено — это награда (дофамин закрепляет только что
        # услышанное объяснение), а «вопрос -> объяснение» гиппокамп
        # запоминает одним эпизодом и закрепит во сне.
        heard = [token for sentence in sentences for token in self.tokenize(sentence)]
        explained = [word for word in dict.fromkeys(heard) if word in self.asked]

        for word in explained:
            self.brain.reward(self._dopamine(self.cfg.curiosity_reward))
            self.hippocampus.store(
                ["что", "такое", word] + heard[: self.cfg.explanation_max_words],
                salience=self.cfg.curiosity_reward,
            )
            self.asked.discard(word)
            self._feel("feedback", self.cfg.curiosity_reward)

        thoughts: list[dict] = []
        self.recall_conversation()

        if last in unknown:
            produced = []
        elif self.cfg.thinking_enabled:
            produced, thoughts = self.think()

            # Итог размышления произносится вслух.
            if self.on_speak:
                for token, sure in produced:
                    self.on_speak(token, sure)
        else:
            produced = self._continue(sample=True)
        self._sync()
        self._feel("answered", [sure for _, sure in produced])
        self.held_mood = None
        self.recall_drive = None

        answer = " ".join(token for token, _ in produced)
        question = ""

        # Врождённый рефлекс любопытства: спросить про новое.
        # Ответ человека — обычная реплика, на ней мозг и учится.
        # Короткие слова («и», «это») — обычно служебные, про них не
        # спрашивает; если в реплике есть слово, про которое он сам
        # спрашивал, — это объяснение: слушает, а не переспрашивает.
        explaining = bool(explained)
        worth_asking = sorted(
            (word for word in unknown if len(word) >= self.cfg.ask_min_letters),
            key=len,
            reverse=True,
        )[:2]

        if worth_asking and not explaining and self.cfg.ask_about_unknown:
            question = "а что такое " + " и ".join(f"«{word}»" for word in worth_asking) + "?"
            self.asked.update(worth_asking)
            self._feel("curious")

        text = " … ".join(part for part in (answer, question) if part) or "не знаю"
        self.dialog.append({"who": "человек", "text": user_text})
        self.dialog.append({"who": "мозг", "text": text, "answer": answer, "rating": 0})
        del self.dialog[: -self.cfg.dialog_log_turns]

        return Reply(
            text,
            report,
            prompt_seconds=prompt_seconds,
            generated=len(produced),
            generate_seconds=time.perf_counter() - start,
            confidence=[sure for _, sure in produced],
            unknown=unknown,
            thoughts=thoughts,
        )

    def familiar(self, token: str) -> bool:
        """Слышал ли мозг это слово раньше (хотя бы раз)."""

        row = self.semantic.index.get(token)

        return row is not None and self.semantic.observations[row] >= self.cfg.familiar_after

    def _feel(self, event: str, *args) -> None:
        """Передать событие лимбической системе, если эмоции включены."""

        # Во сне эмоции не реагируют на проигрываемые фразы.
        if self.cfg.emotions_enabled and not self.sleeping:
            getattr(self.emotions, event)(*args)

        self.brain.modulation = self._arousal_gain()

    def _arousal_gain(self) -> float:
        """Норадреналин: во сне и без эмоций — 1."""

        if not self.cfg.emotions_enabled or self.sleeping:
            return 1.0

        return 1.0 + self.cfg.arousal_plasticity * self.emotions.arousal

    def _min_confidence(self) -> float:
        """Порог уверенности ответа: настроение делает смелее или осторожнее."""

        base = self.cfg.generate_min_confidence

        if not self.cfg.emotions_enabled:
            return base

        return base * (1.0 - self.cfg.valence_boldness * self.emotions.valence)

    # ==============================================================
    # ИНИЦИАТИВА: мозг заговаривает сам
    # ==============================================================

    def _note_new_words(self, tokens: list[str], new: list[str]) -> None:
        """Незнакомые содержательные слова из чтения — про них можно спросить."""

        if self.sleeping:
            return

        for word in new:
            if len(word) >= self.cfg.ask_min_letters and word not in self.asked:
                self.wonder.append(word)

        del self.wonder[: -100]

    # Нейроны лимбической зоны на каждое чувство: у каждого своя,
    # непересекающаяся группа (как разные ядра миндалины). При случайных
    # пересекающихся наборах выученное «мне скучно» всплывало и в радости.
    def _feeling_neurons(self, mood: str) -> torch.Tensor | None:
        area = self.brain.areas.get("limbic")

        if area is None:
            return None

        cached = self._feelings.get(mood)

        if cached is None:
            moods = sorted(FACES)
            block = area.E.size // len(moods)
            index = moods.index(mood) if mood in moods else 0
            rng = torch.Generator().manual_seed(0)
            order = torch.randperm(area.E.size, generator=rng)
            cached = order[index * block: (index + 1) * block].to(self.device)
            self._feelings[mood] = cached

        return cached

    def _feel_inside(self) -> None:
        """
        Сигнал «изнутри»: текущее чувство возбуждает свой набор нейронов
        лимбической зоны, пока мозг слышит слова. Так путь «чувство ->
        смысл» учится: какими словами говорят о том, что он чувствует.
        """

        if "limbic" not in self.brain.areas:
            return

        if not self.cfg.emotions_enabled:
            self.brain.feel("limbic", None, 0.0)
            return

        emotions = self.emotions
        amplitude = self.cfg.interoception_mv + self.cfg.interoception_arousal_mv * emotions.arousal
        # Чувство держится всю реплику: настроение за фразу не скачет.
        mood = self.held_mood or emotions.mood.name
        self.brain.feel("limbic", self._feeling_neurons(mood), amplitude)

    def initiative(self) -> tuple[str, str] | None:
        """
        Заговорить самому — из того, что чувствует.

        Мозг «прислушивается к себе»: текущее чувство возбуждает
        лимбическую зону, через выученные связи будит слова, и из этого
        состояния рождается фраза (как обычный ответ). Любопытство сначала
        вспоминает незнакомое слово из чтения; в скуке и спокойствии
        незнакомое слово — запасной повод, если чувство ничего не будит.
        Нет ни того, ни другого — молчит.

        Заготовок нет: «мне скучно» он скажет, только если слышал эти
        слова, когда ему было скучно.
        """

        if "limbic" not in self.brain.areas or not self.cfg.emotions_enabled:
            return None

        mood = self.emotions.mood.name

        if self.last_rates is not None:
            self.rest()

        self._feel_inside()
        # Дать чувству установиться: клетки прошлого чувства ещё
        # несколько миллисекунд досрабатывают по инерции.
        self.brain.run(self.cfg.feel_settle_ms)
        rates = self.brain.observe(self.cfg.feel_ms)
        self.last_rates, self.last_token = rates, None

        first = None

        # [замер] раньше в скуке незнакомое слово шло первым: после любого
        # чтения в wonder до 100 слов, и выученное «мне скучно» не
        # звучало никогда (тест проходил лишь потому, что чувство
        # просачивалось в ожидание каждого слова — см. predictive).
        if not (self.wonder and mood in ("любопытство", "удивление")):
            # Первое слово должно разбудить само чувство (через выученные
            # связи «чувство -> смысл»), а не случайное воспоминание.
            ranked = self.semantic.decode(
                self.brain.learned_drive(rates, "association", source="limbic"), top=20
            )

            if (
                ranked
                and ranked[0][1] >= self.cfg.generate_min_score
                and Confidence.margin(ranked) >= self.cfg.generate_min_margin
            ):
                first = [ranked[0][0]]

        if first is None:
            if not (self.wonder and mood in ("любопытство", "удивление", "спокойствие", "скука")):
                return None

            # Вспомнить слово, о котором хочется спросить.
            first = [self.wonder.pop()]
            self.asked.add(first[0])

        self.experience(first, learn=False)
        produced = self._continue(sample=True)

        return mood, " ".join(first + [token for token, _ in produced])

    def wander(self) -> tuple[str, list[dict]] | None:
        """
        Блуждание мыслей в тишине: взять тему (последнюю реплику
        человека или пережитую фразу) и подумать про себя. Мысли видны
        (on_think / on_step), но не запоминаются и не меняют синапсы.
        """

        topic = next(
            (turn["text"] for turn in reversed(self.dialog) if turn["who"] == "человек"),
            None,
        )

        if (topic is None or self.hippocampus.random.random() < 0.5) and self.hippocampus.episodes:
            episode = self.hippocampus.random.choice(self.hippocampus.episodes)
            topic = " ".join(episode.tokens[:3])

        tokens = [t for t in self.tokenize(topic or "") if self.familiar(t)][-3:]

        if not tokens:
            return None

        self.musing = True

        try:
            with self.isolated():
                self.experience(tokens, learn=False)
                produced, log = self.think()
        finally:
            self.musing = False

        return " ".join(tokens + [t for t, _ in produced]), log

    # ==============================================================
    # ВСПОМИНАНИЕ РАЗГОВОРА
    # ==============================================================

    def recall_conversation(self) -> None:
        """
        Перед ответом гиппокамп реактивирует следы последних реплик:
        их слова будят свои выученные связи в коре, и это подталкивает
        ответ (прайминг). Чем реплика старше, тем слабее. Текст чата на
        вход не подаётся — это вспоминание, а не перечитывание.
        """

        self.recall_drive = None
        turns = [
            turn.get("answer", turn["text"]) if turn["who"] == "мозг" else turn["text"]
            for turn in self.dialog
            if turn.get("rating", 0) >= 0 and not turn.get("initiative")
            and (self.cfg.recall_own_answers or turn["who"] != "мозг")
        ][-self.cfg.recall_turns:]

        if not turns or self.cfg.recall_gain <= 0:
            return

        drive = torch.zeros(self.meaning.E.size, device=self.device)

        for age, text in enumerate(reversed(turns)):
            words = [w for w in self.tokenize(text) if w in self.semantic]

            if not words:
                continue

            trace = sum(self.semantic.prototype(word) for word in words)
            part = self.brain.learned_drive(self._full(trace), "association")
            norm = part.norm()

            if norm > 0:
                drive += (self.cfg.recall_decay ** age) * part / norm

        self.recall_drive = drive if drive.any() else None

    def _with_recall(self, expected: torch.Tensor) -> torch.Tensor:
        """Ожидание + след вспомненного разговора (в долях силы ожидания)."""

        if self.recall_drive is None or not expected.any():
            return expected

        scale = expected.norm() * self.cfg.recall_gain

        return expected + scale * self.recall_drive / self.recall_drive.norm()

    # ==============================================================
    # ПАМЯТЬ ДИАЛОГА
    # ==============================================================

    def _rate_last_turn(self, amplitude: float) -> None:
        """
        Похвала: вопрос + ответ запоминаются гиппокампом как эпизод
        и закрепятся во сне. Ругань: ответ помечается, чтобы мозг не
        перечитывал и не закреплял свою ошибку.
        """

        turn = next((t for t in reversed(self.dialog) if t["who"] == "мозг"), None)

        if turn is None:
            return

        turn["rating"] = 1 if amplitude > 0 else -1

        if amplitude > 0 and turn.get("answer"):
            index = self.dialog.index(turn)
            question = self.dialog[index - 1]["text"] if index else ""
            tokens = self.tokenize(question) + self.tokenize(turn["answer"])
            self.hippocampus.store(tokens, salience=amplitude)

    def reread(self, turns: int, progress=None, stop=None) -> ReadReport:
        """
        Перечитать переписку (как человек перечитывает чат, когда забыл).
        Свои ответы мозг читает как текст; отруганные — пропускает.
        """

        lines = [
            turn.get("answer", turn["text"]) if turn["who"] == "мозг" else turn["text"]
            for turn in self.dialog[-turns:]
            if turn.get("rating", 0) >= 0
        ]

        return self.read_text(".\n".join(line for line in lines if line), progress, stop)

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def _ended(self, tokens: list[str]) -> list[str]:
        return tokens + [END] if self.cfg.sentence_end and tokens else tokens

    def _chances(self, ranked: list[tuple[str, float]]) -> list[tuple[str, float]]:
        """Вероятности кандидатов при выборе по вероятности (для показа)."""

        top = ranked[: self.cfg.speak_top_k]
        temperature = max(self.cfg.speak_temperature, 1e-3)
        weights = [2.718281828 ** ((score - top[0][1]) / temperature) for _, score in top]
        total = sum(weights)

        return [(token, weight / total) for (token, _), weight in zip(top, weights)]

    def _pick(self, ranked: list[tuple[str, float]], sample: bool) -> str:
        """
        Какое слово сказать. sample=False — самое вероятное (экзамен,
        пробы). sample=True — по вероятности среди speak_top_k лучших:
        сильный почти всегда побеждает, близкий второй иногда тоже —
        мозг не повторяет один и тот же ответ слово в слово.
        """

        if not sample or self.cfg.speak_temperature <= 0:
            return ranked[0][0]

        top = ranked[: self.cfg.speak_top_k]
        best = top[0][1]
        weights = [2.718281828 ** ((score - best) / self.cfg.speak_temperature) for _, score in top]

        return self._random.choices([token for token, _ in top], weights=weights)[0]

    def _continue(
        self, max_tokens: int | None = None, sample: bool = False
    ) -> list[tuple[str, float]]:
        """
        state -> ожидание -> рот -> слово -> снова в ухо -> ...

        Собственная речь слышна мозгу (обратная связь через ухо),
        но не даёт surprise-dopamine.
        """

        if self.last_rates is None:
            return []

        produced: list[tuple[str, float]] = []
        # Привыкание: прозвучавшие в этой фразе слова (реплика человека
        # и свой ответ) мозг не повторяет — иначе ответ ходит по кругу.
        # repeat_prompt_words — разрешить слова вопроса (см. config).
        recent = (
            {self.last_token} if self.cfg.repeat_prompt_words and self.last_token
            else set(self.heard_in_phrase)
        )

        for _ in range(max_tokens or self.cfg.generate_max_tokens):
            expected = self._with_recall(self.expectation(self.last_rates))

            # Слово выбирает ассоциативная кора; моторная зона
            # пока только отражает речь (см. «рот произнёс бы»).
            ranked = self.semantic.decode(expected, exclude=recent, top=20)

            if not ranked:
                break

            score = ranked[0][1]
            token = self._pick(ranked, sample)
            margin = Confidence.margin(ranked)

            # Мысль кончилась.
            if token == END:
                break
            sure = self.confidence(margin)

            if score < self.cfg.generate_min_score or margin < self.cfg.generate_min_margin:
                break

            # Когда мозг уже знает свою надёжность — молчит,
            # если почти наверняка ошибётся.
            if (
                self.confidence.experience >= self.cfg.confidence_min_experience
                and sure < self._min_confidence()
            ):
                break

            produced.append((token, sure))

            if self.on_choice:
                self.on_choice(token, self._chances(ranked))

            if self.on_speak:
                self.on_speak(token, sure)
            self.experience([token], learn=False)
            recent.add(token)

        return produced

    # ==============================================================
    # РАЗМЫШЛЕНИЕ: внутренняя речь + проверяющий
    # ==============================================================

    def _associations(self, token: str, cache: dict) -> dict[str, int]:
        """Места всех слов в ожидании после слова token (по выученным синапсам)."""

        if token not in cache:
            prototype = self.semantic.prototype(token)
            drive = self.brain.learned_drive(self._full(prototype), "association")
            drive = drive.masked_fill(prototype > 0, 0.0)
            ranked = self.semantic.decode(drive, {token}, top=len(self.semantic))
            cache[token] = {word: place + 1 for place, (word, _) in enumerate(ranked)}

        return cache[token]

    def consistency(self, thought: str, goal: list[str], cache: dict) -> float:
        """
        Проверяющий: связана ли мысль с тем, о чём речь?
        Сила связи в обе стороны (слово цели -> мысль и мысль -> слово
        цели) по выученным синапсам, среднее двух сильнейших. 1.0 —
        мысль первая в ожидании, 0 — не связана.
        """

        links = sorted(
            (
                max(
                    1.0 / self._associations(word, cache).get(thought, 10**6),
                    1.0 / self._associations(thought, cache).get(word, 10**6),
                )
                for word in goal
                if word != thought and word in self.semantic
            ),
            reverse=True,
        )[:2]

        return sum(links) / len(links) if links else 0.0

    def think(self) -> tuple[list[tuple[str, float]], list[dict]]:
        """
        Размышление про себя.

            генератор: ожидание коры -> кандидат
            проверяющий: связан ли кандидат с целью (вопросом)?
              ✗ -> в состояние не попадает, генератор берёт следующий
              ✓ -> звучит про себя (Брока -> слух, без вывода наружу),
                   меняет состояние, из него рождается следующая мысль
            остановка: уверенность упала или ветки кончились

        Возвращает итог (слова с уверенностью) и ход мыслей.
        """

        if self.last_rates is None or self.last_token not in self.semantic:
            return [], []

        cfg = self.cfg
        goal = [token for token in self.heard_in_phrase if self.familiar(token)]
        cache: dict = {}
        recent = set(self.heard_in_phrase)
        rejected: set[str] = set()
        produced: list[tuple[str, float]] = []
        log: list[dict] = []

        for _ in range(cfg.think_max_steps):
            ranked = self.semantic.decode(
                self._with_recall(self.expectation(self.last_rates)), exclude=recent | rejected | {END}, top=20
            )

            if not ranked:
                break

            margin = Confidence.margin(ranked)
            sure = self.confidence(margin)
            known = self.confidence.experience >= cfg.confidence_min_experience

            if margin < cfg.generate_min_margin or (known and sure < self._min_confidence()):
                log.append({"thought": ranked[0][0], "verdict": "stop", "score": round(sure, 3)})
                break

            accepted = None

            for thought, _ in ranked[: cfg.think_breadth]:
                score = self.consistency(thought, goal + [t for t, _ in produced], cache)
                ok = score >= cfg.think_accept
                log.append({"thought": thought, "verdict": "ok" if ok else "no", "score": round(score, 3)})

                if self.on_think:
                    self.on_think(log[-1])

                if ok:
                    accepted = thought
                    break

                rejected.add(thought)

            if accepted is None:
                break

            # Мысль звучит про себя: слышна мозгу, наружу не выходит.
            self.experience([accepted], learn=False)
            produced.append((accepted, sure))
            recent.add(accepted)

            if len(produced) >= cfg.generate_max_tokens:
                break

        return produced, log

    def generate(self, prompt: str, think: bool | None = None) -> list[tuple[str, float]]:
        """Продолжение без следа (read-only)."""

        tokens = self.tokenize(prompt)

        if not tokens:
            return []

        # От незнакомого слова продолжать не от чего — «не знаю».
        if not self.familiar(tokens[-1]):
            return []

        think = self.cfg.thinking_enabled if think is None else think

        with self.isolated():
            self.experience(tokens, learn=False)
            return self.think()[0] if think else self._continue()

    # ==============================================================
    # READ-ONLY ДИАГНОСТИКА
    # ==============================================================

    class _Isolated:
        def __init__(self, interface: "LanguageInterface"):
            self.interface = interface

        def __enter__(self):
            i = self.interface
            self.saved = (i.last_rates, i.last_token, i.experienced_tokens, i.heard_in_phrase)
            # След памяти и прайминг — тоже с чистого листа и обратно.
            self.traces = (i._memory_trace().clone(), i._priming_trace().clone(), i._place)
            self.topic = (i._topic, i._topic_drive, i._sentence_places)
            i._memory_trace().zero_()
            i._priming_trace().zero_()
            i._place = None
            i._topic, i._topic_drive, i._sentence_places = None, None, []
            self.frozen = i.brain.frozen()
            self.frozen.__enter__()
            # Чистый контекст: пауза тишины, прошлое затухает само.
            i.brain.run(i.cfg.probe_pause_ms)
            i.last_rates, i.last_token, i.heard_in_phrase = None, None, []
            return self

        def __exit__(self, *exc):
            i = self.interface
            self.frozen.__exit__(*exc)
            i.last_rates, i.last_token, i.experienced_tokens, i.heard_in_phrase = self.saved
            memory, priming, i._place = self.traces
            i._memory_trace().copy_(memory)
            i._priming_trace().copy_(priming)
            i._topic, i._topic_drive, i._sentence_places = self.topic
            return False

    def isolated(self) -> "_Isolated":
        """
        Наблюдение не является опытом: веса не меняются,
        readout не обновляется, состояние нейронов
        после выхода возвращается как было.
        """

        return self._Isolated(self)

    def probe(self, text: str) -> dict | None:
        """Состояние мозга после фразы — без следа."""

        tokens = self.tokenize(text)

        if not tokens:
            return None

        with self.isolated():
            self.experience(tokens, learn=False)
            rates = self.last_rates
            expected = self.expectation(rates)

            return {
                "tokens": tokens,
                "rates": rates.clone(),
                "meaning": self._area(rates, self.meaning).clone(),
                "context": self._area(rates, self.context).clone(),
                "expected": expected,
                "ranked": self.semantic.decode(expected, {tokens[-1]}, top=5),
                "spoken": self.mouth.decode(self.speak(expected), {tokens[-1]}, top=5),
            }

    def debug_prediction(self, text: str) -> str:
        result = self.probe(text)

        if result is None:
            return "Формат: /predict слово [слово ...]"

        if not result["ranked"]:
            return "[нет ожидания — контекст ещё ни с чем не связан]"

        lines = [f"после «{' '.join(result['tokens'])}» мозг ожидает:"]
        lines += [f"  {t:20s} {s:.3f}" for t, s in result["ranked"]]

        if result["spoken"]:
            lines.append(
                "рот произнёс бы: "
                + ", ".join(f"{t} ({s:.2f})" for t, s in result["spoken"][:3])
            )

        return "\n".join(lines)

    def compare(self, first: str, second: str) -> str:
        """
        Рабочая память: один и тот же последний токен
        в разных контекстах должен давать разные состояния.

            /compare открыть файл | закрыть файл
        """

        a = self.probe(first)
        b = self.probe(second)

        if a is None or b is None:
            return "Формат: /compare фраза | фраза"

        def top(r):
            return ", ".join(f"{t} ({s:.2f})" for t, s in r["ranked"][:3]) or "—"

        return "\n".join(
            [
                f"A: {' '.join(a['tokens'])}",
                f"B: {' '.join(b['tokens'])}",
                f"сходство смысла (association):      {_cosine(a['meaning'], b['meaning']):.3f}",
                f"сходство контекста (prefrontal):    {_cosine(a['context'], b['context']):.3f}",
                f"сходство ожиданий:                  {_cosine(a['expected'], b['expected']):.3f}",
                f"A ожидает: {top(a)}",
                f"B ожидает: {top(b)}",
                "1.000 = мозг не различает контексты; чем ниже, тем лучше память.",
            ]
        )

    def debug_word(self, word: str) -> str:
        tokens = self.tokenize(word)

        if not tokens:
            return "Формат: /debug слово"

        token = tokens[0]
        result = self.probe(token)
        meaning = result["meaning"]

        lines = [f"токен: {token}"]

        for area in self.brain.areas.values():
            rates = result["rates"]
            active = int((rates[area.E.slice] > 0).sum())
            lines.append(
                f"  {area.name:12s} активно {active:6d} / {area.E.size} E "
                f"({active / area.E.size:.1%})"
            )

        if token in self.semantic:
            size = self.cfg.readout_top_k
            current = set(meaning.topk(size).indices.tolist())
            reference = set(self.semantic.assembly(token, size).tolist())
            lines.append(
                f"совпадение с прошлым опытом: {len(current & reference)}/{size} нейронов"
            )
        else:
            lines.append("токен встречен впервые")

        lines.append(
            "decoder: "
            + ", ".join(f"{t} ({s:.2f})" for t, s in self.semantic.decode(meaning, top=3))
        )

        return "\n".join(lines)

    def debug_association(self, first: str, second: str) -> str:
        """
        Реальные E -> E синапсы ассоциативной коры между
        ансамблями двух слов, обратное направление и фон.
        """

        a_tokens, b_tokens = self.tokenize(first), self.tokenize(second)

        if not a_tokens or not b_tokens:
            return "Формат: /debug_association слово слово"

        a_token, b_token = a_tokens[0], b_tokens[0]

        for token in (a_token, b_token):
            if token not in self.semantic:
                return f"Мозг ещё не встречал «{token}» — ансамбля нет"

        size = self.cfg.readout_top_k // 2
        offset = self.meaning.E.start
        a = set((self.semantic.assembly(a_token, size) + offset).tolist())
        b = set((self.semantic.assembly(b_token, size) + offset).tolist())
        shared = a & b

        def ids(s):
            return torch.tensor(sorted(s), device=self.device, dtype=torch.long)

        proj = self.brain.projection("association: E->E")
        forward = self.brain.weights_between(proj, ids(a - shared), ids(b - shared))
        backward = self.brain.weights_between(proj, ids(b - shared), ids(a - shared))

        def describe(w: torch.Tensor) -> str:
            if not w.numel():
                return "связей нет"

            return (
                f"{w.numel():5d} синапсов, avg {float(w.mean()):.4f}, "
                f"max {float(w.max()):.4f}"
            )

        ranked = self.predict(
            self._full(self.semantic.prototype(a_token)), {a_token}, top=50
        )
        position = next(
            (i + 1 for i, (t, _) in enumerate(ranked) if t == b_token), None
        )

        lines = [
            f"ансамбли: {a_token}={len(a)}  {b_token}={len(b)}  общих={len(shared)}",
            f"{a_token} -> {b_token}: {describe(forward)}",
            f"{b_token} -> {a_token}: {describe(backward)}",
            f"фон E -> E:  {describe(proj.weight.flatten())}",
        ]

        if position is not None:
            lines.append(
                f"ожидание после «{a_token}» (без контекста): "
                f"«{b_token}» на месте {position}"
            )

        return "\n".join(lines)

    def _full(self, meaning: torch.Tensor) -> torch.Tensor:
        full = torch.zeros(self.brain.n, device=self.device)
        full[self.meaning.E.slice] = meaning
        return full

    # ==============================================================
    # STATS / SAVE
    # ==============================================================

    def stats(self) -> str:
        brain = self.brain
        lines = [
            f"- устройство: {brain.device}",
            f"- нейронов: {brain.n:,}, синапсов: {brain.synapse_count():,} "
            f"({brain.memory_bytes() / 2**20:.0f} МБ)",
            f"- известных слов: {len(self.semantic)}",
            f"- прожито токенов: {self.experienced_tokens:,}, реплик диалога: {self.dialog_turns}",
            f"- биологическое время: {brain.time_ms / 1000:.1f} s",
            f"- контекст: фиксированного окна нет — рабочая память затухает "
            f"с τ={self.cfg.memory_tau_syn:.0f} мс, это ~"
            f"{self.cfg.memory_tau_syn / (self.cfg.stimulus_ms + self.cfg.gap_ms):.0f} последних слов",
            f"- структурная пластичность: выросло новых синапсов {brain.synapses_grown:,} "
            f"(вместо отмерших)",
            f"- потеряно спайков (переполнение event-таблицы): {float(brain.dropped):.0f}",
            "- зоны:",
        ]

        for area in brain.areas.values():
            lines.append(
                f"    {area.name:12s} {area.E.size + area.I.size:9,d} нейронов  {area.title}"
            )

        lines.append("- пластичные пути:")

        for proj in brain.plastic:
            w = proj.weight
            learned = int((w > proj.innate_max).sum())
            lines.append(
                f"    {proj.name:28s} {w.numel():11,d} синапсов, "
                f"avg {float(w.mean()):.4f}, выучено (> врождённого) {learned:,}"
            )

        return "\n".join(lines)

    def state_dict(self) -> dict:
        return {
            "semantic": self.semantic.state_dict(),
            "mouth": self.mouth.state_dict(),
            "concepts": self.concepts.state_dict(),
            "memory_places": self.memory.state_dict(),
            "experienced_tokens": self.experienced_tokens,
            "sentences_lived": self.sentences_lived,
            "confidence": self.confidence.state_dict(),
            "emotions": self.emotions.state_dict(),
            "hippocampus": self.hippocampus.state_dict(),
            "dialog": self.dialog,
            "wonder": self.wonder,
            "asked": sorted(self.asked),
            "dialog_turns": self.dialog_turns,
        }

    def load_state_dict(self, state: dict) -> None:
        self.semantic.load_state_dict(state["semantic"])
        self.mouth.load_state_dict(state["mouth"])

        if "concepts" in state:
            self.concepts.load_state_dict(state["concepts"])

        if "memory_places" in state:
            self.memory.load_state_dict(state["memory_places"])
        self.experienced_tokens = state["experienced_tokens"]
        self.dialog_turns = state["dialog_turns"]
        self.sentences_lived = state.get("sentences_lived", 0)

        if "confidence" in state:
            self.confidence.load_state_dict(state["confidence"])

        if "emotions" in state:
            self.emotions.load_state_dict(state["emotions"])

        if "hippocampus" in state:
            self.hippocampus.load_state_dict(state["hippocampus"])

        self.dialog = list(state.get("dialog", []))
        self.wonder = list(state.get("wonder", []))
        self.asked = set(state.get("asked", []))
