from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import math
import random
import re

import nest

from brain.network import BrainNetwork


State = dict[int, float]


def _cosine(a: State, b: State) -> float:
    if not a or not b:
        return 0.0

    if len(a) > len(b):
        a, b = b, a

    dot = sum(
        value * b.get(gid, 0.0)
        for gid, value in a.items()
    )

    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))

    if not na or not nb:
        return 0.0

    return dot / (na * nb)


# ==================================================================
# SENSORY ENCODER
# ==================================================================


class SensoryEncoder:
    """
    Токен -> sensory spike pattern.

    Одинаковый токен всегда даёт один и тот же набор
    sensory-каналов и одинаковые времена спайков
    (аналог: одинаковый звук -> одинаковая активация улитки).

    Какие нейроны КОРЫ ответят на этот паттерн,
    здесь не решается — это делает сама сеть.
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def encode(self, token: str) -> dict[int, list[float]]:
        rng = random.Random(
            hashlib.blake2b(
                token.encode("utf-8"),
                digest_size=16,
            ).digest()
        )

        channels = rng.sample(
            range(self.cfg.sensory_size),
            self.cfg.sensory_channels_per_token,
        )

        rate_per_ms = self.cfg.sensory_rate_hz / 1000.0

        trains: dict[int, list[float]] = {}

        for channel in channels:
            times = []

            t = rng.uniform(0.0, 1.0 / rate_per_ms)

            while t < self.cfg.language_stimulus_ms:
                times.append(t)
                t += rng.expovariate(rate_per_ms)

            trains[channel] = times

        return trains


# ==================================================================
# DECODER / READOUT
# ==================================================================


class Readout:
    """
    Decoder отвечает только на вопрос:

        «Какой известный sensory/token pattern
         сейчас соответствует этому состоянию мозга?»

    Для каждого токена хранится усреднённый прототип
    состояния коры, которое возникает, когда мозг этот
    токен воспринимает. Это статистический readout,
    а НЕ память: связи между токенами (столица -> франции)
    здесь не хранятся и храниться не могут.
    """

    def __init__(self, rate: float, ensemble_size: int):
        self.rate = rate
        self.ensemble_size = ensemble_size
        self.prototypes: dict[str, State] = {}
        self.observations: dict[str, int] = {}

    def __contains__(self, token: str) -> bool:
        return token in self.prototypes

    def __len__(self) -> int:
        return len(self.prototypes)

    def observe(self, token: str, state: State) -> None:
        if not state:
            return

        self.observations[token] = (
            self.observations.get(token, 0) + 1
        )

        proto = self.prototypes.get(token)

        if proto is None:
            self.prototypes[token] = dict(state)
            return

        keep = 1.0 - self.rate

        for gid in set(proto) | set(state):
            value = (
                keep * proto.get(gid, 0.0)
                + self.rate * state.get(gid, 0.0)
            )

            if value > 1e-3:
                proto[gid] = value
            else:
                proto.pop(gid, None)

    def decode(
        self,
        state: State,
        exclude: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        exclude = exclude or set()

        ranked = [
            (token, _cosine(state, proto))
            for token, proto in self.prototypes.items()
            if token not in exclude
        ]

        ranked.sort(key=lambda item: item[1], reverse=True)

        return ranked

    def assembly(self, token: str) -> tuple[int, ...]:
        """Наиболее характерные нейроны токена (только для отладки)."""

        proto = self.prototypes.get(token)

        if not proto:
            return ()

        ranked = sorted(
            proto.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        return tuple(
            gid for gid, _ in ranked[: self.ensemble_size]
        )


# ==================================================================
# INTERFACE
# ==================================================================


@dataclass
class Step:
    """Один такт восприятия: что пришло, что мозг ожидал, ошибка."""

    token: str
    state: State
    expected: str | None = None
    expected_score: float = 0.0
    error: float | None = None
    dopamine: float = 0.0


@dataclass
class ReadReport:
    tokens: int = 0
    sentences: int = 0
    errors: list[float] = field(default_factory=list)
    hits: int = 0

    def mean_error(self, part: slice = slice(None)) -> float:
        values = self.errors[part]
        return sum(values) / len(values) if values else 0.0


class SpikingLanguageInterface:
    """
    Языковой интерфейс к BrainNetwork.

        text
          ↓
        tokenizer
          ↓
        SensoryEncoder          (стабильный sensory-код)
          ↓
        BrainNetwork            (E/I cortex, STDP + dopamine)
          ↓
        current state
          ├──► recurrent drive  (ожидание следующего состояния,
          │                      вычисляется из реальных E -> E весов)
          │          ↓
          │    prediction error ──► dopamine ──► пластичность
          ↓
        Readout                 (state -> token, только чтение)
          ↓
        text

    Здесь нет question -> answer и нет словаря знаний.
    Последовательность «столица -> франции -> париж»
    может храниться только в синапсах коры.
    """

    def __init__(
        self,
        brain: BrainNetwork,
        cfg,
    ):
        self.brain = brain
        self.cfg = cfg

        self.encoder = SensoryEncoder(cfg)

        self.readout = Readout(
            rate=cfg.readout_rate,
            ensemble_size=cfg.language_ensemble_size,
        )

        # Последнее воспринятое состояние — с него
        # строится ожидание следующего.
        self.last_state: State = {}
        self.last_token: str | None = None

        self.experienced_tokens = 0

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
        return [
            part
            for part in re.split(r"[.!?;\n]+(?:\s|$)|\n", text)
            if part.strip()
        ]

    # ==============================================================
    # PERCEPTION
    # ==============================================================

    def perceive(self, token: str) -> State:
        """
        Подать один токен в мозг и вернуть
        состояние коры во время стимула.
        """

        self.brain.schedule_sensory(
            self.encoder.encode(token)
        )

        state = self.brain.observe(
            self.cfg.language_stimulus_ms
        )

        self.brain.simulate(self.cfg.language_gap_ms)

        self.readout.observe(token, state)

        self.experienced_tokens += 1

        return state

    def rest(self) -> None:
        """
        Тишина между фразами: ожидание обнуляется,
        гомеостаз возвращает синапсы в рабочий диапазон.
        """

        self.brain.simulate(self.cfg.language_rest_ms)

        if self.cfg.synaptic_scaling_enabled:
            self.brain.synaptic_scaling()

        self.last_state = {}
        self.last_token = None

    # ==============================================================
    # PREDICTION
    # ==============================================================

    def expectation(self, state: State) -> State:
        """
        Ожидаемое следующее состояние.

        Это возбуждение, которое текущий ансамбль
        отправляет через E -> E синапсы в нейроны,
        которые сейчас молчат. Сильные выученные связи
        A -> B дают сильное ожидание B.
        """

        drive = self.brain.recurrent_drive(
            state,
            # Врождённые веса не превышают ee_weight_max:
            # всё, что выше, — след опыта.
            baseline=self.cfg.ee_weight_max,
        )

        return {
            gid: value
            for gid, value in drive.items()
            if gid not in state
        }

    def predict(
        self,
        state: State,
        exclude: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        return self.readout.decode(
            self.expectation(state),
            exclude=exclude,
        )

    # ==============================================================
    # EXPERIENCE
    # ==============================================================

    def _step(
        self,
        token: str,
        surprise_learning: bool,
    ) -> Step:
        expected_state: State = {}
        expected: list[tuple[str, float]] = []

        if self.last_state:
            expected_state = self.expectation(
                self.last_state
            )

            expected = self.readout.decode(
                expected_state,
                exclude={self.last_token}
                if self.last_token
                else None,
            )

        state = self.perceive(token)

        step = Step(token=token, state=state)

        if expected:
            step.expected, step.expected_score = expected[0]

        if self.last_state and token in self.readout:
            # prediction vs actual
            step.error = 1.0 - _cosine(
                expected_state,
                self.readout.prototypes[token],
            )

            if surprise_learning:
                step.dopamine = (
                    self.cfg.surprise_dopamine_gain
                    * step.error
                )

                # Неожиданный переход закрепляется:
                # dopamine превращает eligibility traces
                # только что прошедших пар в изменение весов.
                self.brain.reward(step.dopamine)

        self.last_state = state
        self.last_token = token

        return step

    def experience(
        self,
        tokens: list[str],
        surprise_learning: bool = False,
    ) -> list[Step]:
        """Мозг проживает последовательность токенов."""

        return [
            self._step(token, surprise_learning)
            for token in tokens
        ]

    def reward(self, amplitude: float | None = None) -> None:
        """
        Внешняя «печенька».

        Укрепляет то, что было активно недавно
        (eligibility traces живут ~tau_c = 1.5 с
        биологического времени).
        """

        self.brain.reward(amplitude)

        self.brain.simulate(self.cfg.language_rest_ms)

    # ==============================================================
    # LEARN (учебный интерфейс)
    # ==============================================================

    def learn(
        self,
        text: str,
        trials: int | None = None,
    ) -> list[list[Step]]:
        """
        /learn столица франции => париж

        Внутри только:

            столица -> франции -> париж -> reward

        повторённое несколько раз. «=>» — просто
        разделитель для человека, мозг видит один поток.
        """

        tokens = self.tokenize(text.replace("=>", " "))

        if not tokens:
            return []

        trials = max(
            1,
            trials or self.cfg.language_learning_trials,
        )

        history = []

        for _ in range(trials):
            history.append(
                self.experience(tokens)
            )

            self.reward()

            self.rest()

        return history

    # ==============================================================
    # READ (основной способ обучения)
    # ==============================================================

    def read_text(self, text: str) -> ReadReport:
        """
        Мозг просто живёт внутри потока текста.

        Никаких пар «вопрос -> ответ». Пластичность
        регулируется prediction error: неожиданное
        закрепляется сильнее, ожидаемое — слабее.
        """

        report = ReadReport()

        for sentence in self.sentences(text):
            tokens = self.tokenize(sentence)

            if not tokens:
                continue

            for step in self.experience(
                tokens,
                surprise_learning=True,
            ):
                report.tokens += 1

                if step.error is not None:
                    report.errors.append(step.error)

                    if step.expected == step.token:
                        report.hits += 1

            report.sentences += 1

            self.rest()

        return report

    def read_path(
        self,
        path: str,
        passes: int = 1,
    ) -> list[ReadReport]:
        target = Path(path).expanduser()

        if target.is_dir():
            files = sorted(
                p
                for p in target.rglob("*")
                if p.is_file()
                and p.suffix.lower() in {".txt", ".md", ".rst"}
            )
        else:
            files = [target]

        text = "\n".join(
            p.read_text(encoding="utf-8", errors="ignore")
            for p in files
        )

        return [
            self.read_text(text)
            for _ in range(max(1, passes))
        ]

    # ==============================================================
    # GENERATION
    # ==============================================================

    def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
    ) -> list[tuple[str, float]]:
        """
        prompt -> brain state -> ожидание -> decoder
               -> token -> снова в мозг -> ...

        Это продолжение состояния, а не поиск ответа.
        """

        tokens = self.tokenize(prompt)

        if not tokens:
            return []

        self.rest()

        self.experience(tokens)

        produced: list[tuple[str, float]] = []

        recent = list(tokens[-2:])

        for _ in range(
            max_tokens or self.cfg.generate_max_tokens
        ):
            ranked = self.predict(
                self.last_state,
                exclude=set(recent),
            )

            if not ranked:
                break

            token, score = ranked[0]

            if score < self.cfg.generate_min_score:
                break

            produced.append((token, score))

            self.experience([token])

            recent = (recent + [token])[-2:]

        self.rest()

        return produced

    def ask(self, text: str) -> str:
        produced = self.generate(text)

        if not produced:
            return (
                "[мозг не сформировал ожидания — "
                "этот контекст ещё не связан ни с чем]"
            )

        return " ".join(token for token, _ in produced)

    # ==============================================================
    # DEBUG
    # ==============================================================

    def debug_word(self, word: str) -> str:
        tokens = self.tokenize(word)

        if not tokens:
            return "Формат: /debug слово"

        token = tokens[0]

        known = token in self.readout
        reference = self.readout.assembly(token)

        self.rest()

        state = self.perceive(token)

        self.rest()

        if not state:
            return (
                f"{token}: кора не ответила "
                "на sensory pattern"
            )

        current = tuple(
            gid
            for gid, _ in sorted(
                state.items(),
                key=lambda item: item[1],
                reverse=True,
            )[: self.cfg.language_ensemble_size]
        )

        ranked = self.readout.decode(state)[:3]

        lines = [
            f"токен: {token}",
            f"активных E-нейронов: {len(state)} "
            f"из {len(self.brain.living_excitatory())}",
            f"current assembly: {' '.join(map(str, current))}",
        ]

        if known and reference:
            overlap = len(set(current) & set(reference))
            lines.append(
                f"совпадение с прошлым опытом: "
                f"{overlap}/{len(reference)} нейронов"
            )
        else:
            lines.append("токен встречен впервые")

        lines.append(
            "decoder: "
            + ", ".join(f"{t} ({s:.2f})" for t, s in ranked)
        )

        return "\n".join(lines)

    def _weights(self, source, target) -> list[float]:
        connections = self.brain.recurrent_connections(
            source=list(source),
            target=list(target),
        )

        if connections is None:
            return []

        return list(connections.get("weight"))

    def debug_association(self, first: str, second: str) -> str:
        """
        Реальные E -> E синапсы между ансамблями двух токенов.

        Сравниваем с обратным направлением и со средним
        весом по всей коре: ассоциация есть, только если
        A -> B заметно сильнее фона.
        """

        a_tokens = self.tokenize(first)
        b_tokens = self.tokenize(second)

        if not a_tokens or not b_tokens:
            return "Формат: /debug_association слово слово"

        a_token, b_token = a_tokens[0], b_tokens[0]

        for token in (a_token, b_token):
            if token not in self.readout:
                return (
                    f"Мозг ещё не встречал «{token}» — "
                    "ансамбля нет"
                )

        a = set(self.readout.assembly(a_token))
        b = set(self.readout.assembly(b_token))

        shared = a & b
        a_only = a - shared
        b_only = b - shared

        forward = self._weights(a_only, b_only)
        backward = self._weights(b_only, a_only)

        all_connections = self.brain.recurrent_connections()
        background = (
            list(all_connections.get("weight"))
            if all_connections is not None
            else []
        )

        def describe(weights: list[float]) -> str:
            if not weights:
                return "связей нет"

            return (
                f"{len(weights):4d} синапсов, "
                f"avg {sum(weights) / len(weights):.4f}, "
                f"max {max(weights):.4f}, "
                f"sum {sum(weights):.3f}"
            )

        ranked = self.predict(
            self.readout.prototypes[a_token],
            exclude={a_token},
        )

        position = next(
            (i + 1 for i, (t, _) in enumerate(ranked) if t == b_token),
            None,
        )

        lines = [
            f"ансамбли: {a_token}={len(a)}  {b_token}={len(b)}  "
            f"общих нейронов={len(shared)}",
            f"{a_token} -> {b_token}: {describe(forward)}",
            f"{b_token} -> {a_token}: {describe(backward)}",
            f"фон E -> E:       {describe(background)}",
        ]

        if position is not None:
            lines.append(
                f"ожидание после «{a_token}»: "
                f"«{b_token}» на месте {position} из {len(ranked)}"
            )

        return "\n".join(lines)

    def debug_prediction(self, text: str) -> str:
        tokens = self.tokenize(text)

        if not tokens:
            return "Формат: /predict слово [слово ...]"

        self.rest()

        self.experience(tokens)

        ranked = self.predict(
            self.last_state,
            exclude={self.last_token},
        )[:5]

        self.rest()

        if not ranked:
            return "[нет ожидания]"

        return "\n".join(
            [f"после «{' '.join(tokens)}» мозг ожидает:"]
            + [f"  {token:20s} {score:.3f}" for token, score in ranked]
        )

    # ==============================================================
    # STATS
    # ==============================================================

    def stats(self) -> str:
        lines = [
            f"- известных токенов (readout): {len(self.readout)}",
            f"- прожито токенов: {self.experienced_tokens}",
            f"- brain excitatory neurons: "
            f"{len(self.brain.living_excitatory())}",
            f"- brain inhibitory neurons: "
            f"{len(self.brain.living_inhibitory())}",
            f"- sensory channels: {len(self.brain.sensory)}",
            f"- biological time: "
            f"{float(nest.biological_time) / 1000:.1f} s",
        ]

        connections = self.brain.recurrent_connections()

        if connections is not None:
            weights = list(connections.get("weight"))
            strong = sum(1 for w in weights if w > 0.5)
            lines.append(
                f"- E -> E синапсов: {len(weights)}, "
                f"avg {sum(weights) / len(weights):.4f}, "
                f"сильных (>0.5): {strong}"
            )

        if len(self.readout):
            lines.append("\nТокены (readout, не память):")

            for token in sorted(self.readout.prototypes):
                lines.append(
                    f"- {token}: "
                    f"{self.readout.observations[token]} предъявлений"
                )

        return "\n".join(lines)
