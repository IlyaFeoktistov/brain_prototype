from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import re
import time
from typing import Callable

import torch

from brain.engine import SpikingBrain


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
        raster = self.cache.get(token)

        if raster is not None:
            return raster

        cfg = self.cfg
        seed = int.from_bytes(
            hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(),
            "little",
        ) & 0x7FFFFFFFFFFFFFFF
        rng = torch.Generator().manual_seed(seed)

        steps = int(round(cfg.stimulus_ms / cfg.dt_ms))
        channels = torch.randperm(cfg.sensory_size, generator=rng)[
            : cfg.sensory_channels_per_token
        ]
        p = cfg.sensory_rate_hz * cfg.dt_ms / 1000.0

        raster = torch.zeros(steps, cfg.sensory_size)
        raster[:, channels] = (
            torch.rand(steps, len(channels), generator=rng) < p
        ).float()

        raster = raster.to(self.device)

        if len(self.cache) > 50_000:
            self.cache.clear()

        self.cache[token] = raster

        return raster


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


@dataclass
class Reply:
    """Ответ в диалоге + скорость (как у LLM: prompt и генерация)."""

    text: str
    report: "ReadReport"
    prompt_seconds: float = 0.0
    generated: int = 0
    generate_seconds: float = 0.0


@dataclass
class ReadReport:
    tokens: int = 0
    sentences: int = 0
    errors: list[float] = field(default_factory=list)
    hits: int = 0
    dopamine: float = 0.0

    def mean_error(self) -> float:
        return sum(self.errors) / len(self.errors) if self.errors else 0.0

    def merge(self, other: "ReadReport") -> None:
        self.tokens += other.tokens
        self.sentences += other.sentences
        self.errors += other.errors
        self.hits += other.hits
        self.dopamine += other.dopamine


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

        self.last_rates: torch.Tensor | None = None
        self.last_token: str | None = None

        self.experienced_tokens = 0
        self.sentences_lived = 0
        # Наблюдатель (веб-интерфейс): вызывается после каждого слова.
        self.on_step: Callable[[Step, bool], None] | None = None
        self.dialog_turns = 0

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

    def _area(self, rates: torch.Tensor, area) -> torch.Tensor:
        return rates[area.E.slice]

    # ==============================================================
    # PERCEPTION
    # ==============================================================

    def perceive(self, token: str, learn_forms: bool = True) -> torch.Tensor:
        """Услышать один токен; вернуть частоты всех нейронов."""

        rates = self.brain.observe(
            self.cfg.stimulus_ms,
            self.ear.encode(token),
        )

        self.brain.run(self.cfg.gap_ms)
        self.brain.homeostasis(rates)

        if learn_forms:
            self.semantic.observe(token, self._area(rates, self.meaning))
            self.mouth.observe(token, self._area(rates, self.motor))
            self.experienced_tokens += 1

        return rates

    def rest(self) -> None:
        """
        Тишина между фразами: гомеостаз возвращает синапсы
        в рабочий диапазон. Рабочая память затухает сама.
        """

        self.brain.run(self.cfg.rest_ms)

        if self.cfg.synaptic_scaling_enabled:
            self.brain.synaptic_scaling()

        self.sentences_lived += 1

        if (
            self.cfg.structural_plasticity_enabled
            and self.sentences_lived % self.cfg.rewire_every_sentences == 0
        ):
            self.brain.rewire()

        self.last_rates = None
        self.last_token = None

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

        drive = self.brain.learned_drive(rates, "association")
        active = self._area(rates, self.meaning) > 0

        return drive.masked_fill(active, 0.0)

    def speak(self, expected: torch.Tensor) -> torch.Tensor:
        """Ожидание смысла -> ожидаемый моторный паттерн (рот)."""

        full = torch.zeros(self.brain.n, device=self.device)
        full[self.meaning.E.slice] = expected
        proj = self.brain.projection("association -> motor")

        got = full.index_select(0, proj.pre.flatten()).view_as(proj.weight)

        return (got * proj.weight).sum(1)

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
                top=1,
            )

            if ranked:
                step.expected, step.expected_score = ranked[0]

        rates = self.perceive(token, learn_forms=learn)

        if expected_state is not None and token in self.semantic:
            step.error = 1.0 - _cosine(
                expected_state, self.semantic.prototype(token)
            )

            if learn:
                step.dopamine = self.surprise_dopamine(
                    step.error, hit=step.expected == token
                )
                # Неожиданный переход закрепляется: dopamine
                # превращает eligibility traces только что
                # прошедших пар в изменение весов.
                self.brain.reward(step.dopamine)

        self.last_rates = rates
        self.last_token = token

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
    ) -> ReadReport:
        """
        Мозг живёт внутри потока текста. Пластичность
        регулируется prediction error.
        """

        report = ReadReport()

        for sentence in self.sentences(text):
            if stop is not None and stop.is_set():
                break

            tokens = self.tokenize(sentence)

            if not tokens:
                continue

            for step in self.experience(tokens):
                report.tokens += 1
                report.dopamine += step.dopamine

                if step.error is not None:
                    report.errors.append(step.error)
                    report.hits += step.expected == step.token

            report.sentences += 1
            self.rest()

            if progress and report.sentences % 20 == 0:
                progress(report)

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

        amplitude = self.feedback(text)

        if amplitude is not None:
            self.brain.reward(amplitude)
            self.brain.run(self.cfg.rest_ms)
            word = "закрепил" if amplitude > 0 else "ослабил"
            return Reply(f"[dopamine {amplitude:+.0f}: {word} последний ответ]", ReadReport())

        report = ReadReport()
        sentences = [s for s in self.sentences(text) if self.tokenize(s)]
        start = time.perf_counter()

        for i, sentence in enumerate(sentences):
            for step in self.experience(self.tokenize(sentence)):
                report.tokens += 1
                report.dopamine += step.dopamine

                if step.error is not None:
                    report.errors.append(step.error)
                    report.hits += step.expected == step.token

            report.sentences += 1

            # Между фразами — пауза, но после последней мозг
            # сразу продолжает мысль.
            if i < len(sentences) - 1:
                self.rest()

        self._sync()
        prompt_seconds = time.perf_counter() - start

        start = time.perf_counter()
        produced = self._continue()
        self._sync()

        return Reply(
            " ".join(token for token, _ in produced) or "...",
            report,
            prompt_seconds=prompt_seconds,
            generated=len(produced),
            generate_seconds=time.perf_counter() - start,
        )

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def _continue(self, max_tokens: int | None = None) -> list[tuple[str, float]]:
        """
        state -> ожидание -> рот -> слово -> снова в ухо -> ...

        Собственная речь слышна мозгу (обратная связь через ухо),
        но не даёт surprise-dopamine.
        """

        if self.last_rates is None:
            return []

        produced: list[tuple[str, float]] = []
        recent = [self.last_token] if self.last_token else []

        for _ in range(max_tokens or self.cfg.generate_max_tokens):
            expected = self.expectation(self.last_rates)

            # Слово выбирает ассоциативная кора; моторная зона
            # пока только отражает речь (см. «рот произнёс бы»).
            ranked = self.semantic.decode(expected, exclude=set(recent), top=1)

            if not ranked:
                break

            token, score = ranked[0]

            if score < self.cfg.generate_min_score:
                break

            produced.append((token, score))
            self.experience([token], learn=False)
            recent = (recent + [token])[-3:]

        return produced

    def generate(self, prompt: str) -> list[tuple[str, float]]:
        """Продолжение без следа (read-only)."""

        tokens = self.tokenize(prompt)

        if not tokens:
            return []

        with self.isolated():
            self.experience(tokens, learn=False)
            return self._continue()

    # ==============================================================
    # READ-ONLY ДИАГНОСТИКА
    # ==============================================================

    class _Isolated:
        def __init__(self, interface: "LanguageInterface"):
            self.interface = interface

        def __enter__(self):
            i = self.interface
            self.saved = (i.last_rates, i.last_token, i.experienced_tokens)
            self.frozen = i.brain.frozen()
            self.frozen.__enter__()
            # Чистый контекст: пауза тишины, прошлое затухает само.
            i.brain.run(i.cfg.probe_pause_ms)
            i.last_rates, i.last_token = None, None
            return self

        def __exit__(self, *exc):
            i = self.interface
            self.frozen.__exit__(*exc)
            i.last_rates, i.last_token, i.experienced_tokens = self.saved
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
            "experienced_tokens": self.experienced_tokens,
            "sentences_lived": self.sentences_lived,
            "dialog_turns": self.dialog_turns,
        }

    def load_state_dict(self, state: dict) -> None:
        self.semantic.load_state_dict(state["semantic"])
        self.mouth.load_state_dict(state["mouth"])
        self.experienced_tokens = state["experienced_tokens"]
        self.dialog_turns = state["dialog_turns"]
        self.sentences_lived = state.get("sentences_lived", 0)
