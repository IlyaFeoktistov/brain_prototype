from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re

import nest

from brain.network import BrainNetwork


@dataclass
class LearnedResponse:
    question: str
    answer: str
    repetitions: int = 1


class SpikingLanguageInterface:
    """
    Языковой интерфейс к настоящему BrainNetwork.

    ВАЖНО:

    Здесь нет:
        question -> answer

    и нет отдельной output-сети.

    Есть только:

        text
          ↓
        sensory spikes
          ↓
        BrainNetwork.excitatory
          ↓
        emergent neural assembly
          ↓
        decoder

    Decoder нужен только для того, чтобы человек мог увидеть,
    какой словесный label соответствует возникшему ансамблю.
    Само обучение происходит внутри NEST.
    """

    def __init__(
        self,
        brain: BrainNetwork,
        cfg,
    ):
        self.brain = brain
        self.cfg = cfg

        self.sensory = brain.create_sensory_input(
            cfg.language_input_size
        )

        self.sensory_word_connections = {}

        # ----------------------------------------------------------
        # Emergent assemblies.
        #
        # Это READOUT, а не память.
        # ----------------------------------------------------------

        self.word_assemblies: dict[str, tuple[int, ...]] = {}

        self.neuron_words: dict[int, set[str]] = {}

        self.responses: list[LearnedResponse] = []

        self.ensemble_size = max(
            4,
            min(
                cfg.language_ensemble_size,
                32,
            ),
        )

    # ==============================================================
    # TEXT
    # ==============================================================

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(
            r"[^\W_]+(?:[-'][^\W_]+)*",
            text.lower(),
            flags=re.UNICODE,
        )

    # ==============================================================
    # SENSORY CODING
    # ==============================================================

    def _word_seed(self, word: str) -> bytes:
        return hashlib.blake2b(
            word.encode("utf-8"),
            digest_size=16,
        ).digest()

    def _word_sensory_neurons(
        self,
        word: str,
    ) -> list[int]:
        """
        Deterministic sensory representation.

        Это не semantic memory.

        Аналогия:
            одинаковый звук/образ
            каждый раз приходит
            через одинаковый sensory pathway.
        """

        seed = self._word_seed(word)

        neurons = self.sensory.tolist()

        selected: list[int] = []

        for i in range(8):
            value = (
                seed[i]
                + seed[i + 8]
                + i * 31
            )

            index = (
                value
                % len(neurons)
            )

            gid = neurons[index]

            if gid not in selected:
                selected.append(gid)

        return selected

    def _ensure_word_connections(
        self,
        word: str,
    ) -> list[int]:
        if word in self.sensory_word_connections:
            return self.sensory_word_connections[word]

        sensory_ids = (
            self._word_sensory_neurons(word)
        )

        sensory_nodes = nest.NodeCollection(
            sorted(sensory_ids)
        )

        self.brain.connect_sensory(
            sensory_nodes,
            self.brain.living_excitatory(),
            self.cfg.input_weight,
        )

        self.sensory_word_connections[word] = (
            sensory_ids
        )

        return sensory_ids

    # ==============================================================
    # STIMULUS
    # ==============================================================

    def _present_word(
        self,
        word: str,
    ) -> tuple[float, float]:
        """
        Подать одно слово в мозг.

        Возвращает интервал времени,
        в котором нужно анализировать активность.
        """

        sensory_ids = (
            self._ensure_word_connections(word)
        )

        now = float(
            nest.biological_time
        )

        spike_times = []

        for index, _gid in enumerate(
            sensory_ids
        ):
            spike_times.append(
                now
                + 1.0
                + index * 2.0
            )

        generator = nest.Create(
            "spike_generator",
            params={
                "spike_times": spike_times,
            },
        )

        sensory_nodes = nest.NodeCollection(
            sorted(sensory_ids)
        )

        nest.Connect(
            generator,
            sensory_nodes,
            syn_spec={
                "synapse_model": "static_synapse",
                "weight": 10.0,
                "delay": 0.1,
            },
        )

        start = float(
            nest.biological_time
        )

        self.brain.simulate(
            self.cfg.language_stimulus_ms
        )

        end = float(
            nest.biological_time
        )

        self.brain.simulate(
            self.cfg.language_rest_ms
        )

        return start, end

    # ==============================================================
    # EMERGENT ASSEMBLY
    # ==============================================================

    def _find_active_assembly(
        self,
        start_ms: float,
        end_ms: float,
    ) -> tuple[int, ...]:
        """
        Находит реально активировавшийся ансамбль
        среди 800 excitatory нейронов.

        Никакого:
            word -> заранее выбранные neurons.

        Сначала сеть спайкает,
        потом мы смотрим кто победил.
        """

        rates = self.brain.firing_rates(
            start_ms,
            end_ms,
        )

        active = [
            (gid, rate)
            for gid, rate in rates.items()
            if rate >= self.cfg.language_output_threshold_hz
        ]

        active.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        if not active:
            return ()

        winner = [
            gid
            for gid, _rate in active[
                : self.ensemble_size
            ]
        ]

        return tuple(winner)

    def _register_assembly(
        self,
        word: str,
        assembly: tuple[int, ...],
    ) -> None:
        if not assembly:
            return

        existing = self.word_assemblies.get(
            word
        )

        if existing:
            # Если слово уже имеет representation,
            # не создаём новый ensemble.
            return

        self.word_assemblies[word] = assembly

        for gid in assembly:
            self.neuron_words.setdefault(
                gid,
                set(),
            ).add(word)

    # ==============================================================
    # LEARNING
    # ==============================================================

    def _learn_word(
        self,
        word: str,
    ) -> tuple[int, ...]:
        """
        Первое знакомство мозга со словом.

        Мы не выбираем ансамбль заранее.

        Сеть сама отвечает на sensory pattern.
        Первый устойчивый winner становится
        representation этого слова.
        """

        existing = self.word_assemblies.get(
            word
        )

        if existing:
            return existing

        best: tuple[int, ...] = ()

        for _ in range(3):
            start, end = self._present_word(
                word
            )

            assembly = (
                self._find_active_assembly(
                    start,
                    end,
                )
            )

            if len(assembly) > len(best):
                best = assembly

        if best:
            self._register_assembly(
                word,
                best,
            )

        return best

    def _experience(
        self,
        words: list[str],
        reward: bool,
    ) -> list[tuple[int, ...]]:
        """
        Показываем мозгу последовательность событий.

        Например:

            столица
            франции
            париж

        Все три слова проходят через один и тот же
        мозговой субстрат.

        Между соседними состояниями работают
        рекуррентные E -> E синапсы.
        """

        assemblies = []

        for word in words:
            start, end = self._present_word(
                word
            )

            assembly = (
                self._find_active_assembly(
                    start,
                    end,
                )
            )

            if assembly:
                self._register_assembly(
                    word,
                    assembly,
                )

            assemblies.append(
                assembly
            )

        if reward:
            self.brain.reward(
                nest.biological_time,
            )

            # Даём dopamine/STDP немного времени
            # обработать reward.
            self.brain.simulate(
                self.cfg.language_rest_ms
            )

        return assemblies

    def teach(
        self,
        question: str,
        answer: str,
    ) -> None:
        """
        Обучение через повторяющийся опыт.

        ВАЖНО:
        answer здесь не записывается в dictionary
        как память.

        Слова предъявляются сети как последовательность.
        Reward приходит после последовательности.
        """

        question_words = self._tokenize(
            question
        )

        answer_words = self._tokenize(
            answer
        )

        words = (
            question_words
            + answer_words
        )

        if not words:
            return

        # Сначала знакомим мозг с самими словами.
        for word in words:
            self._learn_word(word)

        trials = max(
            1,
            self.cfg.language_learning_trials,
        )

        for _ in range(trials):
            self._experience(
                words,
                reward=True,
            )

        existing = next(
            (
                item
                for item in self.responses
                if item.question == question
                and item.answer == answer
            ),
            None,
        )

        if existing:
            existing.repetitions += 1
        else:
            self.responses.append(
                LearnedResponse(
                    question=question,
                    answer=answer,
                    repetitions=1,
                )
            )

    # ==============================================================
    # RECALL
    # ==============================================================

    def _decode_assembly(
        self,
        assembly: tuple[int, ...],
    ) -> str | None:
        if not assembly:
            return None

        scores: dict[str, float] = {}

        assembly_set = set(assembly)

        for word, neurons in (
            self.word_assemblies.items()
        ):
            overlap = len(
                assembly_set
                & set(neurons)
            )

            if overlap:
                scores[word] = (
                    overlap
                    / max(
                        1,
                        len(neurons),
                    )
                )

        if not scores:
            return None

        return max(
            scores.items(),
            key=lambda item: item[1],
        )[0]

    def _recall_word(
        self,
        word: str,
    ) -> str | None:
        start, end = self._present_word(
            word
        )

        assembly = (
            self._find_active_assembly(
                start,
                end,
            )
        )

        if not assembly:
            return None

        return self._decode_assembly(
            assembly
        )

    def ask(
        self,
        text: str,
    ) -> str:
        words = self._tokenize(text)

        if not words:
            return "[пустой вход]"

        decoded: list[str] = []

        for word in words:
            result = self._recall_word(
                word
            )

            if result is None:
                return (
                    "[SNN не сформировал "
                    "устойчивое состояние]"
                )

            decoded.append(result)

        return " ".join(decoded)

    # ==============================================================
    # DEBUG
    # ==============================================================

    def debug_word(
        self,
        word: str,
    ) -> str:
        word = word.lower().strip()

        assembly = self.word_assemblies.get(
            word
        )

        if not assembly:
            # Пытаемся реально активировать слово.
            start, end = self._present_word(
                word
            )

            assembly = (
                self._find_active_assembly(
                    start,
                    end,
                )
            )

        if not assembly:
            return (
                "[SNN не сформировал "
                "устойчивое состояние]"
            )

        lines = [
            f"слово: {word}",
            f"ensemble: {list(assembly)}",
            f"размер: {len(assembly)}",
        ]

        rates = self.brain.firing_rates(
            max(
                0.0,
                float(nest.biological_time)
                - self.cfg.language_rest_ms
                - self.cfg.language_stimulus_ms,
            ),
            float(nest.biological_time),
        )

        lines.append("активность:")

        for gid in assembly:
            lines.append(
                f"  {gid}: "
                f"{rates.get(gid, 0.0):.2f} Hz"
            )

        return "\n".join(lines)

    def debug_association(
        self,
        first: str,
        second: str,
    ) -> str:
        first = first.lower().strip()
        second = second.lower().strip()

        a = self.word_assemblies.get(first)
        b = self.word_assemblies.get(second)

        if not a:
            return f"Нет ансамбля для слова: {first}"

        if not b:
            return f"Нет ансамбля для слова: {second}"

        connections = self.brain.connections_between(
            list(a),
            list(b),
        )

        if not len(connections):
            return (
                f"{first} -> {second}: "
                "связей нет"
            )

        weights = connections.get(
            "weight"
        )

        if not weights:
            return (
                f"{first} -> {second}: "
                "связи есть, веса недоступны"
            )

        minimum = min(weights)
        maximum = max(weights)
        average = sum(weights) / len(weights)

        return (
            f"{first} -> {second}\n"
            f"connections: {len(weights)}\n"
            f"weight min: {minimum:.4f}\n"
            f"weight max: {maximum:.4f}\n"
            f"weight avg: {average:.4f}"
        )

    # ==============================================================
    # STATS
    # ==============================================================

    def stats(self) -> str:
        free_neurons = (
            len(
                self.brain.living_excitatory()
            )
            - len(self.neuron_words)
        )

        lines = [
            f"- сформированных слов: "
            f"{len(self.word_assemblies)}",
            f"- brain excitatory neurons: "
            f"{len(self.brain.living_excitatory())}",
            f"- brain inhibitory neurons: "
            f"{len(self.brain.living_inhibitory())}",
            f"- sensory neurons: "
            f"{len(self.sensory)}",
            f"- neurons per ensemble: "
            f"{self.ensemble_size}",
            f"- свободных excitatory neurons: "
            f"{max(0, free_neurons)}",
            f"- learning trials: "
            f"{self.cfg.language_learning_trials}",
        ]

        lines.append(
            "\nИзвестные нейронные представления:"
        )

        if not self.word_assemblies:
            lines.append(
                "- [пока нет сформированных ансамблей]"
            )
        else:
            for word, assembly in (
                sorted(
                    self.word_assemblies.items()
                )
            ):
                lines.append(
                    f"- {word}: "
                    f"{list(assembly)}"
                )

        if self.responses:
            lines.append(
                "\nИстория предъявленных опытов:"
            )

            for response in self.responses:
                lines.append(
                    f"- {response.question} "
                    f"-> {response.answer} "
                    f"(повторений: "
                    f"{response.repetitions})"
                )

        return "\n".join(lines)