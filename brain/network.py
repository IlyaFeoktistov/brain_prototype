from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import random

import nest


@dataclass
class NeuronRecord:
    gid: int
    excitatory: bool
    born_ms: float
    alive: bool = True
    low_activity_epochs: int = 0


class BrainNetwork:
    """
    Основной нейронный субстрат.

    Архитектура:

        sensory
           │
           ▼
        E (memory population)
        │ ▲
        │ │
        ▼ │
        I ─┘

    sensory -> E, sensory -> I:
        статическая проекция (органы чувств не учатся)

    E -> E:
        dopamine-modulated STDP

    E -> I:
        fast excitation

    I -> E:
        lateral inhibition

    Таким образом язык не имеет собственной output-сети.
    Память находится непосредственно в excitatory population.
    """

    def __init__(self, cfg):
        self.cfg = cfg

        self.records: Dict[int, NeuronRecord] = {}

        self.excitatory = nest.NodeCollection()
        self.inhibitory = nest.NodeCollection()

        self.sensory = nest.NodeCollection()
        self.sensory_generators = nest.NodeCollection()

        self.background = None
        self.activity_recorder = None
        self.window_recorder = None

        self.dopamine_neurons = None
        self.dopamine_parrot = None
        self.dopamine_transmitter = None

        self.excitatory_synapse_model = None

        self.previous_event_counts: Dict[int, int] = {}

    # ==============================================================
    # SETUP
    # ==============================================================

    def setup(self) -> None:
        nest.ResetKernel()

        nest.local_num_threads = self.cfg.threads
        nest.resolution = self.cfg.resolution_ms
        nest.rng_seed = self.cfg.rng_seed
        nest.verbosity = nest.VerbosityLevel.ERROR

        self._create_dopamine_system()
        self._configure_synapses()
        self._create_initial_population()
        self._create_sensory_layer()
        self._create_background_input()
        self._create_activity_recorder()

        if self.cfg.structural_plasticity_enabled:
            nest.EnableStructuralPlasticity()

    # ==============================================================
    # DOPAMINE
    # ==============================================================

    def _create_dopamine_system(self) -> None:
        self.dopamine_neurons = nest.Create(
            "iaf_psc_alpha",
            8,
            params={
                "V_th": -50.0,
                "tau_syn_ex": 2.0,
                "tau_syn_in": 5.0,
            },
        )

        self.dopamine_parrot = nest.Create(
            "parrot_neuron"
        )

        self.dopamine_transmitter = nest.Create(
            "volume_transmitter"
        )

        nest.Connect(
            self.dopamine_neurons,
            self.dopamine_parrot,
            syn_spec={
                "synapse_model": "static_synapse",
                "weight": 1.0,
                "delay": 1.0,
            },
        )

        nest.Connect(
            self.dopamine_parrot,
            self.dopamine_transmitter,
            syn_spec={
                "synapse_model": "static_synapse",
                "weight": 1.0,
                "delay": 1.0,
            },
        )

    # ==============================================================
    # SYNAPSES
    # ==============================================================

    def _configure_synapses(self) -> None:
        """
        Вся долговременная ассоциативная память E-сети
        работает через dopamine-modulated STDP.
        """

        nest.CopyModel(
            "stdp_dopamine_synapse",
            "brain_dopamine",
            {
                "volume_transmitter": self.dopamine_transmitter,

                "weight": 0.20,

                "delay": 1.0,

                "Wmin": 0.0,
                "Wmax": 3.0,

                "A_plus": 0.010,
                "A_minus": 0.004,

                "tau_plus": 30.0,
                "tau_c": self.cfg.dopamine_tau_c,
                "tau_n": 100.0,

                "b": self.cfg.dopamine_baseline,
            },
        )

        self.excitatory_synapse_model = (
            "brain_dopamine"
        )

        # Быстрая тормозная связь.
        nest.CopyModel(
            "static_synapse",
            "brain_inhibitory",
            {
                "weight": self.cfg.inhibitory_weight,
                "delay": 1.0,
            },
        )

        # Structural plasticity использует те же
        # excitatory/inhibitory элементы.
        self.structural_synapses = {
            "excitatory": {
                "synapse_model": self.excitatory_synapse_model,
                "post_synaptic_element": "Den_ex",
                "pre_synaptic_element": "Axon_ex",
            },
            "inhibitory": {
                "synapse_model": "brain_inhibitory",
                "post_synaptic_element": "Den_in",
                "pre_synaptic_element": "Axon_in",
            },
        }

        nest.structural_plasticity_synapses = (
            self.structural_synapses
        )

        nest.structural_plasticity_update_interval = (
            self.cfg.structural_update_interval_ms
        )

    # ==============================================================
    # STRUCTURAL PLASTICITY
    # ==============================================================

    @staticmethod
    def _growth_curve(
        eps: float,
        growth_rate: float,
    ) -> dict:
        return {
            "growth_curve": "gaussian",
            "growth_rate": growth_rate,
            "continuous": False,
            "eta": 0.0,
            "eps": eps,
        }

    def _elements_for(
        self,
        excitatory: bool,
    ) -> dict:
        if excitatory:
            ee = self._growth_curve(
                self.cfg.ca_target_excitatory,
                0.0001,
            )

            return {
                "Den_ex": ee,
                "Den_in": ee,
                "Axon_ex": ee,
            }

        ii = self._growth_curve(
            self.cfg.ca_target_inhibitory,
            0.0003,
        )

        return {
            "Den_ex": ii,
            "Den_in": ii,
            "Axon_in": ii,
        }

    def _neuron_params(
        self,
        excitatory: bool,
    ) -> dict:
        return {
            "synaptic_elements": self._elements_for(
                excitatory
            )
        }

    # ==============================================================
    # MAIN POPULATION
    # ==============================================================

    def _create_initial_population(self) -> None:
        e = nest.Create(
            self.cfg.excitatory_neuron_model,
            self.cfg.initial_excitatory,
            params=self._neuron_params(True),
        )

        i = nest.Create(
            self.cfg.inhibitory_neuron_model,
            self.cfg.initial_inhibitory,
            params=self._neuron_params(False),
        )

        self.excitatory = e
        self.inhibitory = i

        for gid in e.tolist():
            self.records[gid] = NeuronRecord(
                gid=gid,
                excitatory=True,
                born_ms=0.0,
            )

        for gid in i.tolist():
            self.records[gid] = NeuronRecord(
                gid=gid,
                excitatory=False,
                born_ms=0.0,
            )

        # ----------------------------------------------------------
        # E -> E
        #
        # Редкая начальная сеть.
        # Сильные связи должны появляться обучением.
        # ----------------------------------------------------------

        nest.Connect(
            e,
            e,
            conn_spec={
                "rule": "pairwise_bernoulli",
                "p": 0.05,
                "allow_autapses": False,
            },
            syn_spec={
                "synapse_model": self.excitatory_synapse_model,
                "weight": nest.random.uniform(
                    min=self.cfg.ee_weight_min,
                    max=self.cfg.ee_weight_max,
                ),
                "delay": 1.0,
            },
        )

        # ----------------------------------------------------------
        # E -> I
        # ----------------------------------------------------------

        nest.Connect(
            e,
            i,
            conn_spec={
                "rule": "pairwise_bernoulli",
                "p": 0.10,
            },
            syn_spec={
                "synapse_model": "static_synapse",
                "weight": nest.random.uniform(
                    min=0.5,
                    max=1.0,
                ),
                "delay": 1.0,
            },
        )

        # ----------------------------------------------------------
        # I -> E
        #
        # Это конкуренция.
        # ----------------------------------------------------------

        nest.Connect(
            i,
            e,
            conn_spec={
                "rule": "pairwise_bernoulli",
                "p": 0.15,
            },
            syn_spec={
                "synapse_model": "brain_inhibitory",
                "weight": self.cfg.inhibitory_weight,
                "delay": 1.0,
            },
        )

    # ==============================================================
    # SENSORY LAYER
    # ==============================================================

    def _create_sensory_layer(self) -> None:
        """
        Органы чувств.

        Каждый sensory-канал — parrot_neuron со своим
        spike_generator. Parrot повторяет каждый входной спайк,
        поэтому sensory-код доходит до коры без потерь
        и без подбора порогов.

        Проекция sensory -> E статическая:
        одинаковый стимул каждый раз приходит
        через один и тот же путь.
        Пластичность живёт только внутри коры (E -> E).
        """

        self.sensory = nest.Create(
            "parrot_neuron",
            self.cfg.sensory_size,
        )

        self.sensory_generators = nest.Create(
            "spike_generator",
            self.cfg.sensory_size,
        )

        nest.Connect(
            self.sensory_generators,
            self.sensory,
            conn_spec="one_to_one",
            syn_spec={
                "synapse_model": "static_synapse",
                "weight": 1.0,
                "delay": 1.0,
            },
        )

        # Один sensory neuron видит часть cortex.
        #
        # Это sparse projection, а не полноценный output layer.
        nest.Connect(
            self.sensory,
            self.excitatory,
            conn_spec={
                "rule": "pairwise_bernoulli",
                "p": self.cfg.sensory_p,
            },
            syn_spec={
                "synapse_model": "static_synapse",
                "weight": self.cfg.input_weight,
                "delay": 1.0,
            },
        )

        # Feedforward inhibition:
        # сильный стимул сразу будит и тормозные клетки,
        # поэтому побеждает компактный ансамбль,
        # а не вся кора.
        nest.Connect(
            self.sensory,
            self.inhibitory,
            conn_spec={
                "rule": "pairwise_bernoulli",
                "p": self.cfg.sensory_p,
            },
            syn_spec={
                "synapse_model": "static_synapse",
                "weight": self.cfg.input_weight
                * self.cfg.sensory_inhibitory_gain,
                "delay": 1.0,
            },
        )

    # ==============================================================
    # BACKGROUND
    # ==============================================================

    def _create_background_input(self) -> None:
        self.background = nest.Create(
            "poisson_generator",
            params={
                "rate": self.cfg.background_rate_hz,
            },
        )

        nest.Connect(
            self.background,
            self.excitatory,
            syn_spec={
                "weight": self.cfg.input_weight * 0.10,
                "delay": 1.0,
            },
        )

        nest.Connect(
            self.background,
            self.inhibitory,
            syn_spec={
                "weight": self.cfg.input_weight * 0.10,
                "delay": 1.0,
            },
        )

    # ==============================================================
    # RECORDER
    # ==============================================================

    def _create_activity_recorder(self) -> None:
        self.activity_recorder = nest.Create(
            "spike_recorder",
            params={
                "record_to": "memory",
            },
        )

        # Отдельный рекордер для коротких окон наблюдения.
        # Его очищают перед каждым окном, поэтому чтение
        # не зависит от того, сколько мозг уже прожил.
        self.window_recorder = nest.Create(
            "spike_recorder",
            params={
                "record_to": "memory",
            },
        )

        self._connect_to_recorder(
            self.excitatory
        )

        self._connect_to_recorder(
            self.inhibitory
        )

    def _connect_to_recorder(self, nodes) -> None:
        if len(nodes):
            nest.Connect(
                nodes,
                self.activity_recorder,
            )

            nest.Connect(
                nodes,
                self.window_recorder,
            )

    # ==============================================================
    # SENSORY ACCESS
    # ==============================================================

    def schedule_sensory(
        self,
        trains: Dict[int, list[float]],
    ) -> None:
        """
        Внешний мир -> sensory spikes.

        trains: индекс sensory-канала -> времена спайков
        относительно текущего момента (мс).

        Никакого teacher output здесь нет.
        """

        now = float(
            nest.biological_time
        )

        # Спайк, отправленный в момент t, приходит
        # в parrot через delay; планируем строго в будущее.
        offset = now + self.cfg.resolution_ms

        for index, times in trains.items():
            if not times:
                continue

            generator = self.sensory_generators[
                index
            ]

            generator.set(
                {
                    "spike_times": [
                        round(offset + t, 1)
                        for t in sorted(times)
                    ],
                }
            )

    def observe(
        self,
        duration_ms: float,
    ) -> Dict[int, float]:
        """
        Прожить duration_ms и вернуть частоты (Hz)
        excitatory нейронов в этом окне.
        """

        self.window_recorder.n_events = 0

        self.simulate(duration_ms)

        senders = self.window_recorder.events[
            "senders"
        ]

        counts: Dict[int, int] = {}

        for gid in senders:
            gid = int(gid)

            record = self.records.get(gid)

            if record is None:
                continue

            if not record.alive or not record.excitatory:
                continue

            counts[gid] = counts.get(gid, 0) + 1

        duration_s = max(
            0.001,
            duration_ms / 1000.0,
        )

        return {
            gid: count / duration_s
            for gid, count in counts.items()
        }

    # ==============================================================
    # FIRING RATES
    # ==============================================================

    def firing_rates(
        self, start_ms: float, end_ms: float
    ) -> Dict[int, float]:
        """
        Возвращает частоту spikes (Hz) для каждого E-нейрона
        в заданном временном окне.
        """
        activity = self.excitatory_activity(
            start_ms, end_ms
        )

        duration_s = max(
            0.001, (end_ms - start_ms) / 1000.0
        )

        return {
            gid: count / duration_s
            for gid, count in activity.items()
        }

    # ==============================================================
    # NODE ACCESS
    # ==============================================================

    def alive_ids(self) -> list[int]:
        return [
            gid
            for gid, record in self.records.items()
            if record.alive
        ]

    def alive_nodes(self):
        ids = self.alive_ids()

        if not ids:
            return nest.NodeCollection()

        return nest.NodeCollection(
            sorted(ids)
        )

    def living_excitatory(self):
        ids = [
            record.gid
            for record in self.records.values()
            if record.alive
            and record.excitatory
        ]

        if not ids:
            return nest.NodeCollection()

        return nest.NodeCollection(
            sorted(ids)
        )

    def living_inhibitory(self):
        ids = [
            record.gid
            for record in self.records.values()
            if record.alive
            and not record.excitatory
        ]

        if not ids:
            return nest.NodeCollection()

        return nest.NodeCollection(
            sorted(ids)
        )

    # ==============================================================
    # ACTIVITY
    # ==============================================================

    def activity_between(
        self,
        start_ms: float,
        end_ms: float,
    ) -> Dict[int, int]:
        events = self.activity_recorder.events

        senders = events.get(
            "senders",
            [],
        )

        times = events.get(
            "times",
            [],
        )

        result: Dict[int, int] = {}

        for gid, time in zip(
            senders,
            times,
        ):
            gid = int(gid)
            time = float(time)

            if time < start_ms:
                continue

            if time > end_ms:
                continue

            result[gid] = (
                result.get(gid, 0) + 1
            )

        return result

    def excitatory_activity(
        self,
        start_ms: float,
        end_ms: float,
    ) -> Dict[int, int]:
        activity = self.activity_between(
            start_ms,
            end_ms,
        )

        result = {}

        for gid, count in activity.items():
            record = self.records.get(gid)

            if record is None:
                continue

            if not record.alive:
                continue

            if not record.excitatory:
                continue

            result[gid] = count

        return result

    def winner_neurons(
        self,
        start_ms: float,
        end_ms: float,
        count: int,
    ) -> list[int]:
        """
        Возвращает самые активные E-нейроны.

        Это не teacher forcing.
        Winner определяется исключительно
        фактической активностью мозга.
        """

        activity = self.excitatory_activity(
            start_ms,
            end_ms,
        )

        ranked = sorted(
            activity.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            gid
            for gid, spikes in ranked[:count]
            if spikes > 0
        ]

    # ==============================================================
    # SIMULATION
    # ==============================================================

    def simulate(
        self,
        duration_ms: float,
    ) -> None:
        if duration_ms <= 0:
            return

        nest.Simulate(
            float(duration_ms)
        )

    # ==============================================================
    # REWARD
    # ==============================================================

    def reward(
        self,
        amplitude: float | None = None,
    ) -> None:
        """
        Dopamine-импульс.

        stdp_dopamine_synapse превращает накопленные
        eligibility traces в изменение весов
        пропорционально концентрации dopamine.
        Концентрация растёт на 1/tau_n с каждым dopamine-спайком,
        поэтому амплитуда — это число спайков в залпе.
        """

        if not self.cfg.dopamine_enabled:
            return

        if amplitude is None:
            amplitude = (
                self.cfg.dopamine_reward_amplitude
            )

        spikes = int(round(amplitude))

        if spikes <= 0:
            return

        now = float(
            nest.biological_time
        )

        generator = nest.Create(
            "spike_generator",
            params={
                "spike_times": [
                    round(now + 1.0 + i * 1.0, 1)
                    for i in range(spikes)
                ],
            },
        )

        nest.Connect(
            generator,
            self.dopamine_parrot,
            syn_spec={
                "synapse_model": "static_synapse",
                "weight": 1.0,
                "delay": 1.0,
            },
        )

    # ==============================================================
    # CONNECTION INSPECTION
    # ==============================================================

    def recurrent_connections(
        self,
        source: list[int] | None = None,
        target: list[int] | None = None,
    ):
        """
        Пластичные E -> E синапсы (вся долговременная
        ассоциативная память коры).
        """

        kwargs = {
            "synapse_model": self.excitatory_synapse_model,
        }

        if source is not None:
            if not source:
                return None

            kwargs["source"] = nest.NodeCollection(
                sorted(set(source))
            )

        if target is not None:
            if not target:
                return None

            kwargs["target"] = nest.NodeCollection(
                sorted(set(target))
            )

        connections = nest.GetConnections(**kwargs)

        if not len(connections):
            return None

        return connections

    def recurrent_drive(
        self,
        rates: Dict[int, float],
        baseline: float = 0.0,
    ) -> Dict[int, float]:
        """
        Куда текущее состояние толкает кору
        через выученные E -> E синапсы.

            drive[post] = sum(rate[pre] * max(0, w(pre -> post) - baseline))

        Это и есть ожидание следующего состояния:
        оно целиком определяется весами сети.
        baseline отсекает врождённый случайный фон,
        оставляя только то, что выучено.
        """

        active = [
            gid
            for gid, rate in rates.items()
            if rate > 0
        ]

        connections = self.recurrent_connections(
            source=active,
        )

        if connections is None:
            return {}

        data = connections.get(
            ["source", "target", "weight"]
        )

        drive: Dict[int, float] = {}

        for pre, post, weight in zip(
            data["source"],
            data["target"],
            data["weight"],
        ):
            weight -= baseline

            if weight <= 0:
                continue

            drive[post] = (
                drive.get(post, 0.0)
                + rates[pre] * weight
            )

        return drive

    # ==============================================================
    # HOMEOSTASIS: SYNAPTIC SCALING
    # ==============================================================

    def synaptic_scaling(self) -> int:
        """
        Homeostatic synaptic scaling (Turrigiano).

        Суммарный входной E -> E вес каждого нейрона
        не может расти бесконечно: если он превысил
        бюджет, все входящие синапсы нейрона
        мультипликативно уменьшаются.

        Без этого нейроны, отвечающие сразу на многие
        стимулы, становятся «хабами» и размывают
        все ассоциации.

        Возвращает число нейронов, которые пришлось масштабировать.
        """

        connections = self.recurrent_connections()

        if connections is None:
            return 0

        data = connections.get(["target", "weight"])

        targets = data["target"]
        weights = data["weight"]

        total: Dict[int, float] = {}
        fan_in: Dict[int, int] = {}

        for post, weight in zip(targets, weights):
            total[post] = total.get(post, 0.0) + weight
            fan_in[post] = fan_in.get(post, 0) + 1

        innate = (
            self.cfg.ee_weight_min
            + self.cfg.ee_weight_max
        ) / 2.0

        factor: Dict[int, float] = {}

        for post, value in total.items():
            budget = (
                fan_in[post]
                * innate
                * self.cfg.synaptic_scaling_budget
            )

            if value > budget:
                factor[post] = budget / value

        if not factor:
            return 0

        connections.set(
            weight=[
                weight * factor.get(post, 1.0)
                for post, weight in zip(targets, weights)
            ]
        )

        return len(factor)

    def alive_connection_count(self) -> int:
        nodes = self.alive_nodes()

        if not len(nodes):
            return 0

        return len(
            nest.GetConnections(
                source=nodes,
                target=nodes,
            )
        )

    # ==============================================================
    # CALCIUM
    # ==============================================================

    def mean_calcium(self) -> tuple[float, float]:
        def read(nodes):
            if not len(nodes):
                return 0.0

            try:
                values = nodes.Ca
            except Exception:
                return 0.0

            if not values:
                return 0.0

            return float(
                sum(values) / len(values)
            )

        return (
            read(self.living_excitatory()),
            read(self.living_inhibitory()),
        )

    # ==============================================================
    # EPOCH RATES
    # ==============================================================

    def neuron_rates(
        self,
        epoch_ms: float,
    ) -> Dict[int, float]:
        events = self.activity_recorder.events

        senders = events.get(
            "senders",
            [],
        )

        current: Dict[int, int] = {}

        for gid in senders:
            gid = int(gid)

            current[gid] = (
                current.get(gid, 0) + 1
            )

        rates = {}

        for gid in self.alive_ids():
            delta = max(
                0,
                current.get(gid, 0)
                - self.previous_event_counts.get(
                    gid,
                    0,
                ),
            )

            rates[gid] = (
                delta
                / (epoch_ms / 1000.0)
            )

        self.previous_event_counts = current

        return rates

    # ==============================================================
    # NEUROGENESIS
    # ==============================================================

    def add_neurons(
        self,
        count: int,
        now_ms: float,
    ) -> list[int]:
        count = min(
            count,
            max(
                0,
                self.cfg.max_total_neurons
                - len(self.alive_ids()),
            ),
        )

        born = []

        for _ in range(count):
            excitatory = (
                random.random()
                < self.cfg.excitatory_fraction
            )

            model = (
                self.cfg.excitatory_neuron_model
                if excitatory
                else self.cfg.inhibitory_neuron_model
            )

            nodes = nest.Create(
                model,
                params=self._neuron_params(
                    excitatory
                ),
            )

            gid = nodes.tolist()[0]

            self.records[gid] = NeuronRecord(
                gid=gid,
                excitatory=excitatory,
                born_ms=now_ms,
            )

            born.append(gid)

            self._connect_to_recorder(
                nodes
            )

            if excitatory:
                self.excitatory = (
                    self.excitatory + nodes
                )
            else:
                self.inhibitory = (
                    self.inhibitory + nodes
                )

        return born

    # ==============================================================
    # NEURON DEATH
    # ==============================================================

    def kill_neuron(
        self,
        gid: int,
    ) -> None:
        record = self.records[gid]

        if not record.alive:
            return

        node = nest.NodeCollection(
            [gid]
        )

        incoming = nest.GetConnections(
            target=node
        )

        outgoing = nest.GetConnections(
            source=node
        )

        if len(incoming):
            incoming.disconnect()

        if len(outgoing):
            outgoing.disconnect()

        node.frozen = True
        record.alive = False