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

        self.background = None
        self.activity_recorder = None

        self.dopamine_neurons = None
        self.dopamine_parrot = None
        self.dopamine_transmitter = None

        self.excitatory_synapse_model = None
        self.sensory_synapse_model = None

        self.previous_event_counts: Dict[int, int] = {}

    # ==============================================================
    # SETUP
    # ==============================================================

    def setup(self) -> None:
        nest.ResetKernel()

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
                "tau_c": 1500.0,
                "tau_n": 100.0,

                "b": self.cfg.dopamine_baseline,
            },
        )

        self.excitatory_synapse_model = (
            "brain_dopamine"
        )

        self.sensory_synapse_model = (
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
                    min=0.05,
                    max=0.20,
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
        self.sensory = nest.Create(
            "iaf_psc_delta",
            self.cfg.language_input_size,
            params={
                "V_th": -55.0,
                "V_reset": -70.0,
                "V_m": -70.0,
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
                "p": 0.12,
            },
            syn_spec={
                "synapse_model": self.sensory_synapse_model,
                "weight": self.cfg.input_weight,
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

    # ==============================================================
    # SENSORY ACCESS
    # ==============================================================

    def create_sensory_input(
        self, size: int
    ) -> nest.NodeCollection:
        """Возвращает sensory слой для языкового интерфейса."""
        return self.sensory

    def connect_sensory(
        self,
        sensory: nest.NodeCollection,
        target: nest.NodeCollection,
        weight: float,
    ) -> None:
        """
        Создаёт синапсы от sensory нейронов к target population.
        Используется языковым интерфейсом для подключения слов.
        """
        if len(sensory) == 0 or len(target) == 0:
            return

        nest.Connect(
            sensory,
            target,
            conn_spec={
                "rule": "pairwise_bernoulli",
                "p": 0.35,
            },
            syn_spec={
                "synapse_model": self.sensory_synapse_model,
                "weight": weight,
                "delay": 1.0,
            },
        )

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
    # SENSORY ACTIVATION
    # ==============================================================

    def stimulate_sensory(
        self,
        neurons: list[int],
        duration_ms: float,
        weight: float = 1.0,
    ) -> tuple[float, float]:
        """
        Активирует sensory pathway.

        Внешний мир → sensory spikes → cortex.

        Никакого teacher output здесь нет.
        """

        if not neurons:
            return (
                float(nest.biological_time),
                float(nest.biological_time),
            )

        now = float(
            nest.biological_time
        )

        sensory = nest.NodeCollection(
            sorted(neurons)
        )

        spike_times = []

        for index, _gid in enumerate(neurons):
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

        nest.Connect(
            generator,
            sensory,
            syn_spec={
                "synapse_model": "static_synapse",
                "weight": weight,
                "delay": 0.1,
            },
        )

        start = float(
            nest.biological_time
        )

        nest.Simulate(
            duration_ms
        )

        end = float(
            nest.biological_time
        )

        return start, end

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
        if not self.cfg.dopamine_enabled:
            return

        if amplitude is None:
            amplitude = (
                self.cfg.dopamine_reward_amplitude
            )

        now = float(
            nest.biological_time
        )

        generator = nest.Create(
            "spike_generator",
            params={
                "spike_times": [
                    now + 1.0
                ],
            },
        )

        nest.Connect(
            generator,
            self.dopamine_parrot,
            syn_spec={
                "synapse_model": "static_synapse",
                "weight": amplitude,
                "delay": 1.0,
            },
        )

    # ==============================================================
    # CONNECTION INSPECTION
    # ==============================================================

    def connections_between(
        self,
        source: list[int],
        target: list[int],
    ):
        if not source or not target:
            return nest.GetConnections()

        return nest.GetConnections(
            source=nest.NodeCollection(
                sorted(source)
            ),
            target=nest.NodeCollection(
                sorted(target)
            ),
        )

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