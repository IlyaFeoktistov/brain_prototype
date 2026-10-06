from dataclasses import dataclass


@dataclass
class BrainConfig:
    resolution_ms: float = 0.1
    epoch_ms: float = 500.0
    epochs: int = 60

    # Research-inspired neuron dynamics.
    # Clopath requires aeif_psc_delta_clopath or hh_psc_alpha_clopath.
    excitatory_neuron_model: str = "aeif_psc_delta_clopath"
    inhibitory_neuron_model: str = "aeif_psc_delta"
    use_clopath: bool = True

    initial_excitatory: int = 800
    initial_inhibitory: int = 200
    excitatory_fraction: float = 0.80

    # False = neurons can be born but never die.
    neuron_death_enabled: bool = False
    neurogenesis_enabled: bool = True
    max_total_neurons: int = 3000
    max_births_per_epoch: int = 8
    birth_rate_threshold_hz: float = 0.3

    death_rate_threshold_hz: float = 0.05
    min_age_ms_before_death: float = 5000.0
    min_connections_before_death: int = 1
    death_patience_epochs: int = 8

    structural_plasticity_enabled: bool = True
    structural_update_interval_ms: float = 1000.0

    ca_target_excitatory: float = 0.05
    ca_target_inhibitory: float = 0.20

    # Общая активность мозга.
    # 8000 Hz здесь слишком агрессивно для delta-синапса
    # с весом порядка mV.
    background_rate_hz: float = 300.0

    # Рекуррентное возбуждение
    excitatory_weight: float = 0.5

    # I -> E
    inhibitory_weight: float = -2.5

    # Сила sensory -> brain
    input_weight: float = 4.0

    # Clopath
    clopath_wmax: float = 3.0
    clopath_wmin: float = 0.0

    # Dopamine neuromodulation.
    dopamine_enabled: bool = True
    dopamine_baseline: float = 0.0
    dopamine_reward_amplitude: float = 2.0

    rng_seed: int = 42

    # Text/SNN interface.
    language_input_size: int = 32
    language_output_size: int = 128
    language_stimulus_ms: float = 100.0
    language_rest_ms: float = 50.0
    language_learning_trials: int = 10
    language_ensemble_size: int = 12
    language_output_threshold_hz: float = 1.0
