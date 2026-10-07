from dataclasses import dataclass


@dataclass
class BrainConfig:
    resolution_ms: float = 0.1
    # Потоки NEST. NEST не умеет structural plasticity в нескольких
    # потоках: для ускорения (~4x на 8 потоках) поставь
    # threads=8 и structural_plasticity_enabled=False.
    threads: int = 1
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

    # Рекуррентное возбуждение E -> E: врождённые веса
    # (случайные, слабые). Сильные связи должны появляться обучением.
    ee_weight_min: float = 0.05
    ee_weight_max: float = 0.20

    # I -> E
    inhibitory_weight: float = -3.5

    # Сила sensory -> brain.
    # Один sensory-вход не должен сам поджигать нейрон:
    # стреляют клетки, где сходится несколько каналов токена.
    input_weight: float = 5.0

    # Clopath
    clopath_wmax: float = 3.0
    clopath_wmin: float = 0.0

    # Dopamine neuromodulation.
    dopamine_enabled: bool = True
    dopamine_baseline: float = 0.0
    dopamine_reward_amplitude: float = 5.0
    # Время жизни eligibility trace (мс). Dopamine закрепляет только
    # совпадения пре/пост за последние ~tau_c: при 70 мс на токен
    # это несколько последних переходов, а не вся прочитанная фраза.
    dopamine_tau_c: float = 1500.0

    # Homeostatic synaptic scaling: суммарный входной E -> E вес
    # нейрона не больше budget * (врождённая сумма).
    synaptic_scaling_enabled: bool = True
    synaptic_scaling_budget: float = 1.5

    rng_seed: int = 42

    # Sensory layer (органы чувств).
    sensory_size: int = 256
    sensory_p: float = 0.05
    sensory_inhibitory_gain: float = 1.5
    # Сколько sensory-каналов кодирует один токен и с какой частотой.
    sensory_channels_per_token: int = 24
    sensory_rate_hz: float = 150.0

    # Text/SNN interface.
    language_stimulus_ms: float = 60.0
    # Пауза между токенами внутри последовательности.
    # Короткая, чтобы конец A и начало B попадали в окно STDP.
    language_gap_ms: float = 10.0
    # Тишина после последовательности.
    language_rest_ms: float = 150.0
    language_learning_trials: int = 10
    language_ensemble_size: int = 24
    language_output_threshold_hz: float = 1.0
    # Скорость обновления readout-прототипа токена.
    readout_rate: float = 0.3
    # /read: dopamine пропорционален prediction error.
    surprise_dopamine_gain: float = 4.0
    # Максимум токенов, генерируемых за один /generate.
    generate_max_tokens: int = 12
    # Ниже этого сходства ожидание считается шумом.
    generate_min_score: float = 0.3
