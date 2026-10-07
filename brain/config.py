from dataclasses import dataclass, field, asdict
import os


# Размер мозга = общее число нейронов всех зон.
# Число синапсов растёт линейно (fixed in-degree).
# Человек: ~86 млрд нейронов, ~1e14 синапсов — в 6 ГБ GPU
# не помещается ни при каком кодировании.
PRESETS = {
    "tiny": 2_000,
    "small": 20_000,
    "medium": 100_000,
    "large": 500_000,
    "xlarge": 1_000_000,
}


@dataclass
class AreaSpec:
    """
    Зона мозга.

    kind = "cortex"  E (AdEx) + I (быстрый LIF), быстрые синапсы
    kind = "memory"  E с медленным NMDA-подобным током (рабочая память)
    share           доля от общего числа нейронов
    """

    name: str
    title: str
    share: float
    kind: str = "cortex"
    # Локальные связи внутри зоны.
    k_ee: int = 50
    ee_plastic: bool = True
    ee_weight: tuple[float, float] = (0.05, 0.20)
    ee_cap: float = 3.0
    # Целевая доля E-нейронов, отвечающих на слово (разреженный код).
    target_activity: float = 0.04


@dataclass
class PathwaySpec:
    """
    Проводящий путь между зонами: E-нейроны src -> E и I нейроны dst.

    plastic=True — путь учится (STDP + dopamine).
    plastic=False — врождённая «проводка».
    """

    src: str
    dst: str
    k: int
    weight: tuple[float, float]
    plastic: bool = False
    cap: float = 2.0
    # Сколько входов от src получает каждый I-нейрон dst
    # (feedforward inhibition: сильный вход сразу будит торможение).
    k_inhibitory: int = 50
    inhibitory_weight: float = 1.5


def default_areas() -> list[AreaSpec]:
    return [
        AreaSpec(
            "auditory", "слуховая кора / зона Вернике (формы слов)",
            0.20, k_ee=30, ee_plastic=False, target_activity=0.05,
        ),
        AreaSpec(
            "association", "ассоциативная кора (смысловая память)",
            0.50, k_ee=100,
        ),
        AreaSpec(
            "prefrontal", "префронтальная кора (рабочая память)",
            0.20, kind="memory", k_ee=20, ee_plastic=False,
            ee_weight=(0.05, 0.05), target_activity=0.08,
        ),
        AreaSpec(
            "motor", "моторная речевая зона / зона Брока (рот)",
            0.10, k_ee=30, ee_plastic=False,
        ),
    ]


def default_pathways() -> list[PathwaySpec]:
    return [
        # Восприятие: формы слов -> смысл.
        PathwaySpec("auditory", "association", k=100, weight=(4.0, 4.0)),
        # Смысл -> рабочая память (эхо последних слов).
        PathwaySpec("association", "prefrontal", k=100, weight=(6.0, 6.0),
                    inhibitory_weight=6.0),
        # Контекст -> ожидание следующего смысла. Учится.
        PathwaySpec("prefrontal", "association", k=20, weight=(0.02, 0.10),
                    plastic=True, cap=2.0, k_inhibitory=0),
        # Смысл -> артикуляция (рот).
        PathwaySpec("association", "motor", k=100, weight=(6.0, 6.0),
                    inhibitory_weight=6.0),
    ]


@dataclass
class BrainConfig:
    # ------------------------------------------------------------------
    # Железо
    # ------------------------------------------------------------------
    device: str = os.environ.get("BRAIN_DEVICE", "auto")  # auto | cuda | cpu
    seed: int = 42
    dt_ms: float = 1.0
    # Верхняя граница доли нейронов, стреляющих за один шаг
    # (размер event-таблицы). Лишние спайки теряются и
    # считаются в /stats.
    max_spike_fraction: float = 0.05

    # ------------------------------------------------------------------
    # Размер и анатомия.
    # ------------------------------------------------------------------
    size: str = os.environ.get("BRAIN_SIZE", "small")
    # Доля тормозных нейронов в зоне (~20% в неокортексе).
    inhibitory_fraction: float = 0.20
    areas: list[AreaSpec] = field(default_factory=default_areas)
    pathways: list[PathwaySpec] = field(default_factory=default_pathways)

    # Локальные связи с торможением внутри каждой зоны.
    k_ei: int = 50
    k_ie: int = 50
    k_ii: int = 25
    ei_weight: float = 2.0
    ie_weight: float = -4.0
    ii_weight: float = -1.5

    # ------------------------------------------------------------------
    # Нейроны (мВ, мс). Cortex E — AdEx, I — быстрый LIF,
    # memory E — LIF с медленным NMDA-подобным током.
    # ------------------------------------------------------------------
    e_tau_m: float = 20.0
    # AMPA/GABA ~5 мс: при 2 мс зоны срывались в синхронные залпы.
    e_tau_syn: float = 5.0
    e_delta_t: float = 2.0
    e_adapt_tau: float = 150.0
    e_adapt_b: float = 1.0
    i_tau_m: float = 10.0
    # Быстрые интернейроны (PV) ближе к порогу, чем пирамидные
    # клетки: торможение должно успевать раньше залпа E.
    i_excitability: float = 8.0
    i_tau_syn: float = 5.0
    # Время жизни следа в рабочей памяти: при 70 мс на токен
    # 400 мс — это контекст в несколько последних слов.
    memory_tau_syn: float = 400.0
    # Вклад медленного тока относительно быстрого.
    memory_nmda_ratio: float = 0.5
    memory_tau_m: float = 20.0
    v_rest: float = -70.0
    v_threshold: float = -50.0
    v_reset: float = -58.0
    refractory_ms: float = 2.0

    # Гомеостаз возбудимости: после каждого слова возбудимость
    # зоны (ток покоя, мВ) сдвигается к целевой разреженности.
    homeostasis_rate: float = 0.5
    homeostasis_bias_range: tuple[float, float] = (-10.0, 18.0)
    # «Пренатальная» калибровка: столько случайных звуков мозг
    # слышит при рождении, пока гомеостаз находит рабочую точку.
    calibration_tokens: int = 400

    # Фоновый шум: Poisson-вход на каждый нейрон.
    background_rate_hz: float = 1000.0
    # Слабый шум: иначе ответ зоны определяет шум, а не слово.
    background_weight: float = 0.1
    # Рабочей памяти шум нужнее: её нейроны висят у порога на
    # тоническом NMDA-токе и без шума срываются в синхронные залпы.
    memory_background_weight: float = 0.3

    # ------------------------------------------------------------------
    # Пластичность: STDP -> eligibility trace -> dopamine.
    # Вес меняется только когда приходит dopamine.
    # ------------------------------------------------------------------
    # Окно LTP шире окна LTD: конец слова A и начало слова B
    # (между ними ~10-60 мс) должны успевать связаться.
    stdp_tau_plus: float = 40.0
    # Окно сбалансировано с лёгким перевесом депрессии
    # (A-·τ- > A+·τ+, Song/Miller/Abbott 2000): случайная совместная
    # активность не растит веса, растут только причинные пары A -> B.
    stdp_tau_minus: float = 40.0
    stdp_a_plus: float = 0.010
    stdp_a_minus: float = 0.012
    eligibility_tau: float = 1500.0
    dopamine_learning_rate: float = 1.0

    # Homeostatic synaptic scaling: суммарный входной пластичный
    # вес нейрона не больше budget * врождённая сумма.
    synaptic_scaling_enabled: bool = True
    synaptic_scaling_budget: float = 1.5

    # Структурная пластичность: синапс, ослабший ниже
    # prune_fraction × минимальный врождённый вес пути, отмирает,
    # вместо него растёт новый. Проверка раз в N фраз.
    structural_plasticity_enabled: bool = True
    prune_fraction: float = 0.25
    rewire_every_sentences: int = 50

    # ------------------------------------------------------------------
    # Dopamine от prediction error.
    #   предсказание сбылось          -> 0
    #   ошибка ниже порога            -> 0 (мелкая неожиданность)
    #   ошибка выше порога            -> растёт квадратично
    # Поэтому выученное перестаёт перестраивать сеть.
    # ------------------------------------------------------------------
    surprise_threshold: float = 0.3
    surprise_dopamine_gain: float = 6.0
    # Врождённые сигналы из диалога: поощрение и наказание.
    praise_words: tuple[str, ...] = ("+", "да", "верно", "правильно", "молодец", "отлично")
    blame_words: tuple[str, ...] = ("-", "нет", "неверно", "неправильно", "ошибка")
    feedback_dopamine: float = 6.0

    # ------------------------------------------------------------------
    # Ухо: текст -> sensory-каналы -> слуховая кора.
    # ------------------------------------------------------------------
    sensory_size: int = 2048
    sensory_channels_per_token: int = 32
    sensory_rate_hz: float = 150.0
    k_sensory: int = 20
    sensory_weight: float = 3.0
    sensory_inhibitory_gain: float = 1.0

    # ------------------------------------------------------------------
    # Язык.
    # ------------------------------------------------------------------
    stimulus_ms: float = 60.0
    gap_ms: float = 10.0
    rest_ms: float = 150.0
    # Пауза перед диагностической пробой: рабочая память
    # затухает (~2.5 τ), проба видит только свои слова.
    probe_pause_ms: float = 1000.0
    readout_rate: float = 0.3
    readout_top_k: int = 64
    generate_max_tokens: int = 12
    generate_min_score: float = 0.08

    # Сохранение мозга между запусками.
    save_path: str = os.environ.get("BRAIN_SAVE", "data/brain.pt")
    autosave: bool = True

    # Нейрогенез и смерть нейронов (этап 13 roadmap) пока не
    # реализованы: сначала стабильная память и контекст.

    @property
    def total_neurons(self) -> int:
        return PRESETS.get(self.size) or int(self.size)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "BrainConfig":
        data = dict(data)
        data["areas"] = [AreaSpec(**a) for a in data.get("areas", [])] or default_areas()
        data["pathways"] = [PathwaySpec(**p) for p in data.get("pathways", [])] or default_pathways()

        for key in ("praise_words", "blame_words", "homeostasis_bias_range"):
            if key in data:
                data[key] = tuple(data[key])

        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in data.items() if k in known})
