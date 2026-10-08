"""
Эмоции — лимбическая система мозга.

Эмоция здесь не задаётся снаружи, а складывается из сигналов,
которые в мозге уже есть (циркумплексная модель Рассела):

    настроение   (valence, -1..1)  похвала / ругань, сбывшиеся и
                                   несбывшиеся ожидания; медленное,
                                   как фон серотонина
    возбуждение  (arousal, 0..1)   выбросы дофамина на неожиданном;
                                   быстрое, как норадреналин
    растерянность (0..1)           низкая выученная уверенность,
                                   ответы «не знаю»

Потребности (драйвы, 0..1) — эмоция как сдвиг потребности:

    голод по новому   растёт в тишине и в монотонности (всё угадывает),
                      утоляется неожиданным → скука
    одобрение         растёт со временем и от ругани, утоляется
                      похвалой → похвала радует сильнее, когда её ждал
    фрустрация        цель заблокирована: был уверен в ответе — а его
                      отругали → злость (а не грусть, как при неуверенном)

Всё затухает к спокойному фону: мозг, которого не трогают,
приходит в равновесие.
"""

from __future__ import annotations

from dataclasses import dataclass


# Скорости затухания на одно событие (слово, ответ).
# [не подбиралось] настроение держится десятки слов,
# возбуждение — несколько слов.
VALENCE_DECAY = 0.97
AROUSAL_DECAY = 0.85
CONFUSION_DECAY = 0.9

# Потребности. [не подбиралось] Голод по новому: +0.03 за угаданное
# слово (монотонность), в тишине до максимума за ~4 минуты; неожиданное
# слово утоляет до 0.3. Одобрение: в тишине до максимума за ~8 минут.
NOVELTY_PER_GUESS = 0.03
NOVELTY_PER_SECOND = 0.004
NOVELTY_SATED_BY_SURPRISE = 0.3
SOCIAL_PER_SECOND = 0.002
FRUSTRATION_DECAY = 0.95


@dataclass
class Mood:
    name: str
    title: str
    face: str


# Рожицы из ASCII: глаза ^ ^ — радость, o o — удивление, - - — скука…
FACES = {
    "радость": Mood("радость", "радуется", r"""
   .-----.
  /  ^ ^  \
 |  \___/  |
  \_______/"""),
    "довольство": Mood("довольство", "доволен", r"""
   .-----.
  /  o o  \
 |   \_/   |
  \_______/"""),
    "любопытство": Mood("любопытство", "любопытно", r"""
   .-----.
  /  O o  \
 |    o    |
  \_______/"""),
    "удивление": Mood("удивление", "удивлён", r"""
   .-----.
  / (O) (O)\
 |    O    |
  \_______/"""),
    "растерянность": Mood("растерянность", "растерян", r"""
   .-----. ?
  /  o O  \
 |   ~~~   |
  \_______/"""),
    "скука": Mood("скука", "скучает", r"""
   .-----.
  /  - -  \
 |   ___   |
  \_______/"""),
    "грусть": Mood("грусть", "грустит", r"""
   .-----.
  /  ; ;  \
 |   /^\   |
  \_______/"""),
    "злость": Mood("злость", "злится", r"""
   .-----.
  /  \ /  \
 |   ^^^   |
  \_______/"""),
    "досада": Mood("досада", "досадует", r"""
   .-----.
  /  > <  \
 |   ###   |
  \_______/"""),
    "сон": Mood("сон", "спит", r"""
   .-----.  z
  /  - -  \ Z
 |   ...   |  z
  \_______/"""),
    "сонливость": Mood("сонливость", "хочет спать", r"""
   .-----.
  /  = =  \ z
 |   ---   |
  \_______/"""),
    "спокойствие": Mood("спокойствие", "спокоен", r"""
   .-----.
  /  . .  \
 |   ---   |
  \_______/"""),
}


class Emotions:
    def __init__(self):
        self.valence = 0.0
        self.arousal = 0.2
        self.confusion = 0.0
        # Доля сбывшихся ожиданий за последние слова (для скуки).
        self.predictability = 0.0
        # Привычный уровень дофамина (к нему мозг привыкает).
        self.dopamine_habit = 0.0
        # Выставляет язык: спит ли и накопилось ли «давление сна».
        self.sleeping = False
        self.sleepy = False

        # Потребности (см. шапку модуля).
        self.novelty_hunger = 0.0
        self.social_need = 0.3
        self.frustration = 0.0
        # Насколько был уверен в последнем ответе (для фрустрации).
        self.last_sure = 0.0

    # ==============================================================
    # СОБЫТИЯ
    # ==============================================================

    def heard(self, dopamine: float, guessed: bool | None) -> None:
        """Услышал слово: dopamine от неожиданности, угадал ли."""

        # Привыкание: возбуждает не сам дофамин, а его превышение над
        # привычным фоном. Иначе при чтении, где неожиданно почти каждое
        # слово, возбуждение упиралось в 1.0 и не значило ничего.
        excess = max(0.0, dopamine - self.dopamine_habit)
        self.dopamine_habit = self.dopamine_habit * 0.95 + dopamine * 0.05
        news = min(1.0, excess / 6.0)
        self.arousal = self.arousal * AROUSAL_DECAY + news * 0.35
        self.confusion *= CONFUSION_DECAY
        self.valence *= VALENCE_DECAY
        self.frustration *= FRUSTRATION_DECAY
        # Новое утоляет голод по новому.
        self.novelty_hunger -= news * NOVELTY_SATED_BY_SURPRISE

        if guessed is not None:
            # Сбывшееся ожидание приятно, несбывшееся — чуть-чуть нет.
            self.valence += 0.04 if guessed else -0.01
            self.predictability = self.predictability * 0.9 + (0.1 if guessed else 0.0)

            # Монотонность: всё угадывает — хочется нового.
            if guessed:
                self.novelty_hunger += NOVELTY_PER_GUESS

        self._clip()

    def feedback(self, amplitude: float) -> None:
        """Похвала (+) или ругань (-) от человека."""

        if amplitude > 0:
            # Похвала радует тем сильнее, чем больше хотелось одобрения.
            self.valence += 0.3 + 0.4 * self.social_need
            self.social_need *= 0.3
            self.frustration *= 0.3
        else:
            # Ругань: был уверен — цель заблокирована (фрустрация, злость);
            # не был — просто огорчение.
            self.valence -= 0.5
            self.social_need += 0.2
            self.frustration += 0.2 + 0.6 * self.last_sure

        self.arousal += 0.3
        self.confusion *= 0.5
        self._clip()

    def answered(self, confidence: list[float]) -> None:
        """Ответил: насколько был уверен (пусто — «не знаю»)."""

        if not confidence:
            self.last_sure = 0.0
            self.confusion = min(1.0, self.confusion + 0.6)
            self.valence -= 0.05
        else:
            sure = sum(confidence) / len(confidence)
            self.last_sure = sure
            self.confusion = self.confusion * 0.5 + max(0.0, 0.3 - sure)
            self.valence += 0.1 * (sure - 0.2)

        self._clip()

    def curious(self) -> None:
        """Услышал незнакомое слово и спрашивает о нём."""

        self.arousal = max(self.arousal, 0.45)
        self.confusion *= 0.3
        self.novelty_hunger *= 0.7
        self.valence += 0.05
        self._clip()

    def relax(self, seconds: float) -> None:
        """В тишине возбуждение и растерянность спадают, настроение — медленнее."""

        self.arousal *= 0.97 ** seconds
        self.confusion *= 0.97 ** seconds
        self.valence *= 0.995 ** seconds
        self.frustration *= 0.97 ** seconds
        # В тишине потребности растут: хочется нового и внимания.
        self.novelty_hunger += NOVELTY_PER_SECOND * seconds
        self.social_need += SOCIAL_PER_SECOND * seconds
        self._clip()

    def _clip(self) -> None:
        self.valence = max(-1.0, min(1.0, self.valence))
        self.arousal = max(0.0, min(1.0, self.arousal))
        self.confusion = max(0.0, min(1.0, self.confusion))

        for drive in ("novelty_hunger", "social_need", "frustration"):
            setattr(self, drive, max(0.0, min(1.0, getattr(self, drive))))

    # ==============================================================
    # ЧТО ОН ЧУВСТВУЕТ
    # ==============================================================

    @property
    def mood(self) -> Mood:
        v, a = self.valence, self.arousal

        if self.sleeping:
            name = "сон"
        elif self.confusion > 0.5:
            name = "растерянность"
        elif self.frustration > 0.5 and a > 0.4 and v < 0:
            name = "злость"
        elif a > 0.65 and abs(v) < 0.3:
            name = "удивление"
        elif v > 0.3:
            name = "радость" if a > 0.4 else "довольство"
        elif v < -0.3:
            name = "досада" if a > 0.4 else "грусть"
        elif a > 0.35:
            name = "любопытство"
        elif self.sleepy and a < 0.4:
            name = "сонливость"
        elif a < 0.3 and self.novelty_hunger > 0.6:
            name = "скука"
        else:
            name = "спокойствие"

        return FACES[name]

    def state(self) -> dict:
        mood = self.mood

        return {
            "name": mood.name,
            "title": mood.title,
            "face": mood.face.strip("\n"),
            "valence": round(self.valence, 3),
            "arousal": round(self.arousal, 3),
            "confusion": round(self.confusion, 3),
            "sleeping": self.sleeping,
            "drives": {
                "novelty": round(self.novelty_hunger, 3),
                "social": round(self.social_need, 3),
                "frustration": round(self.frustration, 3),
            },
        }

    def state_dict(self) -> dict:
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "confusion": self.confusion,
            "predictability": self.predictability,
            "dopamine_habit": self.dopamine_habit,
            "novelty_hunger": self.novelty_hunger,
            "social_need": self.social_need,
            "frustration": self.frustration,
            "last_sure": self.last_sure,
        }

    def load_state_dict(self, state: dict) -> None:
        for key, value in state.items():
            setattr(self, key, value)
