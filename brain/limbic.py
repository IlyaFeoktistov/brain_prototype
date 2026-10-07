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
    "досада": Mood("досада", "досадует", r"""
   .-----.
  /  > <  \
 |   ###   |
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

    # ==============================================================
    # СОБЫТИЯ
    # ==============================================================

    def heard(self, dopamine: float, guessed: bool | None) -> None:
        """Услышал слово: dopamine от неожиданности, угадал ли."""

        self.arousal = self.arousal * AROUSAL_DECAY + min(1.0, dopamine / 6.0) * 0.35
        self.confusion *= CONFUSION_DECAY
        self.valence *= VALENCE_DECAY

        if guessed is not None:
            # Сбывшееся ожидание приятно, несбывшееся — чуть-чуть нет.
            self.valence += 0.04 if guessed else -0.01
            self.predictability = self.predictability * 0.9 + (0.1 if guessed else 0.0)

        self._clip()

    def feedback(self, amplitude: float) -> None:
        """Похвала (+) или ругань (-) от человека."""

        self.valence += 0.5 if amplitude > 0 else -0.5
        self.arousal += 0.3
        self.confusion *= 0.5
        self._clip()

    def answered(self, confidence: list[float]) -> None:
        """Ответил: насколько был уверен (пусто — «не знаю»)."""

        if not confidence:
            self.confusion = min(1.0, self.confusion + 0.6)
            self.valence -= 0.05
        else:
            sure = sum(confidence) / len(confidence)
            self.confusion = self.confusion * 0.5 + max(0.0, 0.3 - sure)
            self.valence += 0.1 * (sure - 0.2)

        self._clip()

    def curious(self) -> None:
        """Услышал незнакомое слово и спрашивает о нём."""

        self.arousal = max(self.arousal, 0.45)
        self.confusion *= 0.3
        self.valence += 0.05
        self._clip()

    def _clip(self) -> None:
        self.valence = max(-1.0, min(1.0, self.valence))
        self.arousal = max(0.0, min(1.0, self.arousal))
        self.confusion = max(0.0, min(1.0, self.confusion))

    # ==============================================================
    # ЧТО ОН ЧУВСТВУЕТ
    # ==============================================================

    @property
    def mood(self) -> Mood:
        v, a = self.valence, self.arousal

        if self.confusion > 0.5:
            name = "растерянность"
        elif a > 0.65 and abs(v) < 0.3:
            name = "удивление"
        elif v > 0.3:
            name = "радость" if a > 0.4 else "довольство"
        elif v < -0.3:
            name = "досада" if a > 0.4 else "грусть"
        elif a > 0.35:
            name = "любопытство"
        elif a < 0.12 and self.predictability > 0.5:
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
        }

    def state_dict(self) -> dict:
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "confusion": self.confusion,
            "predictability": self.predictability,
        }

    def load_state_dict(self, state: dict) -> None:
        for key, value in state.items():
            setattr(self, key, value)
