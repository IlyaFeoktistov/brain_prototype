"""
Мир вне мозга: то, что видят и слышат органы, и то, что меняют действия.
Мозг про мир ничего не знает — только сигналы своих рецепторов.

Экран: текст рисуется шрифтом в пиксели, как строка в мессенджере, —
тёмные буквы на светлом фоне. Клавиатура: клавиша → символ (раскладка —
свойство мира, мозг её не знает и выучивает).
"""

from __future__ import annotations

from functools import lru_cache

import torch

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


class Screen:
    """Строка текста на экране: картинка яркости [rows, cols], 1 — фон, 0 — чернила."""

    def __init__(self, rows: int, cols: int, font_px: int = 12, font: str = FONT):
        self.rows = rows
        self.cols = cols
        self.font_px = font_px
        self.font = font

    @lru_cache(maxsize=50_000)
    def picture(self, text: str, x: int = 1) -> torch.Tensor:
        from PIL import Image, ImageDraw, ImageFont

        font = ImageFont.truetype(self.font, self.font_px)
        image = Image.new("L", (self.cols, self.rows), 255)
        draw = ImageDraw.Draw(image)
        # Строка по вертикали — по середине поля: высота строки шрифта
        # (с выносными элементами) ~ font_px.
        draw.text((x, (self.rows - self.font_px) // 2 - 1), text, font=font, fill=0)
        pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        return pixels.view(self.rows, self.cols).float() / 255.0

    @lru_cache(maxsize=50_000)
    def width(self, text: str) -> float:
        """Ширина строки в пикселях."""

        from PIL import ImageFont

        return ImageFont.truetype(self.font, self.font_px).getlength(text)


# ЙЦУКЕН: ряды клавиш слева направо, сверху вниз, затем пробел.
LAYOUT = "1234567890-" "йцукенгшщзхъ" "фывапролджэ" "ячсмитьбю." "ё,?!:;" " "


class Keyboard:
    """Клавиши по порядку на доске; нажатая клавиша печатает свой символ."""

    def __init__(self, layout: str = LAYOUT):
        self.layout = layout

    def __len__(self) -> int:
        return len(self.layout)

    def type(self, key: int) -> str:
        return self.layout[key]

    def key(self, char: str) -> int | None:
        """Для стендов (проверка снаружи мозга): клавиша символа."""

        i = self.layout.find(char)
        return i if i >= 0 else None


class Voice:
    """
    Голос мира: текст → речь (звуковая волна), как его сказал бы человек.
    Синтезатор речи — piper (голос в ~/.brain/voices); сказанное кэшируется
    файлом в ~/.brain/sounds/<голос>/, повтор — тот же звук. Записи людей и
    свои файлы слышны так же: ухо (body.Cochlea) получает волну, откуда бы
    она ни была (Voice.load).
    """

    def __init__(self, voice: str = "ru_RU-irina-medium"):
        from brain.config import BRAIN_HOME

        self.name = voice
        self.model = BRAIN_HOME / "voices" / f"{voice}.onnx"
        self.cache = BRAIN_HOME / "sounds" / voice
        self.cache.mkdir(parents=True, exist_ok=True)
        self._voice = None

    def say(self, text: str) -> tuple[torch.Tensor, int]:
        """Сказать текст: (волна, частота записи)."""

        import hashlib
        import wave

        # Имя файла — по тексту (кэш); сам текст — рядом, в .txt.
        stem = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
        path = self.cache / f"{stem}.wav"

        if not path.exists():
            if self._voice is None:
                from piper import PiperVoice

                self._voice = PiperVoice.load(str(self.model))

            with wave.open(str(path), "wb") as out:
                self._voice.synthesize_wav(text, out)

            path.with_suffix(".txt").write_text(text, encoding="utf-8")

        return self.load(path)

    @staticmethod
    def load(path) -> tuple[torch.Tensor, int]:
        """Звуковой файл (wav, flac, ogg …) → (волна моно, частота записи)."""

        import soundfile

        data, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
        return torch.from_numpy(data.mean(1)), rate
