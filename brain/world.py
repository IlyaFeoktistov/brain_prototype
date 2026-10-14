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
