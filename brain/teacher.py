"""
Учитель — часть мира, а не мозга: что показать на экране, куда вести
указку, что похвалить (вкус). Что увидеть, куда посмотреть, что нажать и что
выучить — решает мозг (brain/session.py, веса меняются сами).

Уроки: азбука, лепет руки, копирование букв, курс data/course (азбука →
слоги → слова → фразы → предложения → факты → вопросы), чтение текста с
указкой. Любой урок прерывается session.halt.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

ALPHABET = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
VOWELS = set("аеёиоуыэюя")
COURSE = Path(__file__).resolve().parent.parent / "data" / "course"


def syllables(word: str) -> list[str]:
    """Слоги — как их показывает учитель: открытый слог до гласной."""

    out, cur = [], ""

    for i, c in enumerate(word):
        cur += c

        if c in VOWELS and any(x in VOWELS for x in word[i + 1:]):
            out.append(cur)
            cur = ""

    return out + ([cur] if cur else []) if out else [word]


def clean(line: str) -> str:
    """Строка для экрана: строчные буквы и знаки, которые видит глаз."""

    line = re.sub(r"[^а-яё ,.?!-]", " ", line.lower())
    return re.sub(r"\s+", " ", line).strip()


def course(first: int = 1, last: int = 99) -> list[tuple[str, str, list[str]]]:
    """Файлы курса по номерам: (имя, шаг показа, куски)."""

    out = []

    for path in sorted(COURSE.glob("[0-9]*.txt")):
        number = int(path.name.split("-")[0])

        if not first <= number <= last:
            continue

        lines = path.read_text(encoding="utf-8").splitlines()
        step = lines[0].split(":", 1)[1].strip()
        body = [line for line in lines if line.strip() and not line.startswith("#")]

        if step == "qa":
            items = body
        elif step == "letter":
            items = [clean(line) for line in body]
        else:
            # «буква: слово, слово» (03) и «слог слог» (02) — по одному куску.
            items = [clean(x) for line in body for x in re.split(r"[,\s]+", line.split(":", 1)[-1]) if clean(x)] \
                if step == "syllable" else [clean(line) for line in body]

        out.append((path.name, step, items))

    return out


class Teacher:
    def __init__(self, session, seed: int = 0, dwell: float = 300.0, report=print):
        self.session = session
        self.rng = random.Random(seed)
        self.dwell = dwell
        # report(text) — ход урока (стенд — печать, веб — лента).
        self.report = report

    def point(self, text: str, step: str) -> None:
        """Указка под строкой: куски по step (letter, syllable, word), на каждом dwell мс."""

        s = self.session
        s.line, s.gaze = text, 0.0
        x = 0.0

        for word in text.split(" "):
            parts = list(word) if step == "letter" else syllables(word) if step == "syllable" else [word]

            for part in parts:
                if s.halt:
                    break
                width = s.screen.width(part)
                s.pointer = x + width / 2
                s.live(self.dwell)
                x += width

            x += s.screen.width(" ")

        s.pointer = None

    def letters(self, n: int, alphabet: str = "") -> None:
        """Азбука: буква на экране, указка под ней."""

        shown = alphabet or ALPHABET

        for i in range(n):
            if self.session.halt:
                break
            self.point(self.rng.choice(shown), "letter")
            self.session.live(200.0)

            if i % 10 == 9:
                self.report(f"азбука: {i + 1}/{n} букв")

    def babble(self, n: int) -> str:
        """Лепет: пустой экран; что рука нажала — видно, указка у символа."""

        s = self.session
        s.line, s.gaze = "", 0.0
        typed = ""

        for _ in range(n):
            if s.halt:
                break
            new = s.live(500.0)
            typed += new

            if new:
                s.pointer = s.screen.width(s.line) - 3

        s.pointer = None
        self.report(f"лепет: напечатано {typed!r}")
        return typed

    def copy(self, n: int) -> int:
        """Учитель пишет букву, мозг печатает: совпало — сладкое, нет — горькое."""

        s, right = self.session, 0

        for i in range(n):
            if s.halt:
                break
            target = self.rng.choice(ALPHABET)
            answer = s.reply(target + " ", max_ms=3000.0, silence_ms=800.0)
            good = answer[:1] == target
            right += good
            s.praise(good)

            if i % 20 == 19:
                self.report(f"копирование: верно {right}/{i + 1}, последний «{target}» → {answer!r}")

        return right

    def course(self, laps: int = 1, first: int = 1, last: int = 99) -> None:
        """Курс data/course по номерам: шаг показа — из заголовка файла."""

        s = self.session

        for lap in range(laps):
            for name, step, items in course(first, last):
                right = 0

                for item in items:
                    if s.halt:
                        return

                    if step == "qa":
                        question, answer = item.split("|")[:2]
                        reply = s.reply(clean(question), max_ms=10000.0)
                        good = answer.strip().split()[-1][:4] in reply
                        right += good
                        s.praise(good)
                    else:
                        self.point(item, step)
                        s.live(200.0)

                tail = f", верно {right}/{len(items)}" if step == "qa" else ""
                self.report(f"курс {lap + 1}/{laps}: {name} ({step}, {len(items)}){tail}")

    def read(self, text: str, step: str = "word") -> None:
        """Прочитать текст с указкой: строка за строкой."""

        lines = [clean(line) for line in text.splitlines() if clean(line)]

        for i, line in enumerate(lines):
            if self.session.halt:
                break
            self.point(line, step)
            self.session.live(200.0)

            if i % 10 == 9:
                self.report(f"чтение: {i + 1}/{len(lines)} строк")
