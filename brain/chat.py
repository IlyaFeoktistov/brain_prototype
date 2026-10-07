from __future__ import annotations

import unicodedata

from brain.commands import HELP, Session
from brain.config import BrainConfig
from brain.mind import Mind


_HIDDEN = frozenset(
    " ​‌‍﻿"
)


BANNER = """
╔══════════════════════════════════════════════╗
║              ARTIFICIAL BRAIN               ║
║    GPU SNN · AdEx · STDP + dopamine         ║
║    зоны: слух · смысл · рабочая память · рот ║
╚══════════════════════════════════════════════╝
"""


def _clean(text: str) -> str:
    """Убрать невидимые символы и нормализовать пробелы."""
    out: list[str] = []

    for ch in text:
        if ch in _HIDDEN:
            continue

        if unicodedata.category(ch) == "Zs" and ch != " ":
            continue

        out.append(ch)

    return "".join(out).strip()


def _out(text: str, kind: str) -> None:
    if kind == "brain":
        print("Мозг:", text)
    elif kind == "progress":
        print("\r  " + text, end="", flush=True)
    else:
        print("  " + text.replace("\n", "\n  ") if kind == "info" and text.startswith("[") else text)


def main() -> None:
    cfg = BrainConfig()

    print("Мозг: просыпаюсь...")
    mind, restored = Mind.open(cfg)
    session = Session(mind, _out)

    print(BANNER)
    print(HELP)
    print(
        ("\nЗагружен сохранённый мозг: " if restored else "\nНовый мозг: ")
        + mind.cfg.save_path
    )
    print(mind.language.stats())

    while True:
        try:
            line = _clean(input("\nТы: "))
        except (EOFError, KeyboardInterrupt):
            print()
            break

        try:
            if not session.handle(line):
                break
        except KeyboardInterrupt:
            session.stop.set()
            print("\nОстановлено. Прочитанное уже в синапсах.")

    if session.mind.cfg.autosave:
        print(f"Мозг сохранён: {session.mind.save()}")


if __name__ == "__main__":
    main()
