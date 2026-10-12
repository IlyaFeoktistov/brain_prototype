from __future__ import annotations

import queue
import threading
import time
import unicodedata

from brain.commands import HELP, Session
from brain.config import BrainConfig
from brain.live import Liveliness
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


_streamed = False
_last_face = ""
# Строка прогресса без перевода строки (перезаписывается через \r).
_progress = False


def _out(text: str, kind: str) -> None:
    global _streamed, _last_face, _progress

    # Прогресс кончился — следующий вывод с новой строки.
    if _progress and kind != "progress":
        print()
    _progress = kind == "progress"

    if kind == "thought":
        thought, verdict, score = text.split("\t")
        mark = {"ok": "✓", "no": "✗", "stop": "…"}[verdict]
        print(f"  [думаю] {thought} {mark} (связь {float(score):.2f})")
    elif kind == "token":
        token, _, _ = text.partition("\t")
        print(("Мозг: " if not _streamed else " ") + token, end="", flush=True)
        _streamed = True
    elif kind == "face":
        # Рожица печатается, только когда эмоция сменилась.
        if text != _last_face:
            print(text)
            _last_face = text
    elif kind == "brain":
        # Если ответ уже напечатан по словам — только перевод строки.
        print() if _streamed else print("Мозг:", text)
        _streamed = False
    elif kind == "progress":
        print("\r  " + text, end="", flush=True)
    else:
        print("  " + text.replace("\n", "\n  ") if kind == "info" and text.startswith("[") else text)


PROMPT = "\nТы: "


def _aside(text: str) -> None:
    """Сказать в тишине, пока человек, может быть, печатает."""

    print("\r" + text, end=PROMPT, flush=True)


def _idle(session: Session, live: Liveliness, idle_since: float, sleep_left: int, lines: queue.Queue) -> int:
    """Секунда тишины: эмоции успокаиваются, мозг думает, говорит сам или спит."""

    mind = session.mind
    language = mind.language

    if mind.cfg.emotions_enabled:
        language.pass_time(1.0)

    if not sleep_left and live.should_speak_up(mind, idle_since, True, False):
        live.spoke_up_at = time.time()
        said = language.initiative()

        if said:
            reason, text = said
            language.dialog.append({"who": "мозг", "text": text, "initiative": reason})
            _aside(f"Мозг ({reason}): {text}")
    elif not sleep_left and live.should_wander(mind, idle_since, False):
        live.wandered_at = time.time()
        thought = language.wander()

        if thought:
            _aside(f"  (думает: {thought[0]})")

    if not sleep_left and live.should_fall_asleep(mind, idle_since, False):
        sleep_left = language.sleep_pressure * mind.cfg.replays_per_episode
        _aside(f"Засыпаю: новых впечатлений {language.sleep_pressure}, сон на {sleep_left} проигрываний")

    # Спит порциями, чтобы сообщение будило за секунду-две.
    if sleep_left and lines.empty():
        portion = min(mind.cfg.sleep_batch, sleep_left)
        language.sleep(portion)
        sleep_left -= portion
        # Между порциями мозг остаётся спящим.
        language.emotions.sleeping = sleep_left > 0

        if not sleep_left:
            language.emotions.sleepy = False
            _aside("Выспался: новые впечатления закреплены")
            session.autosave(force=True)

    return sleep_left


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

    lines: queue.Queue = queue.Queue()
    # Ввод в своём потоке: пока человек молчит, мозг живёт (думает,
    # заговаривает, засыпает). Приглашение — только когда ответ напечатан.
    can_type = threading.Event()

    def listen() -> None:
        while True:
            can_type.wait()
            can_type.clear()
            try:
                lines.put(_clean(input(PROMPT)))
            except (EOFError, KeyboardInterrupt):
                lines.put(None)
                return

    threading.Thread(target=listen, daemon=True).start()
    can_type.set()
    live = Liveliness()
    idle_since = time.time()
    sleep_left = 0

    while True:
        try:
            line = lines.get(timeout=1.0)
        except queue.Empty:
            try:
                sleep_left = _idle(session, live, idle_since, sleep_left, lines)
            except KeyboardInterrupt:
                print()
                break
            continue
        except KeyboardInterrupt:
            print()
            break

        if line is None:
            print()
            break

        # Сообщение будит; недоспанное останется давлением сна.
        if sleep_left:
            sleep_left = 0
            session.mind.language.emotions.sleeping = False
            print("Проснулся (разбудили)")

        try:
            if not session.handle(line):
                break
        except KeyboardInterrupt:
            session.stop.set()
            print("\nОстановлено. Прочитанное уже в синапсах.")

        idle_since = time.time()
        can_type.set()

    if session.mind.cfg.autosave:
        print(f"Мозг сохранён: {session.mind.save()}")


if __name__ == "__main__":
    main()
