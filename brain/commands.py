"""
Команды мозга — общие для терминального чата и веб-интерфейса.

    session = Session(mind, out=print)
    session.handle("/read train-data.md 2")
    session.handle("Столица Франции Париж.")
"""

from __future__ import annotations

from pathlib import Path
import threading
import time
from typing import Callable

from brain.language import ReadReport, Reply
from brain.mind import Mind


HELP = """
Просто пиши — мозг учится на каждой твоей реплике
(prediction error -> dopamine) и отвечает.

  +  /  да / верно / молодец      поощрить последний ответ
  -  /  нет / неверно             ослабить последний ответ

Загрузить данные:
  /read путь [проходов]       файл или папка, любые текстовые файлы
  /read: текст                прочитать текст прямо из строки

Наблюдение (НЕ меняет мозг):
  /predict слова              чего мозг ожидает дальше
  /continue слова             продолжение фразы
  /compare фраза | фраза      различает ли рабочая память контекст
  /debug слово                ансамбль слова по зонам
  /debug_association a b      E -> E синапсы между ансамблями
  /stats                      зоны, синапсы, что выучено

  /save [путь]  /load путь    сохранить / загрузить мозг
  /help /quit                 (на выходе мозг сохраняется сам)
""".strip("\n")


def report_line(report: ReadReport, elapsed: float | None = None) -> str:
    total = len(report.errors)
    hit_rate = report.hits / total if total else 0.0
    line = (
        f"токенов {report.tokens}, фраз {report.sentences}, "
        f"prediction error {report.mean_error():.3f}, "
        f"угадано следующих {report.hits}/{total} ({hit_rate:.0%}), "
        f"dopamine {report.dopamine:.1f}"
    )

    if elapsed:
        line += f", {report.tokens / elapsed:.0f} ток/с"

    return line


def speed_line(reply: Reply) -> str | None:
    report = reply.report

    if not report.tokens:
        return None

    prompt_speed = report.tokens / max(reply.prompt_seconds, 1e-9)
    line = (
        f"[промпт: {report.tokens} ток за {reply.prompt_seconds:.2f} с "
        f"({prompt_speed:.0f} ток/с)"
    )

    if reply.generated:
        gen_speed = reply.generated / max(reply.generate_seconds, 1e-9)
        line += (
            f" · генерация: {reply.generated} ток за {reply.generate_seconds:.2f} с "
            f"({gen_speed:.0f} ток/с)"
        )

    return (
        line
        + f" · угадал следующих {report.hits}/{len(report.errors)}"
        + f" · dopamine {report.dopamine:.1f}]"
    )


class Session:
    """
    Один собеседник мозга.

    out(text, kind) — куда писать: kind = "brain" (ответ мозга),
    "info" (служебное), "progress" (строка прогресса, перезаписывается).
    """

    def __init__(
        self,
        mind: Mind,
        out: Callable[[str, str], None],
        on_reply: Callable[[Reply], None] | None = None,
    ):
        self.mind = mind
        self.out = out
        self.on_reply = on_reply
        self.stop = threading.Event()

    @property
    def language(self):
        return self.mind.language

    # ==============================================================

    def say(self, text: str) -> Reply:
        reply = self.language.converse(text)
        self.out(reply.text, "brain")

        speed = speed_line(reply)

        if speed:
            self.out(speed, "info")

        if self.on_reply:
            self.on_reply(reply)

        return reply

    def handle(self, line: str) -> bool:
        """Выполнить строку. False — пора выходить."""

        line = line.strip()

        if not line:
            return True

        command, _, payload = line.partition(" ")
        payload = payload.strip()

        if not command.startswith("/"):
            self.say(line)
            return True

        if command == "/quit":
            return False

        out = lambda text: self.out(text, "info")
        language = self.language

        if command == "/help":
            out(HELP)
        elif command == "/stats":
            out(language.stats())
        elif command == "/debug":
            out(language.debug_word(payload) if payload else "Формат: /debug слово")
        elif command == "/debug_association":
            words = payload.replace("=>", " ").split()
            out(
                language.debug_association(*words)
                if len(words) == 2
                else "Формат: /debug_association слово слово"
            )
        elif command == "/predict":
            out(language.debug_prediction(payload))
        elif command == "/continue":
            produced = language.generate(payload)
            self.out(" ".join(t for t, _ in produced) or "...", "brain")
        elif command == "/compare":
            first, sep, second = payload.partition("|")
            out(language.compare(first, second) if sep else "Формат: /compare фраза | фраза")
        elif command == "/read" or command.startswith("/read:"):
            if command.startswith("/read:"):
                payload = ":" + line[len("/read:"):]

            if not payload:
                out("Формат: /read путь [проходов] или /read: текст")
            else:
                self.read(payload)
        elif command == "/save":
            out(f"Сохранено: {self.mind.save(payload or None)}")
        elif command == "/load":
            if not payload or not Path(payload).expanduser().exists():
                out("Формат: /load путь_к_файлу.pt")
            else:
                self.mind = Mind.load(payload)
                out(f"Загружено: {payload}")
        else:
            out("Неизвестная команда. /help — список команд")

        return True

    # ==============================================================

    def read_text(self, name: str, text: str) -> ReadReport:
        """Прочитать текст (файл из браузера, строка). Можно остановить."""

        start = time.time()

        def progress(report: ReadReport) -> None:
            self.out(f"{name}: {report_line(report, time.time() - start)}", "progress")

        report = self.language.read_text(text, progress, self.stop)
        self.out(f"{name}: {report_line(report, time.time() - start)}", "info")

        return report

    def read(self, payload: str) -> None:
        self.stop.clear()

        if payload.startswith(":"):
            self.read_text("текст", payload[1:])
            return

        parts = payload.rsplit(" ", 1)
        passes = 1

        if len(parts) == 2 and parts[1].isdigit():
            payload, passes = parts[0], int(parts[1])

        if not Path(payload).expanduser().exists():
            self.out(f"Нет такого файла или папки: {payload}", "info")
            return

        files = self.language.text_files(payload)
        self.out(f"Читаю {len(files)} файл(ов), проходов {passes}.", "info")

        for index in range(1, max(1, passes) + 1):
            total = ReadReport()
            start = time.time()

            for path in files:
                if self.stop.is_set():
                    self.out("Остановлено. Прочитанное уже в синапсах.", "info")
                    return

                text = self.language.load_text(path)

                if text is None:
                    self.out(f"пропуск (не текст): {path}", "info")
                    continue

                total.merge(self.read_text(path.name, text))

                if self.mind.cfg.autosave:
                    self.mind.save()

            self.out(f"проход {index}: {report_line(total, time.time() - start)}", "info")
