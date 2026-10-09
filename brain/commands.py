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

from brain.config import FEATURES, PRESETS, BrainConfig
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
  /mood                       что мозг сейчас чувствует
  /sleep [фраз]               уснуть: гиппокамп проиграет коре новые впечатления
  /dreams                     что снилось: фразы, проигранные во сне
  /trace вопрос               стенд: как сигнал идёт к ответу по шагам
  /health                     стенд: частоты зон на слове и в тишине, нарушения
  /reread [реплик]            перечитать переписку (по умолчанию 20 последних)
  /emotions on|off            включить / выключить эмоции
  /think on|off               размышлять про себя перед ответом (ход мыслей виден)
  /forget on|off              свежая память затухает, пока сон её не закрепит
  /live on|off                живой режим (веб): думает в тишине, сам заговаривает
  /initiative on|off          в живом режиме — заговаривать первым
  /feature [имя on|off]       все функции мозга: список / включить / выключить

  /feature имя                что функция делает и когда её трогать

  /brains                     список сохранённых мозгов
  /new имя [размер]           новый мозг (tiny, small, medium, large или число нейронов)
  /save [путь]  /load имя     сохранить / переключиться на другой мозг
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

    if report.naps:
        line += f", поспал {report.naps} раз(а)"

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

    if reply.confidence:
        line += f" · уверенность {100 * sum(reply.confidence) / len(reply.confidence):.0f}%"

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
        self._saved_at = time.time()
        # Если задано (веб-сервер) — /read ставит текст в фоновую очередь
        # и сразу возвращается: можно разговаривать, пока мозг читает.
        self.background_read: Callable[[str, str], int] | None = None
        # Если задано (веб-сервер) — /sleep засыпает порциями, будится.
        self.fall_asleep: Callable[[], str] | None = None

    @property
    def language(self):
        return self.mind.language

    # ==============================================================

    def say(self, text: str) -> Reply:
        # Слова ответа уходят наружу сразу, как мозг их выбрал;
        # мысли — отдельно, как ход размышления.
        self.language.on_speak = lambda token, sure: self.out(f"{token}\t{sure:.2f}", "token")
        self.language.on_think = lambda step: self.out(
            f"{step['thought']}\t{step['verdict']}\t{step['score']}", "thought"
        )

        try:
            reply = self.language.converse(text)
        finally:
            self.language.on_speak = None
            self.language.on_think = None

        self.out(reply.text, "brain")

        if self.mind.cfg.emotions_enabled:
            self.out(self.language.emotions.state()["face"], "face")

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
        elif command == "/trace":
            out(language.trace(payload))
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
        elif command == "/reread":
            turns = int(payload) if payload.isdigit() else 20
            start = time.time()
            report = language.reread(turns, stop=self.stop)
            out(f"Перечитал переписку: {report_line(report, time.time() - start)}")
        elif command == "/sleep":
            if self.fall_asleep and not payload:
                out(self.fall_asleep())
            else:
                self.sleep(int(payload) if payload.isdigit() else None)
        elif command == "/think":
            if payload in ("on", "вкл"):
                self.mind.cfg.thinking_enabled = True
            elif payload in ("off", "выкл"):
                self.mind.cfg.thinking_enabled = False
            out(f"Размышление: {'включено' if self.mind.cfg.thinking_enabled else 'выключено'}")
        elif command == "/forget":
            if payload in ("on", "вкл"):
                self.mind.cfg.forgetting_enabled = True
            elif payload in ("off", "выкл"):
                self.mind.cfg.forgetting_enabled = False
            out(f"Забывание без сна: {'включено' if self.mind.cfg.forgetting_enabled else 'выключено'}")
        elif command == "/live":
            if payload in ("on", "вкл"):
                self.mind.cfg.live_mode = True
            elif payload in ("off", "выкл"):
                self.mind.cfg.live_mode = False
            out(f"Живой режим: {'включён' if self.mind.cfg.live_mode else 'выключен'}")
        elif command == "/initiative":
            if payload in ("on", "вкл"):
                self.mind.cfg.initiative_enabled = True
            elif payload in ("off", "выкл"):
                self.mind.cfg.initiative_enabled = False
            out(f"Заговаривать первым: {'да' if self.mind.cfg.initiative_enabled else 'нет'}")
        elif command == "/feature":
            out(self.feature(payload))
        elif command == "/emotions":
            if payload in ("on", "вкл"):
                self.mind.cfg.emotions_enabled = True
            elif payload in ("off", "выкл"):
                self.mind.cfg.emotions_enabled = False
            out(f"Эмоции: {'включены' if self.mind.cfg.emotions_enabled else 'выключены'}")
        elif command == "/health":
            out(language.health.report())
        elif command == "/dreams":
            dreams = list(language.dreams)
            out("Что снилось (последнее внизу):\n" + "\n".join(f"  {d}" for d in dreams) if dreams else "Ещё не спал")
        elif command == "/mood":
            state = language.emotions.state()
            out(
                f"{state['face']}\n  {state['title']}  "
                f"(настроение {state['valence']:+.2f}, возбуждение {state['arousal']:.2f}, "
                f"растерянность {state['confusion']:.2f})"
            )
        elif command == "/brains":
            out(self.brains_text())
        elif command == "/new":
            out(self.new_brain(payload))
        elif command == "/load":
            target = self.resolve_brain(payload)

            if target is None:
                out("Формат: /load имя_или_путь.pt  (список — /brains)")
            else:
                # Текущий мозг сохраняется, чтобы прогресс не потерять.
                self.autosave(force=True)
                self.mind = Mind.load(str(target))
                out(f"Загружено: {target} — {self.mind.brain.n:,} нейронов")
        else:
            out("Неизвестная команда. /help — список команд")

        return True

    # ==============================================================
    # МОЗГИ
    # ==============================================================

    def brain_files(self) -> list[Path]:
        """Сохранённые мозги: *.pt рядом с текущим и в data/."""

        folders = {Path(self.mind.cfg.save_path).expanduser().resolve().parent, Path("data").resolve()}
        files = {p.resolve() for folder in folders if folder.exists() for p in folder.glob("*.pt")}

        return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)

    def resolve_brain(self, name: str) -> Path | None:
        if not name:
            return None

        path = Path(name).expanduser()

        if path.exists():
            return path

        for candidate in self.brain_files():
            if candidate.name in (name, f"{name}.pt"):
                return candidate

        return None

    def new_brain(self, payload: str) -> str:
        """
        /new имя [размер] — новый мозг рядом с прежними: рождается
        (калибровка), сохраняется в имя.pt и становится текущим. Прежний
        сохраняется, к нему можно вернуться /load. Функции — по умолчанию.
        """

        parts = payload.split()

        if not parts:
            return "Формат: /new имя [tiny|small|medium|large|число нейронов]  (размер — как у текущего)"

        name = parts[0].removesuffix(".pt")
        size = parts[1] if len(parts) > 1 else self.mind.cfg.size

        if not name or Path(name).name != name:
            return f"Имя без папок: {parts[0]}"

        if size not in PRESETS and not size.isdigit():
            return f"Нет такого размера: {size} (tiny, small, medium, large, xlarge или число нейронов)"

        path = Path(self.mind.cfg.save_path).expanduser().parent / f"{name}.pt"

        if path.exists():
            return f"Мозг {path.name} уже есть: /load {name} или другое имя"

        # Текущий сохраняется, чтобы прогресс не потерять.
        self.autosave(force=True)
        cfg = BrainConfig()
        cfg.size = size
        cfg.save_path = str(path)
        cfg.device = self.mind.cfg.device
        mind = Mind(cfg)
        mind.calibrate()
        mind.save()
        self.mind = mind

        return (
            f"Новый мозг: {path} — {mind.brain.n:,} нейронов. Он ничего не знает: "
            "дай ему прочитать текст (/read), потом поспать (/sleep). "
            "Как учить — README «Как обучать лучше всего» или справка в вебе."
        )

    def brains_text(self) -> str:
        current = Path(self.mind.cfg.save_path).expanduser().resolve()
        lines = ["Сохранённые мозги (/load имя):"]

        for path in self.brain_files():
            mark = "●" if path == current else " "
            lines.append(
                f" {mark} {path.name:24s} {path.stat().st_size / 2**20:8.0f} МБ  "
                f"{time.strftime('%d.%m %H:%M', time.localtime(path.stat().st_mtime))}"
            )

        return "\n".join(lines)

    # ==============================================================

    def read_text(self, name: str, text: str) -> ReadReport:
        """Прочитать текст (файл из браузера, строка). Можно остановить."""

        start = time.time()

        def progress(report: ReadReport) -> None:
            self.out(f"{name}: {report_line(report, time.time() - start)}", "progress")

        report = self.language.read_text(text, progress, self.stop)
        self.out(f"{name}: {report_line(report, time.time() - start)}", "info")

        return report

    def sleep(self, replays: int | None = None) -> ReadReport:
        """Сон: проиграть коре новые впечатления (по умолчанию — все по N раз)."""

        language = self.language
        self.stop.clear()
        pressure = language.sleep_pressure
        self.out(f"Засыпаю… новых впечатлений {pressure}, всего в гиппокампе {len(language.hippocampus)}", "info")
        self.out(language.emotions.state()["face"], "face")

        start = time.time()

        def progress(report: ReadReport) -> None:
            self.out(f"сон: {report_line(report, time.time() - start)}", "progress")

        if replays is None:
            report = language.sleep_until_rested(self.stop, progress)
        else:
            report = language.sleep(replays, progress, self.stop)

        self.out(
            f"Проснулся. Проиграно фраз {report.sentences}, "
            f"новых впечатлений осталось {language.sleep_pressure}, "
            f"в гиппокампе {len(language.hippocampus)}",
            "info",
        )
        self.out(language.emotions.state()["face"], "face")
        self.autosave(force=True)

        return report

    def feature(self, payload: str) -> str:
        """
        /feature — список функций; /feature имя — что она делает;
        /feature имя on|off — переключить.
        """

        parts = payload.split()

        if len(parts) == 1:
            if parts[0] not in FEATURES:
                return f"Нет такой функции: {parts[0]} (список — /feature)"

            title, _, about = FEATURES[parts[0]]
            state = "включена" if self.mind.features()[parts[0]] else "выключена"

            return f"{parts[0]} — {title} ({state})\n  {about}"

        if len(parts) == 2 and parts[1] in ("on", "off", "вкл", "выкл"):
            try:
                self.mind.set_feature(parts[0], parts[1] in ("on", "вкл"))
            except KeyError:
                return f"Нет такой функции: {parts[0]} (список — /feature)"

        lines = ["Функции мозга (/feature имя — что делает; /feature имя on|off):"]

        for name, enabled in self.mind.features().items():
            lines.append(f"  {'●' if enabled else '○'} {name:32s} {FEATURES[name][0]}")

        return "\n".join(lines)

    def autosave(self, force: bool = False) -> None:
        """Сохранить мозг, но не чаще раза в autosave_minutes (большой мозг весит ГБ)."""

        cfg = self.mind.cfg
        now = time.time()

        if cfg.autosave and (force or now - self._saved_at >= cfg.autosave_minutes * 60):
            self.mind.save()
            self._saved_at = now

    def read(self, payload: str) -> None:
        self.stop.clear()

        if payload.startswith(":"):
            if self.background_read:
                count = self.background_read("текст", payload[1:])
                self.out(f"Читаю в фоне: {count} фраз. Можно разговаривать.", "info")
            else:
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

        if self.background_read:
            total = 0
            for _ in range(max(1, passes)):
                for path in files:
                    text = self.language.load_text(path)
                    if text is not None:
                        total += self.background_read(path.name, text)
            self.out(f"Читаю в фоне: {len(files)} файл(ов), {total} фраз. Можно разговаривать.", "info")
            return

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
                self.autosave()

            self.out(f"проход {index}: {report_line(total, time.time() - start)}", "info")

        self.autosave(force=True)
