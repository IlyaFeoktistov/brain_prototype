from __future__ import annotations

from pathlib import Path
import unicodedata

import nest

from brain.config import BrainConfig
from brain.lifecycle import LifecycleController
from brain.language import SpikingLanguageInterface
from brain.metrics import Metrics
from brain.network import BrainNetwork


_HIDDEN = frozenset(
    " ​‌‍﻿"
)


BANNER = """
╔══════════════════════════════════════════════╗
║              ARTIFICIAL BRAIN               ║
║      NEST + AdEx + STDP + dopamine          ║
╚══════════════════════════════════════════════╝

Опыт:
  /read путь [проходов]       прочитать файл или папку (.txt/.md)
  /read: текст                прочитать текст прямо из строки
  /learn a b c [=> d]         учебный опыт: последовательность + reward
  /reward [сила]              dopamine прямо сейчас («печенька»)

Наблюдение:
  /predict слова              чего мозг ожидает дальше
  /debug слово                текущий нейронный ансамбль слова
  /debug_association a b      реальные E -> E синапсы между ансамблями
  /stats                      состояние коры и readout
  /brain                      состояние мозга

  любой текст                 продолжить его (генерация)
  /help /quit

Пример:
  /learn столица франции => париж
  /debug_association столица франции
  /debug_association франции париж
  столица

Мозг не получает пар «вопрос -> ответ».
Последовательность может храниться только
в весах рекуррентных E -> E синапсов.
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


def _print_brain_state(
    brain: BrainNetwork,
    cfg: BrainConfig,
) -> None:
    print(
        f"neurons={len(brain.alive_ids())}, "
        f"connections={brain.alive_connection_count()}, "
        f"time={nest.biological_time:.0f} ms"
    )
    print(
        f"Dopamine: "
        f"{'ON' if cfg.dopamine_enabled else 'OFF'}"
    )
    print(
        f"Synaptic scaling: "
        f"{'ON' if cfg.synaptic_scaling_enabled else 'OFF'}"
    )
    print(
        f"Structural plasticity: "
        f"{'ON' if cfg.structural_plasticity_enabled else 'OFF'}"
    )
    print(
        f"Death: "
        f"{'ON' if cfg.neuron_death_enabled else 'OFF'}"
    )


def _print_read_report(index: int, report) -> None:
    hit_rate = (
        report.hits / len(report.errors)
        if report.errors
        else 0.0
    )

    print(
        f"  проход {index}: токенов {report.tokens}, "
        f"фраз {report.sentences}, "
        f"prediction error {report.mean_error():.3f}, "
        f"угадано следующих {report.hits}/{len(report.errors)} "
        f"({hit_rate:.0%})"
    )


def _read(
    language: SpikingLanguageInterface,
    payload: str,
) -> None:
    if payload.startswith(":"):
        reports = [language.read_text(payload[1:])]
    else:
        parts = payload.rsplit(" ", 1)
        passes = 1

        if len(parts) == 2 and parts[1].isdigit():
            payload, passes = parts[0], int(parts[1])

        if not Path(payload).expanduser().exists():
            print(f"Нет такого файла или папки: {payload}")
            return

        print("Мозг: читаю...")

        reports = language.read_path(payload, passes)

    for index, report in enumerate(reports, 1):
        _print_read_report(index, report)


def main() -> None:
    cfg = BrainConfig()

    brain = BrainNetwork(cfg)
    brain.setup()

    lifecycle = LifecycleController(
        brain,
        cfg,
    )

    language = SpikingLanguageInterface(
        brain,
        cfg,
    )

    metrics = Metrics()

    # Пока lifecycle и metrics создаются,
    # как и раньше, даже если чат напрямую
    # ими не пользуется.
    _ = lifecycle
    _ = metrics

    print(BANNER)

    _print_brain_state(
        brain,
        cfg,
    )

    while True:
        try:
            line = _clean(
                input("\nТы: ")
            )

        except (
            EOFError,
            KeyboardInterrupt,
        ):
            print()
            break

        if not line:
            continue

        command, _, payload = line.partition(" ")
        payload = payload.strip()

        if command == "/quit":
            break

        if command == "/help":
            print(BANNER)
            continue

        if command == "/stats":
            print(language.stats())
            continue

        if command == "/brain":
            _print_brain_state(
                brain,
                cfg,
            )
            continue

        if command == "/debug":
            if not payload:
                print("Формат: /debug слово")
                continue

            if "=>" in payload:
                first, _, second = payload.partition("=>")
                print(
                    language.debug_association(
                        first,
                        second,
                    )
                )
                continue

            print(language.debug_word(payload))
            continue

        if command == "/debug_association":
            words = payload.replace("=>", " ").split()

            if len(words) != 2:
                print("Формат: /debug_association слово слово")
                continue

            print(
                language.debug_association(
                    words[0],
                    words[1],
                )
            )
            continue

        if command == "/predict":
            print(language.debug_prediction(payload))
            continue

        if command == "/reward":
            try:
                amplitude = (
                    float(payload) if payload else None
                )
            except ValueError:
                print("Формат: /reward [сила]")
                continue

            language.reward(amplitude)

            print(
                "Мозг: dopamine — недавние переходы "
                "закреплены"
            )
            continue

        if command == "/read" or command.startswith("/read:"):
            if command.startswith("/read:"):
                payload = ":" + line[len("/read:"):]

            if not payload:
                print(
                    "Формат: /read путь [проходов] "
                    "или /read: текст"
                )
                continue

            _read(language, payload)
            continue

        if command == "/learn":
            if not language.tokenize(payload):
                print("Формат: /learn слово слово [=> слово]")
                continue

            print(
                "Мозг: проживаю последовательность "
                f"{cfg.language_learning_trials} раз + reward..."
            )

            language.learn(payload)

            print(
                "Мозг: опыт прожит. Проверь "
                "/debug_association и /predict"
            )
            continue

        if command.startswith("/"):
            print("Неизвестная команда. /help — список команд")
            continue

        print(
            "Мозг:",
            language.ask(line),
        )


if __name__ == "__main__":
    main()
