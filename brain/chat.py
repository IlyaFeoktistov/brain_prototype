from __future__ import annotations

import unicodedata

import nest

from brain.config import BrainConfig
from brain.lifecycle import LifecycleController
from brain.language import SpikingLanguageInterface
from brain.metrics import Metrics
from brain.network import BrainNetwork


_HIDDEN = frozenset(
    "\u00a0\u200b\u200c\u200d\ufeff"
)


BANNER = """
╔══════════════════════════════════════════════╗
║              ARTIFICIAL BRAIN               ║
║      NEST + AdEx + Clopath + dopamine       ║
╚══════════════════════════════════════════════╝

Команды:
  /learn вопрос => ответ    показать мозгу опыт
  /stats                      состояние нейронной памяти
  /brain                      состояние мозга
  /debug слово               посмотреть нейронный ансамбль
  /debug связь слово => слово проверить связь между ансамблями
  /help                       помощь
  /quit                       выход

Пример:
  /learn столица франции => париж
  /learn столица франции => париж
  /learn столица франции => париж

Мозг не получает готовый ответ.
Он самостоятельно формирует нейронные ансамбли
и укрепляет связи между ними через повторение,
STDP и dopamine.
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
        f"Clopath: {'ON' if cfg.use_clopath else 'OFF'}"
    )
    print(
        f"Dopamine: "
        f"{'ON' if cfg.dopamine_enabled else 'OFF'}"
    )
    print(
        f"Death: "
        f"{'ON' if cfg.neuron_death_enabled else 'OFF'}"
    )


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

        # ============================================================
        # QUIT
        # ============================================================

        if line == "/quit":
            break

        # ============================================================
        # HELP
        # ============================================================

        if line == "/help":
            print(BANNER)
            continue

        # ============================================================
        # STATS
        # ============================================================

        if line == "/stats":
            print(language.stats())
            continue

        # ============================================================
        # BRAIN
        # ============================================================

        if line == "/brain":
            _print_brain_state(
                brain,
                cfg,
            )
            continue

        # ============================================================
        # DEBUG WORD
        # ============================================================

        if line.startswith("/debug "):
            payload = line[7:].strip()

            if not payload:
                print(
                    "Формат: /debug слово"
                )
                continue

            if "=>" in payload:
                first, second = (
                    payload.split(
                        "=>",
                        1,
                    )
                )

                first = first.strip()
                second = second.strip()

                if not first or not second:
                    print(
                        "Формат: "
                        "/debug связь слово => слово"
                    )
                    continue

                language.debug_association(
                    first,
                    second,
                )
                continue

            language.debug_word(
                payload
            )
            continue

        # ============================================================
        # LEARN
        # ============================================================

        if line.startswith("/learn "):
            payload = line[7:]

            if "=>" not in payload:
                print(
                    "Формат: "
                    "/learn вопрос => ответ"
                )
                continue

            question, answer = (
                payload.split(
                    "=>",
                    1,
                )
            )

            question = question.strip()
            answer = answer.strip()

            if not question or not answer:
                print(
                    "Формат: "
                    "/learn вопрос => ответ"
                )
                continue

            print(
                "Мозг: наблюдаю опыт..."
            )

            language.teach(
                question,
                answer,
            )

            print(
                "Мозг: опыт запомнен через "
                "spike-pattern + STDP + dopamine"
            )

            continue

        # ============================================================
        # ASK
        # ============================================================

        print(
            "Мозг:",
            language.ask(line),
        )


if __name__ == "__main__":
    main()