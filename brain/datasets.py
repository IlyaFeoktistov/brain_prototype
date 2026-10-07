"""
Диалоговые данные для обучения: вопрос -> ответ, запрос -> выполнение.

    python -m brain.datasets            # -> data/corpus/qa/*.txt

Мозг между предложениями делает паузу (rest), и переход через «?» или
«.» не учится. Поэтому вопрос и ответ пишутся ОДНОЙ фразой:

    какая столица франции — париж

Тогда переход «вопрос -> ответ» учится так же, как «франции -> париж»,
а в чате мозг продолжает мысль после вопроса ответом.

Источники (открытые, HuggingFace datasets-server, без регистрации):
    kuznetsoffandrey/sberquad   вопросы по тексту с короткими ответами
    d0rj/alpaca-cleaned-ru      инструкция -> выполнение (только короткие)
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys
import time
import urllib.parse
import urllib.request


ROWS_URL = "https://datasets-server.huggingface.co/rows"
PAGE = 100


def rows(dataset: str, config: str, split: str, limit: int):
    for offset in range(0, limit, PAGE):
        query = urllib.parse.urlencode(
            {"dataset": dataset, "config": config, "split": split, "offset": offset, "length": PAGE}
        )

        for attempt in range(4):
            try:
                with urllib.request.urlopen(f"{ROWS_URL}?{query}", timeout=60) as response:
                    page = json.load(response)["rows"]
                break
            except Exception:  # noqa: BLE001 — сеть: подождать и повторить
                time.sleep(3 * (attempt + 1))
        else:
            return

        if not page:
            return

        for item in page:
            yield item["row"]


def phrase(question: str, answer: str) -> str | None:
    """Вопрос и ответ одной фразой, без знаков конца предложения внутри."""

    clean = lambda text: re.sub(r"\s+", " ", re.sub(r"[.!?;\n]+", " ", text)).strip().lower()
    question, answer = clean(question), clean(answer)

    if not question or not answer:
        return None

    return f"{question} — {answer}."


def short(text: str, words: int) -> bool:
    return 0 < len(text.split()) <= words


def main(limit: int = 3000) -> None:
    out = Path("data/corpus/qa")
    out.mkdir(parents=True, exist_ok=True)

    qa = []
    for row in rows("kuznetsoffandrey/sberquad", "sberquad", "train", limit):
        answer = row["answers"]["text"][0] if row["answers"]["text"] else ""
        if short(answer, 6):
            line = phrase(row["question"], answer)
            if line:
                qa.append(line)

    instructions = []
    for row in rows("d0rj/alpaca-cleaned-ru", "default", "train", limit * 4):
        if row["input"] or not short(row["output"], 15) or not short(row["instruction"], 25):
            continue
        line = phrase(row["instruction"], row["output"])
        if line:
            instructions.append(line)
        if len(instructions) >= limit:
            break

    (out / "вопрос-ответ.txt").write_text("\n".join(qa), encoding="utf-8")
    (out / "запрос-выполнение.txt").write_text("\n".join(instructions), encoding="utf-8")
    print(f"вопрос-ответ: {len(qa)} пар, запрос-выполнение: {len(instructions)} пар -> {out}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3000)
