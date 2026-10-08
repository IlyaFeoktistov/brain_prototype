"""
Экзамен: сколько вопросов мозг отвечает и как часто говорит «не знаю».

    python -m brain.evaluate data/brain.pt [--qa-every 10] [--show 5]

Вопросы двух видов (только чтение, мозг не меняется):

    факты    из train-data.md: «Столица Франции — Париж.» ->
             «столица франции» … «париж»; «Париж является городом.» ->
             «париж является» … «городом»
    вопросы  каждый N-й из data/corpus/qa: «какая столица франции? …»

Метрики:
    ответил     хоть что-то сказал (не «не знаю»)
    первое      первое слово ответа = первому слову правильного ответа
    в тройке    среди первых трёх слов ответа есть слово правильного
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from brain.mind import Mind


@dataclass
class Question:
    prompt: str
    answer: list[str]
    kind: str


@dataclass
class Score:
    total: int = 0
    answered: int = 0
    first: int = 0
    top3: int = 0
    examples: list[str] = field(default_factory=list)

    def line(self, name: str) -> str:
        n = max(1, self.total)
        return (
            f"{name:8s} {self.total:4d} вопросов: ответил {self.answered / n:5.1%}, "
            f"«не знаю» {1 - self.answered / n:5.1%}, первое слово {self.first / n:5.1%}, "
            f"в тройке {self.top3 / n:5.1%}"
        )


def facts(path: Path, tokenize) -> list[Question]:
    questions = []

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip().rstrip(".")

        if not line or line.startswith("#"):
            continue

        if "—" in line:
            left, right = line.split("—", 1)
        elif " является " in line:
            left, right = line.split(" является ", 1)
            left += " является"
        else:
            continue

        prompt, answer = tokenize(left), tokenize(right)

        if prompt and answer:
            questions.append(Question(" ".join(prompt), answer, "факты"))

    return questions


def qa(folder: Path, every: int, tokenize) -> list[Question]:
    questions = []

    for path in sorted(folder.glob("*.txt")):
        lines = [l for l in path.read_text(encoding="utf-8").splitlines() if "?" in l]

        for line in lines[::every]:
            question, answer = line.split("?", 1)
            answer = tokenize(answer)

            if answer:
                questions.append(Question(question.strip() + "?", answer, "вопросы"))

    return questions


def run(mind: Mind, questions: list[Question], show: int) -> dict[str, Score]:
    language = mind.language
    scores: dict[str, Score] = {}

    for q in questions:
        score = scores.setdefault(q.kind, Score())
        said = [token for token, _ in language.generate(q.prompt)]
        score.total += 1
        score.answered += bool(said)
        score.first += bool(said) and said[0] == q.answer[0]
        score.top3 += bool(set(said[:3]) & set(q.answer))

        if len(score.examples) < show:
            score.examples.append(
                f"  {q.prompt} → {' '.join(said) or 'не знаю'}   (надо: {' '.join(q.answer[:6])})"
            )

    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("brain")
    parser.add_argument("--facts", default="train-data.md")
    parser.add_argument("--qa", default="data/corpus/qa")
    parser.add_argument("--qa-every", type=int, default=10)
    parser.add_argument("--show", type=int, default=5)
    args = parser.parse_args()

    mind = Mind.load(args.brain)
    tokenize = mind.language.tokenize
    questions = facts(Path(args.facts), tokenize) + qa(Path(args.qa), args.qa_every, tokenize)

    print(f"{args.brain}: {mind.brain.n:,} нейронов, знает {len(mind.language.semantic):,} слов")

    for name, score in run(mind, questions, args.show).items():
        print(score.line(name))
        print("\n".join(score.examples))


if __name__ == "__main__":
    main()
