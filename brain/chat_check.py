"""
Чат-минимум: можно ли с мозгом поговорить (ROADMAP «Ближайшая цель»).

    python -m brain.chat_check размер имя [key=value ...] [load=путь] [scenario=файл]

Скрипт играет человека (это не код мозга): говорит реплики из
chat-scenario.txt обычным чатом (`Session.say` — мозг учится, как в
разговоре) и проверяет ответы. Ошибся на «поправке» — говорит верный
ответ. «сон» — мозг спит до конца. Считает по группам и ещё:
- подмешивание: в ответе слово из прошлой реплики человека, которого нет
  ни в этом вопросе, ни в верном ответе («меня зовут Игорь Париж»);
- лишние слова после верного (уплывание).

Без load= мозг создаётся и читает train-data.md дважды и спит (как
brain.teach). Расшифровка — data/measure/имя-chat.txt, мозг после
разговора — имя-chatted.pt.
"""

from __future__ import annotations

from dataclasses import dataclass
import sys
import time
from pathlib import Path

import torch

from brain.commands import Session
from brain.config import BrainConfig
from brain.mind import Mind
from brain.teach import FIRST_WORDS, same_word

# Человек читает ответ и печатает свой — тишина перед каждой репликой, мс.
PAUSE_MS = 1000.0
DONT_KNOW = "не знаю"


@dataclass
class Line:
    group: str
    text: str
    answers: list[str]


def load_scenario(path: Path) -> list[Line | None]:
    """Реплики по порядку; None — сон."""

    lines: list[Line | None] = []

    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()

        if not raw or raw.startswith("#"):
            continue

        if raw == "сон":
            lines.append(None)
            continue

        group, text, answer = (part.strip() for part in raw.split("|"))
        answers = [a.strip() for a in answer.split("/") if a.strip()]
        lines.append(Line(group, text, answers))

    return lines


class Human:
    def __init__(self, mind: Mind, transcript: list[str]):
        self.mind = mind
        self.language = mind.language
        self.session = Session(mind, lambda text, kind: None)
        self.transcript = transcript
        self.previous = ""

    def words(self, text: str) -> list[str]:
        return [t for t in self.language.tokenize(text) if t[0].isalnum()]

    def say(self, text: str) -> list[str]:
        self.mind.brain.run(PAUSE_MS)
        reply = self.session.say(text)
        self.transcript.append(f"я: {text}\n   мозг: {reply.text}")

        return self.words(self.language.dialog[-1].get("answer", "")), reply.text

    def check(self, line: Line) -> tuple[bool | None, bool, int]:
        """Верно ли (None — не проверяется), подмешал ли прошлую реплику, слов после верного."""

        previous = self.previous
        said, text = self.say(line.text)
        self.previous = line.text

        if line.answers == [DONT_KNOW]:
            hit = DONT_KNOW in text.casefold() or not said
            drift = 0
        else:
            at = next(
                (i for i, word in enumerate(said[:FIRST_WORDS]) for key in line.answers if same_word(word, key)),
                None,
            )
            hit = at is not None if line.answers else None
            drift = len(said) - at - 1 if at is not None else 0

        own = {w.casefold() for w in self.words(line.text)} | {a.casefold() for a in line.answers}
        mixed = any(
            any(same_word(word, old) for old in self.words(previous))
            and not any(same_word(word, mine) for mine in own)
            for word in said
        )

        if hit is not None:
            self.transcript[-1] += "   ✓" if hit else "   ✗"
        if mixed:
            self.transcript[-1] += "   (подмешал прошлую реплику)"

        return hit, mixed, drift


def main() -> None:
    size = sys.argv[1] if len(sys.argv) > 1 else "small"
    name = sys.argv[2] if len(sys.argv) > 2 else size
    load = None
    scenario = Path("chat-scenario.txt")
    overrides = []

    for kv in sys.argv[3:]:
        k, v = kv.split("=", 1)

        if k == "load":
            load = v
        elif k == "scenario":
            scenario = Path(v)
        else:
            overrides.append((k, v))

    def apply(cfg):
        for k, v in overrides:
            old = getattr(cfg, k)
            setattr(cfg, k, (v == "1") if isinstance(old, bool) else type(old)(v))
            print(f"{k} = {getattr(cfg, k)}")

    t0 = time.time()

    if load:
        mind = Mind.load(load)
        mind.cfg.autosave = False
        apply(mind.cfg)
        mind.brain.recompile()
    else:
        cfg = BrainConfig()
        cfg.size = size
        cfg.autosave = False
        cfg.save_path = "/nonexistent/brain.pt"
        apply(cfg)
        mind = Mind(cfg)
        mind.calibrate()
        reader = Session(mind, lambda text, kind: None)
        text = Path("train-data.md").read_text(encoding="utf-8")

        for _ in range(2):
            reader.read_text("train-data", text)

        mind.language.sleep_until_rested()
        mind.save(f"data/measure/{name}-read.pt")

    if mind.brain.device.type == "cuda":
        torch.cuda.synchronize()
    print(f"{size}: {mind.brain.n:,} нейронов, готов за {time.time() - t0:.0f} с", flush=True)

    transcript: list[str] = []
    human = Human(mind, transcript)
    score: dict[str, list[bool]] = {}
    mixed, drifts = 0, []

    for line in load_scenario(scenario):
        if line is None:
            t = time.time()
            mind.language.sleep_until_rested()
            transcript.append(f"… сон {time.time() - t:.0f} с …")
            human.previous = ""
            continue

        hit, mix, drift = human.check(line)
        mixed += mix

        if hit is not None:
            score.setdefault(line.group, []).append(hit)

        if hit:
            drifts.append(drift)

        # Ошибся на поправке — человек говорит верный ответ.
        if line.group == "поправка" and hit is False:
            human.say(line.answers[0])
            human.previous = line.answers[0]

    # Мозг после разговора — разбирать, что он вспоминал.
    mind.save(f"data/measure/{name}-chatted.pt")
    total = sum(sum(v) for v in score.values())
    count = sum(len(v) for v in score.values())
    lines = [f"{group:10s} {sum(v)}/{len(v)}" for group, v in score.items()] + [
        f"всего      {total}/{count}",
        f"подмешал прошлую реплику  {mixed} раз",
        f"слов после верного        {sum(drifts) / max(1, len(drifts)):.1f} в среднем",
    ]
    out = Path(f"data/measure/{name}-chat.txt")
    out.write_text("\n".join(transcript + [""] + lines) + "\n", encoding="utf-8")
    print("\n".join(transcript + [""] + lines), flush=True)
    print(f"расшифровка: {out}")


if __name__ == "__main__":
    main()
