"""
Разговор как учитель: видно ли в чате, что мозг понял поправку.

    python -m brain.teach размер имя [key=value ...] [load=путь] [facts=файл]

Скрипт играет человека (это не код мозга): спрашивает факты из
teach-data.txt обычными репликами (`converse` — мозг учится, как в
чате). Ошибся — говорит верный ответ; через две реплики переспрашивает
тот же вопрос. Потом мозг спит, и проверка без поправок: отложенные
вопросы (прочитаны, но в разговоре не звучали), те же вопросы другими
словами, те же вопросы дословно.

Без load= мозг создаётся и читает train-data.md дважды (как brain.measure)
и сохраняется в data/measure/имя-read.pt — с него можно повторить.
Расшифровка — в data/measure/имя-teach.txt, мозг после урока и сна —
в data/measure/имя-taught.pt.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from brain.commands import Session
from brain.config import BrainConfig
from brain.mind import Mind

# Через сколько реплик переспросить поправленный вопрос: «понял» должно
# пережить хотя бы пару чужих вопросов, иначе это эхо рабочей памяти.
REASK_AFTER = 2
# Человек читает ответ и печатает свой — время тишины перед каждой его
# репликой, мс (pause=…). 0 — реплики вплотную, как было.
PAUSE_MS = 0.0
# Учится ли мозг на проверочных вопросах (test_learns=0 — нет: проверка
# без следа, чтобы отделить, что стирают сами вопросы).
TEST_LEARNS = True
# Верным считается ответ, где ключевое слово среди первых слов: как
# «в тройке» у экзамена — «в европе», «вокруг солнца» начинаются с предлога.
FIRST_WORDS = 3


@dataclass
class Fact:
    question: str
    answer: str
    paraphrases: list[str]
    held_out: bool


def load_facts(path: Path) -> list[Fact]:
    facts = []
    held_out = False

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if line.startswith("# отложенные"):
            held_out = True
            continue

        if not line or line.startswith("#"):
            continue

        parts = [p.strip() for p in line.split("|")]
        paraphrases = [p.strip() for p in parts[2].split(";")] if len(parts) > 2 else []
        facts.append(Fact(parts[0], parts[1], [p for p in paraphrases if p], held_out))

    return facts


def same_word(said: str, key: str) -> bool:
    """Одно слово с точностью до окончания: «звездой» = «звезда»."""

    # Общее начало без последних двух букв, но не короче трёх: «европе» =
    # «европа», «рим» ≠ «и».
    stem = max(3, len(key) - 2)

    said, key = said.casefold(), key.casefold()

    return said[:stem] == key[:stem] and abs(len(said) - len(key)) <= 3


class Teacher:
    def __init__(self, mind: Mind, transcript: list[str]):
        self.mind = mind
        self.language = mind.language
        self.transcript = transcript
        self.session = Session(mind, lambda text, kind: None)

    def ask(self, text: str, key: str, tag: str) -> tuple[bool, int]:
        """Реплика человека → ответ мозга. Верно ли и сколько слов после ключа."""

        if PAUSE_MS > 0:
            self.mind.brain.run(PAUSE_MS)

        reply = self.session.say(text)
        # Слова ответа — без знаков: «верно» и «уплывание» считаются по словам.
        answer = [
            token for token in self.language.tokenize(self.language.dialog[-1].get("answer", ""))
            if token[0].isalnum()
        ]
        hit = next(
            (i for i, word in enumerate(answer[:FIRST_WORDS]) if same_word(word, key)),
            None,
        )
        drift = len(answer) - hit - 1 if hit is not None else 0
        mark = "✓" if hit is not None else "✗"
        self.transcript.append(f"[{tag}] я: {text}\n      мозг: {reply.text}   {mark}")

        return hit is not None, drift

    def probe(self, facts: list[Fact], key) -> tuple[int, list[str]]:
        """
        Проба без следа (generate — изолированно, без обучения): сколько
        вопросов урока мозг сейчас отвечает дословно. Отличает, что стёр
        сон, а что — сами проверочные вопросы (они учат, как любой опыт).
        """

        hits, answers = 0, []

        for fact in facts:
            said = [t for t, _ in self.language.generate(fact.question) if t[0].isalnum()]
            hit = any(same_word(word, key(fact)) for word in said[:FIRST_WORDS])
            hits += hit
            answers.append(f"{fact.question} → {' '.join(said) or 'не знаю'} {'✓' if hit else '✗'}")

        return hits, answers

    def correct(self, fact: Fact) -> None:
        if PAUSE_MS > 0:
            self.mind.brain.run(PAUSE_MS)

        reply = self.session.say(fact.answer)
        self.transcript.append(f"[поправка] я: {fact.answer}\n      мозг: {reply.text}")


def main() -> None:
    size = sys.argv[1] if len(sys.argv) > 1 else "small"
    name = sys.argv[2] if len(sys.argv) > 2 else size
    load = None
    facts_path = Path("teach-data.txt")
    overrides = []

    for kv in sys.argv[3:]:
        k, v = kv.split("=", 1)

        if k == "load":
            load = v
        elif k == "pause":
            global PAUSE_MS
            PAUSE_MS = float(v)
        elif k == "test_learns":
            global TEST_LEARNS
            TEST_LEARNS = v == "1"
        elif k == "facts":
            facts_path = Path(v)
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

        mind.save(f"data/measure/{name}-read.pt")

    torch.cuda.synchronize()
    print(f"{size}: {mind.brain.n:,} нейронов, готов за {time.time() - t0:.0f} с", flush=True)

    facts = load_facts(facts_path)
    taught = [f for f in facts if not f.held_out]
    held = [f for f in facts if f.held_out]
    key = lambda fact: fact.answer.split()[-1]
    transcript: list[str] = []
    teacher = Teacher(mind, transcript)

    # Урок: спросить, ошибся — поправить, через REASK_AFTER вопросов переспросить.
    first, corrected, after_fix = 0, [], 0
    pending: list[tuple[int, Fact]] = []

    for i, fact in enumerate(taught):
        hit, _ = teacher.ask(fact.question, key(fact), "урок")
        first += hit

        if not hit:
            teacher.correct(fact)
            corrected.append(fact)
            pending.append((i + REASK_AFTER, fact))

        for due, again in [p for p in pending if p[0] <= i]:
            after_fix += teacher.ask(again.question, key(again), "переспрос")[0]
            pending.remove((due, again))

    for _, again in pending:
        after_fix += teacher.ask(again.question, key(again), "переспрос")[0]

    probes = []
    hits, answers = teacher.probe(taught, key)
    probes.append(("проба до сна", hits, answers))

    t = time.time()
    mind.language.sleep_until_rested()
    transcript.append(f"… сон {time.time() - t:.0f} с …")
    # Мозг после урока и сна — для /trace и проб.
    mind.save(f"data/measure/{name}-taught.pt")

    # Проверка без поправок: сначала то, что меньше всего подсказано уроком.
    def test(items: list[tuple[str, Fact]], tag: str) -> tuple[int, list[int]]:
        hits, drifts = 0, []

        for text, fact in items:
            hit, drift = teacher.ask(text, key(fact), tag)
            hits += hit

            if hit:
                drifts.append(drift)

        return hits, drifts

    hits, answers = teacher.probe(taught, key)
    probes.append(("проба после сна", hits, answers))

    mind.brain.plasticity = TEST_LEARNS
    held_hits, held_drift = test([(f.question, f) for f in held], "отложенный")
    para = [(p, f) for f in taught for p in f.paraphrases]
    para_hits, para_drift = test(para, "иначе")
    hits, answers = teacher.probe(taught, key)
    probes.append(("проба после 50 проверочных вопросов", hits, answers))
    slept_hits, slept_drift = test([(f.question, f) for f in taught], "после сна")

    drifts = held_drift + para_drift + slept_drift
    lines = [f"{name:38s} {hits}/{len(taught)}" for name, hits, _ in probes] + [
        f"сразу верно          {first}/{len(taught)}",
        f"после поправки       {after_fix}/{len(corrected)} (переспрос через {REASK_AFTER} реплики)",
        f"после сна, дословно  {slept_hits}/{len(taught)}",
        f"другими словами      {para_hits}/{len(para)}",
        f"отложенные           {held_hits}/{len(held)}",
        f"слов после верного   {sum(drifts) / max(1, len(drifts)):.1f} в среднем (0 — замолк)",
    ]
    out = Path(f"data/measure/{name}-teach.txt")
    detail = [f"\n{name}:\n  " + "\n  ".join(answers) for name, _, answers in probes]
    out.write_text("\n".join(transcript + detail + [""] + lines) + "\n", encoding="utf-8")
    print("\n".join(transcript[:12] + ["…"] + lines), flush=True)
    print(f"расшифровка: {out}")


if __name__ == "__main__":
    main()
