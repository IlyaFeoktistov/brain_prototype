"""
Регрессия обучения: то, что уже работает, не должно сломаться.

Пороги взяты с запасом от замеров (docs/DESIGN.md), чтобы тест ловил
поломку механизма, а не шум отдельного прогона.
"""

import pytest

from brain.language import Confidence
from tests.conftest import FACTS, make_mind


def rank(language, prompt: str, target: str) -> int:
    from brain.language import END

    probe = language.probe(prompt)
    # «Конец фразы» — не слово-ответ, в рангах не участвует.
    ranked = language.semantic.decode(probe["expected"], {language.word(prompt.split()[-1]), END}, top=200)
    return next((i + 1 for i, (token, _) in enumerate(ranked) if token == target), 999)


def test_learns_facts_from_reading(trained):
    language = trained.language
    # Ответ среди первых трёх ожиданий (словарь ~20 слов).
    assert rank(language, "столица франции", "париж") <= 3
    assert rank(language, "столица германии", "берлин") <= 3


def test_prediction_improves_and_dopamine_falls():
    mind = make_mind("small")
    language = mind.language
    first = language.read_text(FACTS)

    for _ in range(4):
        last = language.read_text(FACTS)

    assert last.hits > first.hits
    # Выученное перестаёт перестраивать сеть.
    assert last.dopamine < first.dopamine


def test_unknown_last_word_gets_dont_know(trained):
    assert trained.language.generate("квантовая хромодинамика описывает") == []


def test_does_not_ask_about_short_words(mind):
    language = mind.language
    language.read_text(FACTS)
    assert "что такое" not in language.converse("Столица Франции и").text


def test_asks_about_unknown_words(mind):
    language = mind.language
    language.read_text(FACTS)

    reply = language.converse("Столица Франции хромодинамика")
    assert "что такое «хромодинамика»" in reply.text
    assert reply.unknown == ["хромодинамика"]

    # Ответ человека — обычный опыт: теперь слово знакомо,
    # и на объяснение мозг не отвечает встречными вопросами.
    explained = language.converse("Хромодинамика это раздел физики")
    assert "что такое" not in explained.text
    assert language.familiar("хромодинамика")
    assert "хромодинамика" not in language.converse("Хромодинамика это").text


@pytest.mark.xfail(reason="уплывание после верного ответа (ROADMAP P0) — «Париж Берлин Франции Париж»; "
                   "тест переключается почти от каждого изменения цели, веток, гиппокампа", strict=False)
def test_answer_does_not_repeat_words(trained):
    produced = trained.language.generate("столица франции")
    # Слова, без знаков: «.» может стоять после каждого.
    tokens = [trained.language.word(token) for token, _ in produced if token[0].isalnum()]

    assert len(tokens) == len(set(tokens))
    assert not {"столица", "франции"} & set(tokens)


def test_confidence_is_monotone():
    confidence = Confidence()

    for margin, hit in [(1.0, False)] * 20 + [(1.6, True)] * 10 + [(1.6, False)] * 10 + [(2.5, True)] * 9:
        confidence.update(margin, hit)

    assert confidence(1.0) < confidence(1.6) <= confidence(2.5)
    # Более резкое ожидание не менее надёжно даже без данных.
    assert confidence(3.5) >= confidence(2.5)


def test_praise_changes_weights(mind):
    language = mind.language
    language.read_text(FACTS)
    language.converse("столица франции")
    before = mind.brain.net_weight.clone()

    language.converse("+")

    assert not (before == mind.brain.net_weight).all()


def test_question_mark_does_not_break_learning():
    mind = make_mind("small")
    language = mind.language

    for _ in range(6):
        language.read_text(
            "Какая столица Франции? Париж.\nКакая столица Германии? Берлин.\n"
            "Какая столица Италии? Рим."
        )

    # «?» и «.» — символы внутри строки, пауза — только конец строки.
    assert language.sentences("Какая столица Франции? Париж.\nДа.") == ["Какая столица Франции? Париж.", "Да."]
    assert rank(language, "какая столица франции ?", "париж") <= 3


def test_separate_messages_are_not_glued(mind):
    language = mind.language
    language.read_text(FACTS)
    before = rank(language, "столица германии", "квантовая") if "квантовая" in language.semantic else 999

    for _ in range(5):
        language.converse("Столица Германии")
        language.converse("Квантовая хромодинамика")

    # Конец одной реплики не должен предсказывать начало следующей:
    # сила «квантовая» — шум рядом с настоящим ответом (в словаре из двух
    # десятков слов почти нулевой кандидат легко встаёт вторым по рангу).
    ranked = dict(language.semantic.decode(language.probe("столица германии")["expected"], {"германии"}, top=50))
    assert ranked.get("квантовая", 0.0) < 0.1 * max(ranked.values())
    assert before >= 3


def test_answer_to_own_question_is_rewarded_and_remembered(mind):
    language = mind.language
    language.read_text(FACTS)
    language.converse("Столица Франции хромодинамика")
    assert "хромодинамика" in language.asked

    weights = mind.brain.net_weight.clone()
    episodes = len(language.hippocampus)
    language.converse("Хромодинамика это раздел физики")

    assert "хромодинамика" not in language.asked
    assert len(language.hippocampus) > episodes
    assert any(e.tokens[:3] == ["что", "такое", "хромодинамика"] for e in language.hippocampus.episodes)
    assert not (weights == mind.brain.net_weight).all()


def test_word_choice_is_probabilistic_in_speech(mind):
    language = mind.language
    close = [("париж", 0.50), ("рим", 0.49), ("берлин", 0.10)]
    clear = [("париж", 0.80), ("рим", 0.10)]

    picks = {language._pick(close, sample=True) for _ in range(200)}
    # Близкие кандидаты побеждают оба, далёкий — почти никогда.
    assert {"париж", "рим"} <= picks
    assert all(language._pick(clear, sample=True) == "париж" for _ in range(50))
    # Экзамен и пробы — всегда лучший.
    assert all(language._pick(close, sample=False) == "париж" for _ in range(50))


def test_categories_generalize_to_unheard_pair():
    mind = make_mind("small")
    language = mind.language

    for _ in range(4):
        language.read_text("Кошка спит.\nСобака спит.\nЛошадь спит.\nКошка ест.\nСобака ест.\nЛошадь ест.\nКорова спит.")

    # «Корова ест» мозг не слышал, но корова — в одной категории с
    # кошкой, собакой и лошадью.
    ranked = dict(language.semantic.decode(language.probe("корова")["expected"], {"корова"}, top=10))
    assert ranked.get("ест", 0.0) > 0.05
