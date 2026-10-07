"""Гиппокамп, сон, память диалога, размышление."""

from tests.conftest import FACTS


def test_reading_fills_hippocampus_and_sleep_consolidates(mind):
    language = mind.language
    language.read_text(FACTS)

    assert len(language.hippocampus) == 6
    assert language.sleep_pressure == 6

    before = mind.brain.net_weight.clone()
    report = language.sleep_until_rested()

    assert report.sentences >= 6
    assert language.sleep_pressure == 0
    assert not (before == mind.brain.net_weight).all(), "сон должен менять синапсы"
    # Во сне новые эпизоды не пишутся.
    assert len(language.hippocampus) <= 6


def test_sleep_face(mind):
    language = mind.language
    language.read_text(FACTS)
    seen = []
    language.on_step = lambda step, learn: seen.append(language.emotions.mood.name)
    language.sleep(3)

    assert seen and set(seen) == {"сон"}
    assert language.emotions.mood.name != "сон"


def test_praised_answer_goes_to_hippocampus(mind):
    language = mind.language
    language.read_text(FACTS)
    language.converse("Столица Германии")
    episodes = len(language.hippocampus)

    language.converse("+")

    if language.dialog[-1].get("answer"):
        assert len(language.hippocampus) == episodes + 1
    assert language.dialog[-1]["rating"] == 1


def test_reread_skips_blamed_answers(mind):
    language = mind.language
    language.read_text(FACTS)
    language.converse("Столица Германии")
    language.converse("-")

    blamed = language.dialog[-1].get("answer")
    report = language.reread(10)

    assert report.sentences >= 1
    assert language.dialog[-1]["rating"] == -1
    if blamed:
        assert report.tokens == len(language.tokenize("Столица Германии"))


def test_thinking_logs_thoughts_and_is_read_only(trained):
    language = trained.language
    weights = trained.brain.net_weight.clone()

    with language.isolated():
        language.experience(language.tokenize("столица франции"), learn=False)
        produced, log = language.think()

    assert (weights == trained.brain.net_weight).all()
    assert all(step["verdict"] in ("ok", "no", "stop") for step in log)
    assert len(produced) == sum(step["verdict"] == "ok" for step in log)


def test_dialog_persists(mind, tmp_dir):
    import os
    from brain.mind import Mind

    mind.language.converse("Столица Франции Париж")
    path = os.path.join(tmp_dir, "dialog.pt")
    mind.save(path)

    assert Mind.load(path).language.dialog == mind.language.dialog


def test_new_words_from_reading_survive_restart(mind, tmp_dir):
    import os
    from brain.mind import Mind

    mind.language.read_text(FACTS + " Фотосинтез происходит в хлоропластах.")
    path = os.path.join(tmp_dir, "wonder.pt")
    mind.save(path)

    assert "хлоропластах" in Mind.load(path).language.wonder


def _bored(emotions):
    emotions.valence, emotions.arousal, emotions.confusion, emotions.predictability = 0.0, 0.05, 0.0, 0.9


def _joyful(emotions):
    emotions.valence, emotions.arousal, emotions.confusion, emotions.predictability = 0.6, 0.6, 0.0, 0.0


def test_initiative_is_learned_not_scripted():
    from tests.conftest import make_mind

    mind = make_mind("small")
    language = mind.language
    language.read_text(FACTS)

    scripted = ("дай что-нибудь почитать", "можно я посплю", "что-то не так ответил")

    # Слышит «мне скучно», когда ему скучно, — и связывает чувство со словами.
    for _ in range(8):
        _bored(language.emotions)
        language.converse("мне скучно")

    _bored(language.emotions)
    said = language.initiative()
    assert said is not None and "скучно" in said[1]
    assert not any(phrase in said[1] for phrase in scripted)

    # В другом чувстве выученное про скуку не всплывает.
    _joyful(language.emotions)
    joyful = language.initiative()
    assert joyful is None or "скучно" not in joyful[1]


def test_wandering_does_not_change_brain(mind):
    language = mind.language
    language.read_text(FACTS)
    weights = mind.brain.net_weight.clone()
    episodes = len(language.hippocampus)

    language.wander()

    assert (weights == mind.brain.net_weight).all()
    assert len(language.hippocampus) == episodes
    assert not language.musing
