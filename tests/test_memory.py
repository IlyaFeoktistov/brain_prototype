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


def test_initiative_asks_about_new_word_from_reading(mind, tmp_dir):
    import os
    from brain.mind import Mind

    language = mind.language
    language.read_text(FACTS + " Фотосинтез происходит в хлоропластах.")
    path = os.path.join(tmp_dir, "wonder.pt")
    mind.save(path)

    # Незнакомые слова переживают перезапуск.
    loaded = Mind.load(path).language
    assert "хлоропластах" in loaded.wonder

    language.emotions.valence, language.emotions.arousal = 0.0, 0.5
    said = language.initiative()
    assert said is not None and said[0] == "любопытство"
    assert "что такое" in said[1]


def test_initiative_sleepy_and_sad(mind):
    language = mind.language
    language.emotions.feedback(-6)
    language.emotions.feedback(-6)
    assert language.initiative()[0] == "грусть"

    for i in range(mind.cfg.sleepy_after_episodes):
        language.hippocampus.store(["слово", f"номер{i}"], 1.0)
    assert language.initiative()[0] == "сонливость"


def test_wandering_does_not_change_brain(mind):
    language = mind.language
    language.read_text(FACTS)
    weights = mind.brain.net_weight.clone()
    episodes = len(language.hippocampus)

    language.wander()

    assert (weights == mind.brain.net_weight).all()
    assert len(language.hippocampus) == episodes
    assert not language.musing
