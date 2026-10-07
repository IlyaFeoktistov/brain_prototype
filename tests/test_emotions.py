"""Эмоции: реакции на события и выключатель."""

from brain.limbic import Emotions


def test_praise_makes_happy():
    emotions = Emotions()
    emotions.feedback(+6)
    assert emotions.mood.name == "радость"


def test_blame_makes_upset():
    emotions = Emotions()
    emotions.feedback(-6)
    emotions.feedback(-6)
    assert emotions.mood.name in ("досада", "грусть")


def test_dont_know_confuses():
    emotions = Emotions()
    emotions.answered([])
    assert emotions.mood.name == "растерянность"


def test_surprise_raises_arousal_and_calms_down():
    emotions = Emotions()
    for _ in range(4):
        emotions.heard(6.0, guessed=False)
    assert emotions.arousal > 0.6

    for _ in range(40):
        emotions.heard(0.0, guessed=True)
    assert emotions.arousal < 0.2


def test_emotions_switch_off(mind):
    mind.cfg.emotions_enabled = False
    language = mind.language
    before = language.emotions.state_dict()

    language.converse("Столица Франции Париж")
    language.converse("+")

    assert language.emotions.state_dict() == before


def test_emotions_persist(mind, tmp_dir):
    import os
    from brain.mind import Mind

    mind.language.converse("+")
    path = os.path.join(tmp_dir, "mood.pt")
    mind.save(path)

    loaded = Mind.load(path)
    assert loaded.language.emotions.state_dict() == mind.language.emotions.state_dict()
