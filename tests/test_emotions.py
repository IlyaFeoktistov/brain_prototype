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


def test_habituation_to_steady_surprise():
    emotions = Emotions()
    # Чтение: неожиданно почти каждое слово — к этому мозг привыкает.
    for _ in range(200):
        emotions.heard(4.0, guessed=False)
    assert emotions.arousal < 0.5

    # А резкий всплеск сверх привычного снова будоражит.
    for _ in range(3):
        emotions.heard(12.0, guessed=False)
    assert emotions.arousal > 0.5


def test_arousal_strengthens_learning(mind):
    language = mind.language
    brain = mind.brain
    language.read_text("Столица Франции Париж.")
    eligibility = brain.eligibility.clone()

    def change(arousal):
        language.emotions.arousal = arousal
        language._feel("relax", 0.0)
        before = brain.net_weight.clone()
        brain.eligibility.copy_(eligibility)
        brain.reward(1.0)
        delta = (brain.net_weight - before).abs().sum()
        brain.net_weight.copy_(before)
        return float(delta)

    calm, excited = change(0.0), change(1.0)
    assert excited > calm * 1.5


def test_mood_changes_boldness(mind):
    language = mind.language
    language.emotions.valence = 1.0
    bold = language._min_confidence()
    language.emotions.valence = -1.0
    careful = language._min_confidence()
    assert bold < mind.cfg.generate_min_confidence < careful


def test_emotional_episode_is_more_salient(mind):
    language = mind.language
    language.emotions.arousal = 0.0
    language._remember(["спокойная", "фраза"], 1.0)
    language.emotions.arousal = 1.0
    language._remember(["важная", "фраза"], 1.0)
    calm, excited = language.hippocampus.episodes[-2:]
    assert excited.salience > calm.salience
