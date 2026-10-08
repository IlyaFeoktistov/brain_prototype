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


def test_blamed_when_sure_gets_angry_when_unsure_gets_sad():
    sure = Emotions()
    sure.answered([0.9, 0.9])
    sure.feedback(-6)
    assert sure.mood.name == "злость"

    unsure = Emotions()
    unsure.answered([0.1])
    unsure.confusion = 0.0
    unsure.feedback(-6)
    assert unsure.mood.name in ("досада", "грусть")


def test_idle_and_monotony_make_bored():
    idle = Emotions()
    idle.relax(300)
    assert idle.mood.name == "скука"

    monotony = Emotions()
    for _ in range(30):
        monotony.heard(0.0, guessed=True)
    hungry = monotony.novelty_hunger
    assert hungry > 0.6

    # Неожиданное утоляет голод по новому.
    monotony.heard(6.0, guessed=False)
    assert monotony.novelty_hunger < hungry - 0.25


def test_praise_is_sweeter_when_wanted():
    lonely, sated = Emotions(), Emotions()
    lonely.social_need, sated.social_need = 1.0, 0.0
    lonely.feedback(+6)
    sated.feedback(+6)
    assert lonely.valence > sated.valence
    assert lonely.social_need < 1.0
