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


def test_nuclei_make_dopamine_and_locus_coeruleus_habituates():
    from tests.conftest import make_mind

    mind = make_mind("small")
    language, brain = mind.language, mind.brain
    brain.run(500)

    # Сильная неожиданность — больше дофамина, чем слабая.
    weak, _ = language._nucleus_burst(0.3)
    brain.run(500)
    strong, first = language._nucleus_burst(0.9)
    assert strong > weak > 0

    # Серия неожиданностей: голубое пятно стихает (адаптация клеток)…
    for _ in range(15):
        brain.run(60)
        dopamine, last = language._nucleus_burst(0.9)
    assert last < first * 0.6
    # …а дофамин — нет: он падает от предсказания, не от усталости.
    assert dopamine > strong * 0.7

    # После затишья снова отвечает.
    brain.run(10_000)
    _, rested = language._nucleus_burst(0.9)
    assert rested > last * 1.5


def test_joy_of_guessing_habituates():
    emotions = Emotions()
    for _ in range(200):
        emotions.heard(0.0, guessed=True)

    # Привык к угадыванию — радость не упирается в потолок…
    assert emotions.valence < 0.5
    # …и ругань его действительно огорчает.
    emotions.feedback(-6)
    assert emotions.valence < 0


def test_praise_bursts_and_blame_pauses_dopamine_cells():
    from tests.conftest import make_mind

    mind = make_mind("small")
    language, brain = mind.language, mind.brain
    assert 0.1 < language._vta_base() < 0.7

    brain.run(300)
    praise = language._dopamine(+6.0)
    brain.run(300)
    blame = language._dopamine(-6.0)

    # Похвала — залп выше фона, ругань — пауза ниже фона.
    assert praise > 1.0
    assert blame < -1.0


def test_amygdala_learns_what_leads_to_blame():
    from tests.conftest import make_mind

    mind = make_mind("small")
    language = mind.language
    language.read_text("Компьютер сломался.\nКошка спит.")

    def value_of(word):
        with language.isolated():
            language.experience([word], learn=False)
            return language.value(language.last_rates)

    before = value_of("компьютер")
    said = set()
    # Каждый раз, когда говорили про компьютер, — ругань.
    for _ in range(3):
        said |= set(language.words(language.converse("компьютер сломался").text))
        language.converse("-")

    assert value_of("компьютер") < before - 0.1
    # Ругань задевает и ответ мозга («кошка» — он ждёт её после
    # «сломался»); слова, которых не было в ссоре, остаются нейтральными.
    calm = [word for word in ("кошка", "спит") if word not in said]
    assert calm
    assert all(abs(value_of(word)) < 0.05 for word in calm)


def test_praise_words_are_learned_not_scripted():
    from tests.conftest import make_mind

    mind = make_mind("small")
    language = mind.language
    language.read_text("Столица Франции Париж. Ты молодец.")

    # Слово само по себе пока ничего не значит.
    language.converse("Столица Франции")
    assert language.converse("молодец").praised == 0.0

    # Человек говорит «молодец» и жмёт «Верно» — врождённая награда.
    for _ in range(4):
        language.converse("Столица Франции")
        language.converse("молодец")
        language.converse("+")

    # Теперь «молодец» — условная награда: дофамин без кнопки.
    language.converse("Столица Франции")
    assert language.converse("молодец").praised > 0


def test_unanswered_question_builds_urge_to_speak():
    calm, ignored = Emotions(), Emotions()
    ignored.spoke()

    for emotions in (calm, ignored):
        emotions.relax(40)

    # Не ответили — ожидание не сбылось: досада и порыв заговорить снова.
    assert ignored.frustration > calm.frustration
    assert ignored.urge > calm.urge

    ignored.replied()
    assert not ignored.awaiting
