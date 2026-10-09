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
    emotions.valence, emotions.arousal, emotions.confusion, emotions.novelty_hunger = 0.0, 0.05, 0.0, 0.9


def _joyful(emotions):
    emotions.valence, emotions.arousal, emotions.confusion, emotions.novelty_hunger = 0.6, 0.6, 0.0, 0.0


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


def test_fresh_memory_fades_without_sleep(mind):
    brain = mind.brain
    mind.language.read_text(FACTS)
    fresh = float(brain.early.abs().sum())
    assert fresh > 0

    for _ in range(mind.cfg.early_ltp_half_life):
        brain.forget()

    # За полураспад свежая часть уходит вдвое (с поправкой на округление).
    assert float(brain.early.abs().sum()) < fresh * 0.55


def test_sleep_consolidates_fresh_memory(mind):
    brain, language = mind.brain, mind.language
    language.read_text(FACTS)
    plastic = slice(0, brain.n_plastic)

    def consolidated() -> float:
        # Выученное, которое не затухнет: вес без свежей части.
        return float((brain.net_weight[plastic] - brain.early - brain.birth_weight).abs().sum())

    before = consolidated()
    language.sleep(replays=len(language.hippocampus))

    assert consolidated() > before * 2
    assert brain.consolidating is False


def test_forgetting_can_be_switched_off(mind):
    brain = mind.brain
    mind.language.read_text(FACTS)
    mind.cfg.forgetting_enabled = False
    weights = brain.net_weight.clone()

    brain.forget()

    assert (brain.net_weight == weights).all()


def test_long_reading_naps_to_keep_memory(mind):
    language = mind.language
    mind.cfg.nap_after_episodes = 8
    text = " ".join(f"Слово{i} идёт за словом{i + 1}." for i in range(20))

    report = language.read_text(text)

    assert report.naps >= 2
    # Проспал — новых непроигранных впечатлений меньше порога.
    assert language.sleep_pressure < mind.cfg.nap_after_episodes


def test_places_forget_words_heard_once_not_repeated():
    """Места кончились: забывается слышанное однажды, повторённое — нет."""

    import torch
    from brain.concepts import Concepts

    c = Concepts(8, 2, 0, "cpu", reuse=True, keep=frozenset({"."}))
    for word in ("кот", "кот", ".", "пёс", "лис"):
        c.get(word, allocate=True)
    assert len(c) == 4

    c.get("ёж", allocate=True)
    assert "пёс" not in c and c.evicted[0][0] == "пёс"
    assert {"кот", ".", "лис", "ёж"} <= set(c.places)
    assert (c.usage <= 1).all()

    # Слышанных однажды нет — места делятся, мозг сообщает, что тесно.
    for word in ("лис", "ёж"):
        c.get(word, allocate=True)
    c.get("уж", allocate=True)
    assert c.full and len(c) == 5


def test_forgotten_word_releases_synapses(mind):
    language = mind.language
    language.read_text("Кошка мяукает. Кошка мяукает.")
    place = language.concepts.get("мяукает", allocate=False)
    proj = next(p for p in mind.brain.plastic if p.post is language.meaning.E and p.pre_pop is language.meaning.E)
    rows = proj.weight[place]
    assert (rows != proj.birth[place].float()).any()

    language.concepts.places.pop("мяукает")
    language.concepts.evicted.append(("мяукает", place))
    language._forget_evicted()

    assert (proj.weight[place] == proj.birth[place].float()).all()
    assert "мяукает" not in language.semantic
