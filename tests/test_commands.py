"""Команды чата (общие для терминала и веба)."""

import os

from brain.commands import Session


def make_session(mind):
    output = []
    session = Session(mind, lambda text, kind: output.append((kind, text)))
    return session, output


def test_read_inline_and_say(mind):
    session, output = make_session(mind)
    session.handle("/read: Столица Франции Париж. Столица Германии Берлин.")
    session.handle("Столица Франции")

    assert any(kind == "brain" for kind, _ in output)
    assert mind.language.experienced_tokens > 0


def test_unknown_command(mind):
    session, output = make_session(mind)
    session.handle("/абракадабра")
    assert "Неизвестная команда" in output[-1][1]


def test_brains_list_and_switch(mind, tmp_dir):
    other_path = os.path.join(tmp_dir, "other.pt")
    mind.save(other_path)
    mind.cfg.save_path = os.path.join(tmp_dir, "first.pt")
    mind.save()

    session, output = make_session(mind)
    session.handle("/brains")
    assert "other.pt" in output[-1][1]

    session.handle("/load other.pt")
    assert session.mind.cfg.save_path == other_path


def test_emotions_toggle_command(mind):
    session, output = make_session(mind)
    session.handle("/emotions off")
    assert mind.cfg.emotions_enabled is False
    session.handle("/emotions on")
    assert mind.cfg.emotions_enabled is True


def test_feature_command_lists_and_toggles(mind):
    session, output = make_session(mind)
    session.handle("/feature")
    assert "dendritic_context" in output[-1][1]

    session.handle("/feature thinking_enabled on")
    assert mind.cfg.thinking_enabled is True
    session.handle("/feature thinking_enabled off")
    assert mind.cfg.thinking_enabled is False

    session.handle("/feature нет_такой on")
    assert "Нет такой функции" in output[-1][1]


def test_dendritic_context_switches_on_the_fly(mind):
    """Флаг меняет шаг нейрона: после переключения мозг читает и отвечает."""

    session, _ = make_session(mind)
    mind.language.read_text("Столица Франции Париж. Столица Италии Рим.")

    session.handle("/feature dendritic_context on")
    assert mind.cfg.dendritic_context is True
    report = mind.language.read_text("Столица Франции Париж.")
    assert report.tokens > 0
    assert float(mind.brain.i_dend.abs().sum()) > 0

    session.handle("/feature dendritic_context off")
    mind.language.read_text("Столица Италии Рим.")


def test_quit_returns_false(mind):
    session, _ = make_session(mind)
    assert session.handle("/quit") is False


def test_neural_speech_answers_from_spikes():
    """Ответ нейронами: слово — место, выстрелившее при открытых воротах."""

    from tests.conftest import make_mind

    mind = make_mind("small")
    mind.set_feature("dendritic_context", True)
    mind.set_feature("neural_speech", True)
    language = mind.language

    for _ in range(6):
        language.read_text("Столица Франции Париж. Столица Италии Рим.")

    # [замер] «столица италии» здесь пока тоже → «париж»: контекст на
    # дендритах на маленьком мозге и 6 прочтениях слабее формулы.
    said = [token for token, _ in language.generate("столица франции")]
    assert said and said[0] == "париж"


def test_sleep_in_web_falls_asleep_wakeably(mind):
    """В вебе /sleep не держит мозг: засыпает порциями (его будят)."""

    session, output = make_session(mind)
    session.fall_asleep = lambda: "засыпаю порциями"
    session.handle("/sleep")
    assert output[-1][1] == "засыпаю порциями"


def test_terminal_sleeps_in_silence_and_wakes_up(mind):
    """Терминал в тишине засыпает порциями и высыпается."""

    import queue
    from brain.chat import _idle
    from brain.live import Liveliness

    session, _ = make_session(mind)
    session.handle("/read: Кошка мяукает. Собака лает. Солнце светит.")
    mind.cfg.sleepy_after_episodes = 1
    mind.cfg.autosave = False
    pressure = mind.language.sleep_pressure
    assert pressure > 0

    sleep_left = _idle(session, Liveliness(), 0.0, 0, queue.Queue())
    assert sleep_left == 0 or mind.language.emotions.sleeping

    for _ in range(100):
        if not sleep_left:
            break
        sleep_left = _idle(session, Liveliness(), 0.0, sleep_left, queue.Queue())

    assert sleep_left == 0
    assert mind.language.sleep_pressure == 0
    assert not mind.language.emotions.sleeping

    # Сообщение в очереди будит: порция сна не начинается.
    lines = queue.Queue()
    lines.put("привет")
    assert _idle(session, Liveliness(), 0.0, 5, lines) == 5


def test_dreams_list_what_was_replayed(mind):
    session, output = make_session(mind)
    session.handle("/dreams")
    assert output[-1][1] == "Ещё не спал"

    session.handle("/read: Кошка мяукает.")
    session.handle("/sleep 2")
    session.handle("/dreams")
    assert "кошка мяукает" in output[-1][1]


def test_trace_shows_paths_and_choice(mind):
    session, output = make_session(mind)
    session.handle("/read: Столица Франции Париж. Столица Франции Париж.")
    before = mind.brain.net_weight.clone()
    session.handle("/trace столица франции")
    text = output[-1][1]

    assert "«франции»" in text and "ворота «говори»" in text
    assert (before == mind.brain.net_weight).all(), "трассировка не меняет мозг"
