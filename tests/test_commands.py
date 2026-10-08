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
