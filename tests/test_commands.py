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


def test_quit_returns_false(mind):
    session, _ = make_session(mind)
    assert session.handle("/quit") is False
