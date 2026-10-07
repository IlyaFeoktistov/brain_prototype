"""
Общие фикстуры.

Окружение задаётся ДО импорта brain.*: часть значений BrainConfig
читается из переменных окружения при импорте.
"""

import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="brain-tests-")
os.environ.setdefault("BRAIN_SIZE", "tiny")
os.environ["BRAIN_SAVE"] = os.path.join(_TMP, "server.pt")

import pytest  # noqa: E402

from brain.config import BrainConfig  # noqa: E402
from brain.mind import Mind  # noqa: E402


FACTS = """
Столица Франции Париж.
Столица Германии Берлин.
Столица Италии Рим.
Земля вращается вокруг Солнца.
Луна вращается вокруг Земли.
Человек пьёт воду.
"""


def make_mind(size: str = "tiny", **overrides) -> Mind:
    cfg = BrainConfig()
    cfg.size = size
    cfg.save_path = os.path.join(_TMP, f"{size}.pt")
    cfg.calibration_tokens = 150

    for key, value in overrides.items():
        setattr(cfg, key, value)

    mind = Mind(cfg)
    mind.calibrate()

    return mind


@pytest.fixture
def mind() -> Mind:
    """Свежий маленький мозг (2k нейронов)."""

    return make_mind()


@pytest.fixture(scope="session")
def trained() -> Mind:
    """
    Мозг на 20k нейронов, который 6 раз прочитал FACTS.
    Общий для тестов обучения — они его не меняют (только пробы).
    """

    mind = make_mind("small")

    for _ in range(6):
        mind.language.read_text(FACTS)

    return mind


@pytest.fixture
def tmp_dir():
    return tempfile.mkdtemp(prefix="brain-test-", dir=_TMP)
