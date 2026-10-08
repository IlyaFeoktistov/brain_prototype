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
    """
    Свежий откалиброванный мозг. Калибровка — основная цена (tiny 1.4 с,
    small 3 с), загрузка — 0.03 / 0.1 с: каждый размер калибруется один
    раз за сессию, тест получает загруженную копию. Копия — всегда, даже
    в первый раз, чтобы все тесты стартовали из одного состояния.
    """

    key = "-".join([size] + [f"{k}={v}" for k, v in sorted(overrides.items())])
    born = os.path.join(_TMP, f"born-{key}.pt")

    if not os.path.exists(born):
        cfg = BrainConfig()
        cfg.size = size
        cfg.calibration_tokens = 150

        for name, value in overrides.items():
            setattr(cfg, name, value)

        mind = Mind(cfg)
        mind.calibrate()
        mind.save(born)

    mind = Mind.load(born)
    # Сохранение без пути не должно затирать общий «новорождённый» мозг.
    mind.cfg.save_path = os.path.join(_TMP, f"{size}.pt")

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
