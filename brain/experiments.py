"""
Проверка фундамента без чата:

    python -m brain.experiments [размер]

1. Разреженность: какая доля нейронов каждой зоны отвечает на слово.
2. Скорость: токенов в секунду.
3. Обучение: prediction error и угадывание при повторном чтении.
4. Рабочая память: «открыть файл» vs «закрыть файл».
5. Read-only: /predict не меняет веса.
"""

from __future__ import annotations

import sys
import time

import torch

from brain.config import BrainConfig
from brain.mind import Mind


TEXT = """
Человек может открыть файл для чтения.
Человек может закрыть файл после записи.
Столица Франции Париж.
Столица Германии Берлин.
Столица Италии Рим.
Земля вращается вокруг Солнца.
Луна вращается вокруг Земли.
"""


def sparsity(mind: Mind) -> None:
    probe = mind.language.probe("париж")
    print("1. Разреженность (доля активных E на слово «париж»):")

    for area in mind.brain.areas.values():
        rates = probe["rates"][area.E.slice]
        active = int((rates > 0).sum())
        print(
            f"   {area.name:12s} {active:6d}/{area.E.size:<7d} "
            f"({active / area.E.size:.1%}), средняя частота активных "
            f"{float(rates[rates > 0].mean()) if active else 0:.0f} Hz"
        )


def main() -> None:
    cfg = BrainConfig()
    cfg.save_path = "/dev/null"

    if len(sys.argv) > 1:
        cfg.size = sys.argv[1]

    start = time.time()
    mind = Mind(cfg)
    mind.calibrate()
    language = mind.language
    print(
        f"мозг: {mind.brain.n:,} нейронов, {mind.brain.synapse_count():,} синапсов, "
        f"{mind.brain.memory_bytes() / 2**20:.0f} МБ, {mind.brain.device}, "
        f"создан за {time.time() - start:.1f} c"
    )

    # Прогрев: мозг должен один раз услышать слова, чтобы readout знал формы.
    language.read_text(TEXT)
    sparsity(mind)

    print("\n2-3. Обучение повторным чтением:")

    for epoch in range(1, 9):
        start = time.time()
        report = language.read_text(TEXT)
        elapsed = time.time() - start
        print(
            f"   проход {epoch}: error {report.mean_error():.3f}, "
            f"угадано {report.hits}/{len(report.errors)}, "
            f"dopamine {report.dopamine:5.1f}, {report.tokens / elapsed:.0f} ток/с"
        )

    print("\n4. Рабочая память:")
    print("   " + language.compare("открыть файл", "закрыть файл").replace("\n", "\n   "))

    print("\n   Ожидания:")
    for phrase in ("столица франции", "столица германии", "луна вращается вокруг", "земля вращается вокруг"):
        print("   " + language.debug_prediction(phrase).replace("\n", "\n   "))

    print("\n5. Read-only:")
    before = [p.weight.clone() for p in mind.brain.plastic]
    tokens = language.experienced_tokens
    language.debug_prediction("столица франции")
    language.compare("открыть файл", "закрыть файл")
    language.debug_word("париж")
    same = all(torch.equal(a, p.weight) for a, p in zip(before, mind.brain.plastic))
    print(f"   веса не изменились: {same}, readout не изменился: {tokens == language.experienced_tokens}")


if __name__ == "__main__":
    main()
