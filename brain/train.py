"""
Обучение из командной строки (в фоне, на часы):

    python -m brain.train --size 300000 --save data/brain-max.pt путь [путь ...]

Мозг загружается из --save, если файл уже есть, иначе создаётся.
Каждый путь — файл или папка с любыми текстовыми файлами. После каждого
файла мозг сохраняется, поэтому прерванное обучение не теряется.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time

from brain.commands import Session
from brain.config import BrainConfig
from brain.mind import Mind


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--size", default=None, help="число нейронов или tiny/small/medium/large")
    parser.add_argument("--save", default=None, help="файл мозга")
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--sleep", action="store_true", help="после чтения поспать (консолидация)")
    args = parser.parse_args()

    cfg = BrainConfig()

    if args.size:
        cfg.size = args.size

    if args.save:
        cfg.save_path = args.save

    def out(text: str, kind: str) -> None:
        if kind != "progress" or int(time.time()) % 30 == 0:
            print(time.strftime("%H:%M:%S"), text, flush=True)

    mind, restored = Mind.open(cfg)
    print(
        ("Загружен: " if restored else "Новый мозг: ")
        + f"{mind.cfg.save_path}, {mind.brain.n:,} нейронов, {mind.brain.synapse_count():,} синапсов",
        flush=True,
    )

    session = Session(mind, out)
    signal.signal(signal.SIGTERM, lambda *_: session.stop.set())

    try:
        for path in args.paths:
            session.read(f"{path} {args.passes}")

            if session.stop.is_set():
                break

        if args.sleep and not session.stop.is_set():
            session.sleep()
    except KeyboardInterrupt:
        session.stop.set()
    finally:
        print(f"Сохранено: {mind.save()}", flush=True)
        print(mind.language.stats(), flush=True)

    sys.exit(0)


if __name__ == "__main__":
    main()
