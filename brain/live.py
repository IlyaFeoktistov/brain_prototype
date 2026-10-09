"""
Живой режим: когда мозг в тишине думает, заговаривает сам и засыпает.

Общие правила для веба (brain/server.py) и терминала (brain/chat.py).
Пока это таймеры и пороги в коде — долг ROADMAP P0 «без if-else» (8):
мозг должен крутиться непрерывно, а решения — приниматься нейронами.
"""

from __future__ import annotations

import time

from brain.mind import Mind


class Liveliness:
    """Когда мозг последний раз говорил сам и думал — и пора ли снова."""

    def __init__(self):
        self.spoke_up_at = 0.0
        self.wandered_at = 0.0

    def should_speak_up(self, mind: Mind, idle_since: float, listened: bool, busy: bool) -> bool:
        cfg = mind.cfg
        now = time.time()
        # Порыв из потребностей, а не таймер; таймеры — лишь «не перебивать»
        # и рефрактерность.
        return (
            cfg.live_mode
            and cfg.initiative_enabled
            and listened
            and not busy
            and mind.language.emotions.urge >= cfg.initiative_urge
            and now - idle_since >= cfg.initiative_idle_seconds
            and now - self.spoke_up_at >= cfg.initiative_cooldown_seconds
        )

    def should_wander(self, mind: Mind, idle_since: float, busy: bool) -> bool:
        cfg = mind.cfg
        now = time.time()
        return (
            cfg.live_mode
            and not busy
            and now - idle_since >= cfg.wander_every_seconds
            and now - self.wandered_at >= cfg.wander_every_seconds
        )

    @staticmethod
    def should_fall_asleep(mind: Mind, idle_since: float, reading: bool) -> bool:
        """Засыпает, только когда устал (накопились впечатления) и тихо."""

        cfg = mind.cfg
        return (
            not reading
            and cfg.sleep_when_idle
            and time.time() - idle_since >= cfg.idle_sleep_seconds
            and mind.language.sleep_pressure >= cfg.sleepy_after_episodes
        )
