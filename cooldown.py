import datetime as dt
from typing import Dict, Tuple


class CooldownManager:
    """
    Prevents signal spam.
    Cooldown is applied PER:
      - setup_type
      - direction
    """

    def __init__(self):
        self._last_signal_time: Dict[Tuple[str, str], dt.datetime] = {}

        # cooldown minutes per setup
        self.cooldown_minutes = {
            "PULLBACK_LONG": 20,
            "PULLBACK_SHORT": 20,
            "BREAKOUT_LONG": 30,
            "BREAKOUT_SHORT": 30,
            "BREAKOUT_CONT_LONG": 15,
            "BREAKOUT_CONT_SHORT": 15,
        }

    def can_fire(self, setup_type: str, direction: str, now: dt.datetime) -> bool:
        key = (setup_type, direction)
        cooldown = self.cooldown_minutes.get(setup_type, 20)

        last_time = self._last_signal_time.get(key)
        if last_time is None:
            return True

        elapsed = (now - last_time).total_seconds() / 60
        return elapsed >= cooldown

    def register(self, setup_type: str, direction: str, now: dt.datetime) -> None:
        key = (setup_type, direction)
        self._last_signal_time[key] = now
