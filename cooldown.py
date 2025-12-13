# cooldown.py
import datetime as dt
from typing import Optional


def in_cooldown(now_utc: dt.datetime, last_signal_time: Optional[dt.datetime], minutes: int = 20) -> bool:
    """
    Prevents rapid consecutive signals.
    - last_signal_time: when last signal/trade was opened OR closed
    """
    if last_signal_time is None:
        return False
    if last_signal_time.tzinfo is None:
        last_signal_time = last_signal_time.replace(tzinfo=dt.timezone.utc)
    return (now_utc - last_signal_time) < dt.timedelta(minutes=minutes)
