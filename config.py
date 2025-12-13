# config.py
import os
from dataclasses import dataclass


@dataclass
class Settings:
    telegram_bot_token: str
    telegram_chat_id: str
    twelvedata_api_key: str

    # Twelve Data symbol
    xau_symbol_td: str = "XAU/USD"

    # Trading windows in UTC (HHMM)
    session_1_start: int = 700
    session_1_end: int = 2000
    session_2_start: int = 0
    session_2_end: int = 0

    # Polling / fetching
    poll_seconds: int = 60                # loop sleep
    fetch_interval_seconds: int = 120     # fetch every 2 mins (your request)

    # Cooldown
    cooldown_minutes: int = 20

    # Market open filter
    block_weekends: bool = True
    # Many brokers reopen Sunday night (UTC). Keep simple & safe:
    sunday_open_hour_utc: int = 22        # allow trading only after 22:00 UTC Sunday

    # Invalidation
    invalidation_minutes: int = 20        # how long after entry we allow invalidation checks

    # Journal (live + backtest)
    journal_csv_path: str = "trades.csv"
    journal_rolling_n: int = 60


def _get_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name, str(default)).strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def load_settings() -> Settings:
    return Settings(
        telegram_bot_token=os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        telegram_chat_id=os.environ.get("TELEGRAM_CHAT_ID", ""),
        twelvedata_api_key=os.environ.get("TWELVEDATA_API_KEY", ""),
        xau_symbol_td=os.environ.get("XAU_SYMBOL_TWELVE", "XAU/USD"),
        session_1_start=int(os.environ.get("SESSION_1_START", "700")),
        session_1_end=int(os.environ.get("SESSION_1_END", "2000")),
        poll_seconds=int(os.environ.get("POLL_SECONDS", "60")),
        fetch_interval_seconds=int(os.environ.get("FETCH_INTERVAL_SECONDS", "120")),
        cooldown_minutes=int(os.environ.get("COOLDOWN_MINUTES", "20")),
        block_weekends=_get_bool("BLOCK_WEEKENDS", True),
        sunday_open_hour_utc=int(os.environ.get("SUNDAY_OPEN_HOUR_UTC", "22")),
        invalidation_minutes=int(os.environ.get("INVALIDATION_MINUTES", "20")),
        journal_csv_path=os.environ.get("JOURNAL_CSV_PATH", "trades.csv"),
        journal_rolling_n=int(os.environ.get("JOURNAL_ROLLING_N", "60")),
    )
