# config.py
import os
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass
class Settings:
    # Core API keys / IDs
    twelvedata_api_key: str
    telegram_bot_token: str
    telegram_chat_id: str

    # Symbol
    xau_symbol_td: str = "XAU/USD"

    # Trading session (UTC) as HHMM
    session_1_start: int = 700
    session_1_end: int = 2000

    # Execution intervals
    sleep_seconds: int = 60                  # loop sleep
    fetch_interval_seconds: int = 180        # force API refresh every N seconds (cache in-between)

    # Filters
    trade_weekends: bool = False             # prevent weekend signals
    news_window_minutes: int = 60            # +- minutes around high-impact news to flag/filter

    # Entry gate
    min_entry_score: int = 65                # score-based entry gate

    # Cooldown
    cooldown_minutes: int = 30
    cooldown_same_direction_only: bool = True

    # Journaling / invalidation (DROP 2)
    journal_path: str = "journal.csv"
    max_hold_minutes: int = 240              # expire signal after N minutes if no TP/SL hit
    invalidate_on_opposite_signal: bool = True
    journal_notify_telegram: bool = True


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default


def load_settings() -> Settings:
    load_dotenv()

    return Settings(
        twelvedata_api_key=os.getenv("TWELVEDATA_API_KEY", "").strip(),
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", "").strip(),

        xau_symbol_td=os.getenv("XAU_SYMBOL_TD", "XAU/USD").strip(),

        session_1_start=_env_int("SESSION_1_START", 700),
        session_1_end=_env_int("SESSION_1_END", 2000),

        sleep_seconds=_env_int("SLEEP_SECONDS", 60),
        fetch_interval_seconds=_env_int("FETCH_INTERVAL_SECONDS", 180),

        trade_weekends=_env_bool("TRADE_WEEKENDS", False),
        news_window_minutes=_env_int("NEWS_WINDOW_MINUTES", 60),

        min_entry_score=_env_int("MIN_ENTRY_SCORE", 65),

        cooldown_minutes=_env_int("COOLDOWN_MINUTES", 30),
        cooldown_same_direction_only=_env_bool("COOLDOWN_SAME_DIRECTION_ONLY", True),

        journal_path=os.getenv("JOURNAL_PATH", "journal.csv").strip(),
        max_hold_minutes=_env_int("MAX_HOLD_MINUTES", 240),
        invalidate_on_opposite_signal=_env_bool("INVALIDATE_ON_OPPOSITE_SIGNAL", True),
        journal_notify_telegram=_env_bool("JOURNAL_NOTIFY_TELEGRAM", True),
    )