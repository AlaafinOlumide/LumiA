# config.py
from __future__ import annotations

from dataclasses import dataclass
import os
from dotenv import load_dotenv


@dataclass
class Settings:
    # API / Symbols
    twelvedata_api_key: str
    xau_symbol_td: str

    # Telegram
    telegram_bot_token: str
    telegram_chat_id: str

    # Timing
    sleep_seconds: int
    fetch_interval_seconds: int

    # Sessions / Weekends
    session_1_start: int  # HHMM
    session_1_end: int    # HHMM
    session_2_start: int | None
    session_2_end: int | None
    trade_weekends: bool

    # Cooldown
    cooldown_minutes: int
    cooldown_same_direction_only: bool

    # Filters
    news_window_minutes: int
    min_entry_score: int

    # Score weights
    score_h1_trend: int
    score_m15_structure: int
    score_pullback_zone: int
    score_rsi_reset: int
    score_stoch_reset: int
    score_rejection: int
    score_adx_ok: int
    score_no_news: int

    # Journaling / evaluation
    journal_path: str
    eval_delay_minutes: int        # how long after signal before we start evaluating
    max_hold_minutes: int          # close as EXPIRED after this


def _get_int(name: str, default: int) -> int:
    v = os.getenv(name)
    return int(v) if v is not None and v != "" else default


def _get_float(name: str, default: float) -> float:
    v = os.getenv(name)
    return float(v) if v is not None and v != "" else default


def _get_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_settings() -> Settings:
    load_dotenv()

    return Settings(
        # API / Symbols
        twelvedata_api_key=os.getenv("TWELVEDATA_API_KEY", "").strip(),
        xau_symbol_td=os.getenv("XAU_SYMBOL_TD", "XAU/USD").strip(),

        # Telegram
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", "").strip(),

        # Timing
        sleep_seconds=_get_int("SLEEP_SECONDS", 60),
        fetch_interval_seconds=_get_int("FETCH_INTERVAL_SECONDS", 180),

        # Sessions
        session_1_start=_get_int("SESSION_1_START", 700),
        session_1_end=_get_int("SESSION_1_END", 2000),
        session_2_start=(int(os.getenv("SESSION_2_START")) if os.getenv("SESSION_2_START") else None),
        session_2_end=(int(os.getenv("SESSION_2_END")) if os.getenv("SESSION_2_END") else None),
        trade_weekends=_get_bool("TRADE_WEEKENDS", False),

        # Cooldown
        cooldown_minutes=_get_int("COOLDOWN_MINUTES", 30),
        cooldown_same_direction_only=_get_bool("COOLDOWN_SAME_DIRECTION_ONLY", True),

        # Filters
        news_window_minutes=_get_int("NEWS_WINDOW_MINUTES", 60),
        min_entry_score=_get_int("MIN_ENTRY_SCORE", 65),

        # Score weights
        score_h1_trend=_get_int("SCORE_H1_TREND", 20),
        score_m15_structure=_get_int("SCORE_M15_STRUCTURE", 20),
        score_pullback_zone=_get_int("SCORE_PULLBACK_ZONE", 10),
        score_rsi_reset=_get_int("SCORE_RSI_RESET", 15),
        score_stoch_reset=_get_int("SCORE_STOCH_RESET", 15),
        score_rejection=_get_int("SCORE_REJECTION", 10),
        score_adx_ok=_get_int("SCORE_ADX_OK", 10),
        score_no_news=_get_int("SCORE_NO_NEWS", 10),

        # Journaling
        journal_path=os.getenv("JOURNAL_PATH", "journal.csv").strip(),
        eval_delay_minutes=_get_int("EVAL_DELAY_MINUTES", 15),
        max_hold_minutes=_get_int("MAX_HOLD_MINUTES", 240),
    )