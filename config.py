# config.py
from __future__ import annotations

from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    # API keys / tokens
    twelvedata_api_key: str
    telegram_bot_token: str
    telegram_chat_id: str

    # TwelveData symbol (your repo seems to use XAU/USD)
    xau_symbol_td: str = "XAU/USD"

    # Loop / caching
    fetch_interval_seconds: int = 180   # fetch fresh data every 3 minutes
    sleep_seconds: int = 60            # check loop every 60s

    # News filter
    news_window_minutes: int = 60

    # Trading sessions
    # We hard-enforce TWO windows:
    #   ASIA:      00:00–02:00 UTC
    #   LONDON_NY: 07:00–20:00 UTC
    enable_asia_session: bool = True
    asia_start_hour_utc: int = 0
    asia_end_hour_utc: int = 2

    london_start_hour_utc: int = 7
    london_end_hour_utc: int = 20

    trade_weekends: bool = False

    # Cooldown
    cooldown_minutes: int = 30
    cooldown_same_direction_only: bool = False

    # Entry gate
    min_entry_score: int = 6
    asia_extra_score_buffer: int = 2  # ASIA requires min_entry_score + this

    # Risk / TP tuning
    tp1_rr: float = 1.4
    tp2_rr: float = 2.5
    asia_tp1_rr: float = 1.0          # ASIA: smaller targets
    sl_atr_mult: float = 1.2


def _get_required(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


def load_settings() -> Settings:
    return Settings(
        twelvedata_api_key=_get_required("TWELVEDATA_API_KEY"),
        telegram_bot_token=_get_required("TELEGRAM_BOT_TOKEN"),
        telegram_chat_id=_get_required("TELEGRAM_CHAT_ID"),
        xau_symbol_td=os.getenv("XAU_SYMBOL_TD", "XAU/USD"),
        fetch_interval_seconds=int(os.getenv("FETCH_INTERVAL_SECONDS", "180")),
        sleep_seconds=int(os.getenv("SLEEP_SECONDS", "60")),
        news_window_minutes=int(os.getenv("NEWS_WINDOW_MINUTES", "60")),
        enable_asia_session=os.getenv("ENABLE_ASIA_SESSION", "true").lower() in ("1", "true", "yes"),
        trade_weekends=os.getenv("TRADE_WEEKENDS", "false").lower() in ("1", "true", "yes"),
        cooldown_minutes=int(os.getenv("COOLDOWN_MINUTES", "30")),
        cooldown_same_direction_only=os.getenv("COOLDOWN_SAME_DIRECTION_ONLY", "false").lower() in ("1", "true", "yes"),
        min_entry_score=int(os.getenv("MIN_ENTRY_SCORE", "6")),
        asia_extra_score_buffer=int(os.getenv("ASIA_EXTRA_SCORE_BUFFER", "2")),
        tp1_rr=float(os.getenv("TP1_RR", "1.4")),
        tp2_rr=float(os.getenv("TP2_RR", "2.5")),
        asia_tp1_rr=float(os.getenv("ASIA_TP1_RR", "1.0")),
        sl_atr_mult=float(os.getenv("SL_ATR_MULT", "1.2")),
    )