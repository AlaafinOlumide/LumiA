# config.py
from __future__ import annotations

from dataclasses import dataclass
import os
from dotenv import load_dotenv

# On Render: env vars come from Render dashboard.
# load_dotenv() is harmless locally, but Render does not rely on .env in repo.
load_dotenv()


@dataclass
class Settings:
    # API keys / tokens
    twelvedata_api_key: str
    telegram_bot_token: str
    telegram_chat_id: str

    # TwelveData symbol
    xau_symbol_td: str = "XAU/USD"

    # Data pulling
    outputsize: int = 2000  # IMPORTANT for H1/M15 stability
    fetch_interval_seconds: int = 180
    sleep_seconds: int = 60

    # Candle safety: ignore newest candle (may be forming)
    ignore_latest_candle: bool = True

    # News filter
    news_window_minutes: int = 60

    # Trading sessions (UTC)
    # Wider defaults for gold
    enable_asia_session: bool = True
    asia_start_hour_utc: int = 0
    asia_end_hour_utc: int = 3

    london_start_hour_utc: int = 6
    london_end_hour_utc: int = 21

    trade_weekends: bool = False

    # Cooldown
    cooldown_minutes: int = 30
    cooldown_same_direction_only: bool = False

    # Entry gate
    min_entry_score: int = 5
    asia_extra_score_buffer: int = 1

    # Risk / TP tuning
    tp1_rr: float = 1.4
    tp2_rr: float = 2.5
    asia_tp1_rr: float = 1.0
    sl_atr_mult: float = 1.2

    # Momentum override (to avoid missing dumps/spikes)
    enable_momentum_override: bool = True
    momentum_override_adx_m5: float = 35.0


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

        outputsize=int(os.getenv("OUTPUTSIZE", "2000")),
        fetch_interval_seconds=int(os.getenv("FETCH_INTERVAL_SECONDS", "180")),
        sleep_seconds=int(os.getenv("SLEEP_SECONDS", "60")),
        ignore_latest_candle=os.getenv("IGNORE_LATEST_CANDLE", "true").lower() in ("1", "true", "yes"),

        news_window_minutes=int(os.getenv("NEWS_WINDOW_MINUTES", "60")),

        enable_asia_session=os.getenv("ENABLE_ASIA_SESSION", "true").lower() in ("1", "true", "yes"),
        asia_start_hour_utc=int(os.getenv("ASIA_START_HOUR_UTC", "0")),
        asia_end_hour_utc=int(os.getenv("ASIA_END_HOUR_UTC", "3")),
        london_start_hour_utc=int(os.getenv("LONDON_START_HOUR_UTC", "6")),
        london_end_hour_utc=int(os.getenv("LONDON_END_HOUR_UTC", "21")),
        trade_weekends=os.getenv("TRADE_WEEKENDS", "false").lower() in ("1", "true", "yes"),

        cooldown_minutes=int(os.getenv("COOLDOWN_MINUTES", "30")),
        cooldown_same_direction_only=os.getenv("COOLDOWN_SAME_DIRECTION_ONLY", "false").lower() in ("1", "true", "yes"),

        min_entry_score=int(os.getenv("MIN_ENTRY_SCORE", "5")),
        asia_extra_score_buffer=int(os.getenv("ASIA_EXTRA_SCORE_BUFFER", "1")),

        tp1_rr=float(os.getenv("TP1_RR", "1.4")),
        tp2_rr=float(os.getenv("TP2_RR", "2.5")),
        asia_tp1_rr=float(os.getenv("ASIA_TP1_RR", "1.0")),
        sl_atr_mult=float(os.getenv("SL_ATR_MULT", "1.2")),

        enable_momentum_override=os.getenv("ENABLE_MOMENTUM_OVERRIDE", "true").lower() in ("1", "true", "yes"),
        momentum_override_adx_m5=float(os.getenv("MOMENTUM_OVERRIDE_ADX_M5", "35.0")),
    )# config.py

from dataclasses import dataclass
import os
from dotenv import load_dotenv

# On Render: env vars come from Render dashboard.
# load_dotenv() is harmless locally, but Render does not rely on .env in repo.
load_dotenv()


@dataclass
class Settings:
    # API keys / tokens
    twelvedata_api_key: str
    telegram_bot_token: str
    telegram_chat_id: str

    # TwelveData symbol
    xau_symbol_td: str = "XAU/USD"

    # Data pulling
    outputsize: int = 2000  # IMPORTANT for H1/M15 stability
    fetch_interval_seconds: int = 180
    sleep_seconds: int = 60

    # Candle safety: ignore newest candle (may be forming)
    ignore_latest_candle: bool = True

    # News filter
    news_window_minutes: int = 60

    # Trading sessions (UTC)
    # Wider defaults for gold
    enable_asia_session: bool = True
    asia_start_hour_utc: int = 0
    asia_end_hour_utc: int = 3

    london_start_hour_utc: int = 6
    london_end_hour_utc: int = 21

    trade_weekends: bool = False

    # Cooldown
    cooldown_minutes: int = 30
    cooldown_same_direction_only: bool = False

    # Entry gate
    min_entry_score: int = 5
    asia_extra_score_buffer: int = 1

    # Risk / TP tuning
    tp1_rr: float = 1.4
    tp2_rr: float = 2.5
    asia_tp1_rr: float = 1.0
    sl_atr_mult: float = 1.2

    # Momentum override (to avoid missing dumps/spikes)
    enable_momentum_override: bool = True
    momentum_override_adx_m5: float = 35.0


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

        outputsize=int(os.getenv("OUTPUTSIZE", "2000")),
        fetch_interval_seconds=int(os.getenv("FETCH_INTERVAL_SECONDS", "180")),
        sleep_seconds=int(os.getenv("SLEEP_SECONDS", "60")),
        ignore_latest_candle=os.getenv("IGNORE_LATEST_CANDLE", "true").lower() in ("1", "true", "yes"),

        news_window_minutes=int(os.getenv("NEWS_WINDOW_MINUTES", "60")),

        enable_asia_session=os.getenv("ENABLE_ASIA_SESSION", "true").lower() in ("1", "true", "yes"),
        asia_start_hour_utc=int(os.getenv("ASIA_START_HOUR_UTC", "0")),
        asia_end_hour_utc=int(os.getenv("ASIA_END_HOUR_UTC", "3")),
        london_start_hour_utc=int(os.getenv("LONDON_START_HOUR_UTC", "6")),
        london_end_hour_utc=int(os.getenv("LONDON_END_HOUR_UTC", "21")),
        trade_weekends=os.getenv("TRADE_WEEKENDS", "false").lower() in ("1", "true", "yes"),

        cooldown_minutes=int(os.getenv("COOLDOWN_MINUTES", "30")),
        cooldown_same_direction_only=os.getenv("COOLDOWN_SAME_DIRECTION_ONLY", "false").lower() in ("1", "true", "yes"),

        min_entry_score=int(os.getenv("MIN_ENTRY_SCORE", "5")),
        asia_extra_score_buffer=int(os.getenv("ASIA_EXTRA_SCORE_BUFFER", "1")),

        tp1_rr=float(os.getenv("TP1_RR", "1.4")),
        tp2_rr=float(os.getenv("TP2_RR", "2.5")),
        asia_tp1_rr=float(os.getenv("ASIA_TP1_RR", "1.0")),
        sl_atr_mult=float(os.getenv("SL_ATR_MULT", "1.2")),

        enable_momentum_override=os.getenv("ENABLE_MOMENTUM_OVERRIDE", "true").lower() in ("1", "true", "yes"),
        momentum_override_adx_m5=float(os.getenv("MOMENTUM_OVERRIDE_ADX_M5", "35.0")),
    )