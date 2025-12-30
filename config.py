# config.py
from __future__ import annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import os


@dataclass
class Settings:
    # --- API / symbols ---
    twelvedata_api_key: str
    xau_symbol_td: str = "XAU/USD"

    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # --- run-loop ---
    sleep_seconds: int = 60
    fetch_interval_seconds: int = 180  # refresh TD data every N seconds

    # --- trading window (UTC) ---
    # Hard window (MOST IMPORTANT) – prevents night signals even if helper is bypassed
    hard_session_start_hour_utc: int = 7
    hard_session_end_hour_utc: int = 20  # exclusive (7 <= hour < 20)

    # Optional HHMM session helper (kept for message formatting + future split sessions)
    session_1_start: int = 700
    session_1_end: int = 2000

    # weekend filter
    trade_weekends: bool = False

    # --- news ---
    news_window_minutes: int = 60

    # --- signal gating ---
    min_entry_score: int = 70

    # Liquidity / range filter (removes most low-quality night chop)
    range_filter_lookback: int = 20
    range_filter_min_ratio: float = 0.60  # last_range must be >= 0.60 * avg_range

    # --- cooldown ---
    cooldown_same_direction_only: bool = False
    cooldown_minutes_default: int = 10  # fallback if regime-based not used

    # If True, cooldown minutes are selected based on ADX(H1)
    use_regime_based_cooldown: bool = True


def _get_env(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return v if v is not None else default


def load_settings() -> Settings:
    load_dotenv()

    api_key = _get_env("TWELVEDATA_API_KEY", "")
    if not api_key:
        raise RuntimeError("Missing TWELVEDATA_API_KEY in environment (.env or Render env vars).")

    return Settings(
        twelvedata_api_key=api_key,
        xau_symbol_td=_get_env("XAU_SYMBOL_TD", "XAU/USD"),
        telegram_bot_token=_get_env("TELEGRAM_BOT_TOKEN", ""),
        telegram_chat_id=_get_env("TELEGRAM_CHAT_ID", ""),
        sleep_seconds=int(_get_env("SLEEP_SECONDS", "60")),
        fetch_interval_seconds=int(_get_env("FETCH_INTERVAL_SECONDS", "180")),
        hard_session_start_hour_utc=int(_get_env("HARD_SESSION_START_HOUR_UTC", "7")),
        hard_session_end_hour_utc=int(_get_env("HARD_SESSION_END_HOUR_UTC", "20")),
        session_1_start=int(_get_env("SESSION_1_START", "700")),
        session_1_end=int(_get_env("SESSION_1_END", "2000")),
        trade_weekends=_get_env("TRADE_WEEKENDS", "false").lower() in ("1", "true", "yes", "y"),
        news_window_minutes=int(_get_env("NEWS_WINDOW_MINUTES", "60")),
        min_entry_score=int(_get_env("MIN_ENTRY_SCORE", "70")),
        range_filter_lookback=int(_get_env("RANGE_FILTER_LOOKBACK", "20")),
        range_filter_min_ratio=float(_get_env("RANGE_FILTER_MIN_RATIO", "0.60")),
        cooldown_same_direction_only=_get_env("COOLDOWN_SAME_DIRECTION_ONLY", "false").lower() in ("1", "true", "yes"),
        cooldown_minutes_default=int(_get_env("COOLDOWN_MINUTES_DEFAULT", "10")),
        use_regime_based_cooldown=_get_env("USE_REGIME_BASED_COOLDOWN", "true").lower() in ("1", "true", "yes"),
    )