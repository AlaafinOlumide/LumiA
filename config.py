# config.py
import os
from dataclasses import dataclass

@dataclass
class Settings:
    telegram_bot_token: str
    telegram_chat_id: str
    twelvedata_api_key: str

    # TwelveData symbol (FX format)
    xau_symbol_td: str = "XAU/USD"

    # Trading window (UTC HHMM)
    session_1_start: int = 700
    session_1_end: int = 2000

    # Weekend filter (fixes weekend signals)
    trade_weekends: bool = False  # <-- IMPORTANT

    # Candle / polling
    sleep_seconds: int = 60
    fetch_interval_seconds: int = 180  # data fetch cadence (not “signal cadence”)

    # Cooldown (prevents spam + revenge signals)
    cooldown_minutes: int = 20
    cooldown_same_direction_only: bool = False  # if True: block only same direction

    # Entry gate
    min_entry_score: int = 7  # 0..10

    # News
    news_window_minutes: int = 60

def load_settings() -> Settings:
    return Settings(
        telegram_bot_token=os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        telegram_chat_id=os.environ.get("TELEGRAM_CHAT_ID", ""),
        twelvedata_api_key=os.environ.get("TWELVEDATA_API_KEY", ""),
        xau_symbol_td=os.environ.get("XAU_SYMBOL_TWELVE", "XAU/USD"),
        session_1_start=int(os.environ.get("SESSION_1_START", "700")),
        session_1_end=int(os.environ.get("SESSION_1_END", "2000")),
        trade_weekends=str(os.environ.get("TRADE_WEEKENDS", "false")).lower() in {"1","true","yes"},
        sleep_seconds=int(os.environ.get("SLEEP_SECONDS", "60")),
        fetch_interval_seconds=int(os.environ.get("FETCH_INTERVAL_SECONDS", "180")),
        cooldown_minutes=int(os.environ.get("COOLDOWN_MINUTES", "20")),
        cooldown_same_direction_only=str(os.environ.get("COOLDOWN_SAME_DIR_ONLY", "false")).lower() in {"1","true","yes"},
        min_entry_score=int(os.environ.get("MIN_ENTRY_SCORE", "7")),
        news_window_minutes=int(os.environ.get("NEWS_WINDOW_MINUTES", "60")),
    )
