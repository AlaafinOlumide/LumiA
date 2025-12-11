import os
from dataclasses import dataclass

@dataclass
class Settings:
    telegram_bot_token: str
    telegram_chat_id: str
    twelvedata_api_key: str
    xau_symbol: str = "XAU/USD"

    # Trading windows in UTC (HHMM)
    # Single big window: 07:00–20:00 UTC
    session_1_start: int = 700
    session_1_end: int = 2000
    session_2_start: int = 0   # unused
    session_2_end: int = 0     # unused

def load_settings() -> Settings:
    return Settings(
        telegram_bot_token=os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        telegram_chat_id=os.environ.get("TELEGRAM_CHAT_ID", ""),
        twelvedata_api_key=os.environ.get("TWELVEDATA_API_KEY", ""),
        xau_symbol=os.environ.get("XAU_SYMBOL", "XAU/USD"),
    )
