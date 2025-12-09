import os
from dataclasses import dataclass

@dataclass
class Settings:
    telegram_bot_token: str
    telegram_chat_id: str
    twelvedata_api_key: str
    xau_symbol: str = "XAU/USD"

    # Trading windows in UTC (HHMM)
    session_1_start: int = 800
    session_1_end: int = 1000
    session_2_start: int = 1200
    session_2_end: int = 1600

def load_settings() -> Settings:
    return Settings(
        telegram_bot_token=os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        telegram_chat_id=os.environ.get("TELEGRAM_CHAT_ID", ""),
        twelvedata_api_key=os.environ.get("TWELVEDATA_API_KEY", ""),
        xau_symbol=os.environ.get("XAU_SYMBOL", "XAU/USD"),
    )
