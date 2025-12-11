import os
from dataclasses import dataclass

@dataclass
class Settings:
    telegram_bot_token: str
    telegram_chat_id: str
    twelvedata_api_key: str

    # Symbol for Twelve Data (FX format)
    xau_symbol_td: str = "XAU/USD"

    # Symbol for Yahoo Finance (yfinance) – updated to working gold spot symbol
    xau_symbol_yf: str = "XAU=X"

    # Trading windows in UTC (HHMM) – single big window: 07:00–20:00 UTC
    session_1_start: int = 700
    session_1_end: int = 2000
    session_2_start: int = 0   # unused
    session_2_end: int = 0     # unused


def load_settings() -> Settings:
    return Settings(
        telegram_bot_token=os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        telegram_chat_id=os.environ.get("TELEGRAM_CHAT_ID", ""),
        twelvedata_api_key=os.environ.get("TWELVEDATA_API_KEY", ""),
        xau_symbol_td=os.environ.get("XAU_SYMBOL_TWELVE", "XAU/USD"),
        xau_symbol_yf=os.environ.get("XAU_SYMBOL_YF", "XAU=X"),
    )
