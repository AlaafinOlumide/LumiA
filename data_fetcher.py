import logging
from typing import Optional

import pandas as pd
import requests

from config import Settings

logger = logging.getLogger(__name__)

TD_BASE_URL = "https://api.twelvedata.com/time_series"


def fetch_ohlcv_twelvedata(
    api_key: str,
    symbol: str,
    interval: str,
    outputsize: int = 300,
) -> pd.DataFrame:
    """
    Fetch OHLCV data from Twelve Data and return as DataFrame sorted by time ascending.
    interval example: "5min", "15min", "1h".
    """
    params = {
        "symbol": symbol,
        "interval": interval,
        "apikey": api_key,
        "outputsize": outputsize,
    }
    logger.info("Calling Twelve Data for %s interval %s", symbol, interval)
    r = requests.get(TD_BASE_URL, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()

    if "values" not in data:
        raise RuntimeError(f"Twelve Data response missing 'values': {data}")

    df = pd.DataFrame(data["values"])
    # Ensure correct dtypes
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    if "volume" in df.columns:
        df["volume"] = df["volume"].astype(float)
    else:
        df["volume"] = 0.0

    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def fetch_m5_ohlcv_hybrid(settings: Settings) -> pd.DataFrame:
    """
    Previously: yfinance primary, Twelve Data fallback.
    Now: Twelve Data ONLY (XAU/USD, 5min).

    Kept the function name the same so main.py does not need to change imports.
    """
    if not settings.twelvedata_api_key:
        raise RuntimeError("Twelve Data API key missing; cannot fetch data.")

    df_td = fetch_ohlcv_twelvedata(
        api_key=settings.twelvedata_api_key,
        symbol=settings.xau_symbol_td,  # e.g. "XAU/USD"
        interval="5min",
        outputsize=300,  # ~25 hours of 5m candles
    )
    logger.info("Using Twelve Data data for symbol %s", settings.xau_symbol_td)
    return df_td
