# data_fetcher.py

import logging
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

TD_BASE_URL = "https://api.twelvedata.com/time_series"


def fetch_ohlcv_twelvedata(
    api_key: str,
    symbol: str,
    interval: str,
    outputsize: int = 1200,   # IMPORTANT: enough for H1=60 from M5
) -> pd.DataFrame:
    """
    Fetch OHLCV data from Twelve Data and return as DataFrame sorted by time ascending.
    Forces timezone UTC (prevents future-dated candles).
    """
    params = {
        "symbol": symbol,
        "interval": interval,
        "apikey": api_key,
        "outputsize": outputsize,
        "timezone": "UTC",     # IMPORTANT: ask TwelveData for UTC timestamps
        "format": "JSON",
    }

    logger.info("Calling Twelve Data for %s interval %s", symbol, interval)
    r = requests.get(TD_BASE_URL, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()

    if "values" not in data:
        raise RuntimeError(f"Twelve Data response missing 'values': {data}")

    df = pd.DataFrame(data["values"])

    # Parse timestamps as UTC
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime"])

    # Ensure correct dtypes
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)

    if "volume" in df.columns:
        df["volume"] = df["volume"].astype(float)
    else:
        df["volume"] = 0.0

    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def fetch_m5_ohlcv_twelvedata(symbol: str, api_key: str) -> pd.DataFrame:
    """
    M5 fetch using Twelve Data only.
    """
    if not api_key:
        raise RuntimeError("Twelve Data API key missing; cannot fetch data.")

    return fetch_ohlcv_twelvedata(
        api_key=api_key,
        symbol=symbol,         # e.g. "XAU/USD"
        interval="5min",
        outputsize=1200,       # ~100 hours of M5 candles -> enough for 60 H1
    )
