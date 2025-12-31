# data_fetcher.py
from __future__ import annotations

import logging
import pandas as pd
import requests

logger = logging.getLogger("data_fetcher")


def fetch_m5_ohlcv_twelvedata(symbol: str, api_key: str, outputsize: int = 600) -> pd.DataFrame:
    """
    Fetch M5 OHLCV for XAU/USD from Twelve Data.
    Returns a DataFrame with columns:
      datetime (UTC tz-aware), open, high, low, close, volume
    """
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": "5min",
        "outputsize": outputsize,
        "apikey": api_key,
        "format": "JSON",
    }

    logger.info("Calling Twelve Data for %s interval 5min", symbol)

    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()

    if "status" in data and data["status"] == "error":
        raise RuntimeError(f"TwelveData error: {data.get('message')}")

    values = data.get("values") or []
    if not values:
        return pd.DataFrame()

    df = pd.DataFrame(values)

    # Standardize columns
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Twelve Data uses "datetime" as string in exchange local time.
    # We treat it as UTC to keep your pipeline stable.
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")

    df = df.dropna(subset=["datetime", "open", "high", "low", "close"]).sort_values("datetime")
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df["volume"] = df["volume"].fillna(0.0)

    return df.reset_index(drop=True)