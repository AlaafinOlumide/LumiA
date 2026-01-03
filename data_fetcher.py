# data_fetcher.py
from __future__ import annotations

import logging
import time
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger("data_fetcher")


def fetch_m5_ohlcv_twelvedata(
    symbol: str,
    api_key: str,
    outputsize: int = 2000,
    max_retries: int = 3,
    retry_sleep_seconds: int = 2,
) -> pd.DataFrame:
    """
    Fetch M5 OHLCV from Twelve Data.
    Returns a DataFrame with columns:
      datetime (UTC tz-aware), open, high, low, close, volume

    Notes:
    - TwelveData returns datetime strings (exchange local time). We parse and force UTC to keep pipeline stable.
    - Adds retries to survive occasional API hiccups / Render network blips.
    """
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": "5min",
        "outputsize": outputsize,
        "apikey": api_key,
        "format": "JSON",
    }

    last_err: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.info("Calling Twelve Data (%s) interval=5min outputsize=%s attempt=%s/%s", symbol, outputsize, attempt, max_retries)

            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            data = r.json()

            if "status" in data and data["status"] == "error":
                raise RuntimeError(f"TwelveData error: {data.get('message')}")

            values = data.get("values") or []
            if not values:
                return pd.DataFrame()

            df = pd.DataFrame(values)

            # Standardize numeric columns
            for c in ["open", "high", "low", "close", "volume"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

            # Parse datetime and force UTC
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)

            df = df.dropna(subset=["datetime", "open", "high", "low", "close"]).sort_values("datetime")

            if "volume" not in df.columns:
                df["volume"] = 0.0
            df["volume"] = df["volume"].fillna(0.0)

            return df.reset_index(drop=True)

        except Exception as e:
            last_err = e
            logger.warning("TwelveData fetch failed (attempt %s/%s): %s", attempt, max_retries, e)
            if attempt < max_retries:
                time.sleep(retry_sleep_seconds)

    raise RuntimeError(f"Failed to fetch data from TwelveData after {max_retries} attempts. Last error: {last_err}")