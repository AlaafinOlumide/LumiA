# data_fetcher.py
from __future__ import annotations
import os
import requests
import pandas as pd

def get_ohlcv_df(symbol: str, interval: str, outputsize: int = 500) -> pd.DataFrame:
    """
    Fetch OHLCV and return a DataFrame.
    Update the body to match your provider (TwelveData/AlphaVantage/etc.)
    """
    api_key = os.getenv("TWELVEDATA_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError("Missing API key env var: TWELVEDATA_API_KEY (or API_KEY)")

    # Example for Twelve Data (adjust if your endpoint differs)
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": api_key,
        "format": "JSON",
    }

    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    if "values" not in data:
        raise RuntimeError(f"Bad response: {data}")

    df = pd.DataFrame(data["values"])
    # Typical TwelveData returns strings; convert
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values("datetime").reset_index(drop=True)
    return df