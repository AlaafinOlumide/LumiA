# data_fetcher.py
import logging
import pandas as pd
import requests

logger = logging.getLogger(__name__)

TD_BASE_URL = "https://api.twelvedata.com/time_series"


def fetch_ohlcv_twelvedata(
    api_key: str,
    symbol: str,
    interval: str,
    outputsize: int = 1200,
) -> pd.DataFrame:
    """
    Fetch OHLCV from Twelve Data and return DataFrame sorted by time ascending.
    Forces timezone UTC.
    """
    params = {
        "symbol": symbol,
        "interval": interval,
        "apikey": api_key,
        "outputsize": outputsize,
        "timezone": "UTC",
        "format": "JSON",
    }

    logger.info("Calling Twelve Data for %s interval %s", symbol, interval)
    r = requests.get(TD_BASE_URL, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()

    if "values" not in data:
        raise RuntimeError(f"Twelve Data response missing 'values': {data}")

    df = pd.DataFrame(data["values"])

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime"])

    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)

    if "volume" in df.columns:
        df["volume"] = df["volume"].astype(float)
    else:
        df["volume"] = 0.0

    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def fetch_m5_ohlcv_twelvedata(symbol: str, api_key: str) -> pd.DataFrame:
    """Convenience wrapper for M5 candles."""
    if not api_key:
        raise RuntimeError("Twelve Data API key missing; cannot fetch data.")
    return fetch_ohlcv_twelvedata(
        api_key=api_key,
        symbol=symbol,
        interval="5min",
        outputsize=1200,
    )