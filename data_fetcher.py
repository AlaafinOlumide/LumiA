import logging
import pandas as pd
import requests

logger = logging.getLogger(__name__)

TD_BASE_URL = "https://api.twelvedata.com/time_series"


def fetch_ohlcv_twelvedata(
    api_key: str,
    symbol: str,
    interval: str,
    outputsize: int = 300,
) -> pd.DataFrame:
    """
    Fetch OHLCV data from Twelve Data and return as DataFrame sorted ascending.
    interval example: "5min", "15min", "1h".
    """
    params = {
        "symbol": symbol,
        "interval": interval,
        "apikey": api_key,
        "outputsize": outputsize,
        "format": "JSON",
    }
    logger.info("Calling Twelve Data for %s interval %s", symbol, interval)

    r = requests.get(TD_BASE_URL, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()

    if "values" not in data:
        raise RuntimeError(f"Twelve Data response missing 'values': {data}")

    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)

    if "volume" in df.columns:
        df["volume"] = df["volume"].astype(float)
    else:
        df["volume"] = 0.0

    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def fetch_m5_ohlcv_twelvedata(
    symbol: str,
    api_key: str,
    outputsize: int = 300,
) -> pd.DataFrame:
    """
    M5-only convenience wrapper (keeps your main.py import working).
    """
    if not api_key:
        raise RuntimeError("Twelve Data API key missing; cannot fetch data.")

    return fetch_ohlcv_twelvedata(
        api_key=api_key,
        symbol=symbol,
        interval="5min",
        outputsize=outputsize,
    )
