import requests
import logging
import pandas as pd

logger = logging.getLogger(__name__)

BASE_URL = "https://api.twelvedata.com/time_series"


def fetch_ohlcv(api_key: str, symbol: str, interval: str, outputsize: int = 150) -> pd.DataFrame:
    """
    Fetch OHLCV data from Twelve Data and return as DataFrame sorted by time ascending.
    """
    params = {
        "symbol": symbol,
        "interval": interval,
        "apikey": api_key,
        "outputsize": outputsize,
    }
    r = requests.get(BASE_URL, params=params, timeout=15)
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
