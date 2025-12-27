import requests
import pandas as pd
from datetime import timezone
import logging

log = logging.getLogger("data_fetcher")

TWELVEDATA_URL = "https://api.twelvedata.com/time_series"

def fetch_ohlcv(symbol: str, interval: str, apikey: str, outputsize: int = 300) -> pd.DataFrame:
    """
    Returns DataFrame indexed by UTC timestamp with columns:
    open, high, low, close, volume
    """
    params = {
        "symbol": symbol,
        "interval": interval,
        "apikey": apikey,
        "outputsize": outputsize,
        "format": "JSON",
        "timezone": "UTC",
    }
    log.info("Calling Twelve Data for %s interval %s", symbol, interval)
    r = requests.get(TWELVEDATA_URL, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    if "status" in data and data["status"] == "error":
        raise RuntimeError(f"TwelveData error: {data.get('message')}")

    values = data.get("values", [])
    if not values:
        raise RuntimeError("No OHLCV returned from TwelveData.")

    df = pd.DataFrame(values)
    # TwelveData returns strings; convert
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    else:
        df["volume"] = 0.0

    df = df.sort_values("datetime").set_index("datetime")
    df = df.dropna(subset=["open", "high", "low", "close"])
    df.index = df.index.tz_convert(timezone.utc)
    return df
