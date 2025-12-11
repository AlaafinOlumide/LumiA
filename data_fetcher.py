import logging
from typing import Optional

import pandas as pd
import requests
import yfinance as yf

from config import Settings

logger = logging.getLogger(__name__)

TD_BASE_URL = "https://api.twelvedata.com/time_series"


# ---------- Twelve Data fetch ----------

def fetch_ohlcv_twelvedata(
    api_key: str,
    symbol: str,
    interval: str,
    outputsize: int = 150,
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


# ---------- Yahoo Finance fetch ----------

def fetch_ohlcv_yfinance(
    symbol: str,
    interval: str = "5m",
    period: str = "7d",
) -> pd.DataFrame:
    """
    Fetch OHLCV from Yahoo Finance using yfinance.

    symbol example: "XAUUSD=X"
    interval example: "5m"
    period example: "7d"
    """
    logger.info("Calling Yahoo Finance (yfinance) for %s interval %s period %s", symbol, interval, period)
    df = yf.download(
        symbol,
        interval=interval,
        period=period,
        progress=False,
    )

    if df is None or df.empty:
        raise RuntimeError("Empty DataFrame from yfinance")

    # yfinance returns columns: Open, High, Low, Close, Adj Close, Volume
    df = df.reset_index()  # Make datetime a column named "Datetime"
    # Normalize column names to match the rest of the bot
    rename_map = {
        "Datetime": "datetime",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    df = df.rename(columns=rename_map)

    # Ensure dtype consistency
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)

    df = df.sort_values("datetime").reset_index(drop=True)
    return df


# ---------- Hybrid fetch for M5 ----------

def fetch_m5_ohlcv_hybrid(settings: Settings) -> pd.DataFrame:
    """
    Hybrid data fetch:
    1) Try Yahoo Finance 5m data (XAUUSD=X).
    2) If that fails, fall back to Twelve Data 5min (XAU/USD).

    Returns a 5m OHLCV DataFrame with columns:
    datetime, open, high, low, close, volume
    """
    # Try yfinance first
    try:
        df_yf = fetch_ohlcv_yfinance(
            symbol=settings.xau_symbol_yf,
            interval="5m",
            period="7d",
        )
        if df_yf is not None and not df_yf.empty:
            logger.info("Using yfinance data for symbol %s", settings.xau_symbol_yf)
            return df_yf
        logger.warning("yfinance returned empty data, falling back to Twelve Data.")
    except Exception as e:
        logger.warning("yfinance fetch failed (%s), falling back to Twelve Data.", e)

    # Fallback: Twelve Data
    if not settings.twelvedata_api_key:
        raise RuntimeError(
            "Twelve Data API key missing and yfinance failed; cannot fetch data."
        )

    df_td = fetch_ohlcv_twelvedata(
        api_key=settings.twelvedata_api_key,
        symbol=settings.xau_symbol_td,
        interval="5min",
        outputsize=300,  # ~25 hours of data
    )
    logger.info("Using Twelve Data data for symbol %s", settings.xau_symbol_td)
    return df_td
