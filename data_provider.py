import requests
import pandas as pd
import os

TD_API_KEY = os.getenv("TWELVEDATA_API_KEY")

def get_xauusd_data(interval: str = "5min", outputsize: int = 200) -> pd.DataFrame:
    """Fetch XAU/USD OHLC data from TwelveData as a pandas DataFrame."""
    if TD_API_KEY is None:
        raise RuntimeError("TWELVEDATA_API_KEY not set in environment variables")

    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": "XAU/USD",
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TD_API_KEY,
    }
    r = requests.get(url, params=params, timeout=15).json()

    if "values" not in r:
        raise Exception(f"TwelveData error: {r}")

    df = pd.DataFrame(r["values"])
    df = df.iloc[::-1].reset_index(drop=True)

    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)

    return df
