import pandas as pd

def bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    df["BB_MID"] = df["close"].rolling(period).mean()
    df["BB_STD"] = df["close"].rolling(period).std()
    df["BB_UPPER"] = df["BB_MID"] + std_dev * df["BB_STD"]
    df["BB_LOWER"] = df["BB_MID"] - std_dev * df["BB_STD"]
    return df

def rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    low_min = df["low"].rolling(k_period).min()
    high_max = df["high"].rolling(k_period).max()
    df["%K"] = 100 * (df["close"] - low_min) / (high_max - low_min)
    df["%D"] = df["%K"].rolling(d_period).mean()
    return df

def atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(period).mean()
    return df
