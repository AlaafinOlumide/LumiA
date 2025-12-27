import numpy as np
import pandas as pd

def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()

def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    roll_up = up.ewm(alpha=1/n, adjust=False).mean()
    roll_down = down.ewm(alpha=1/n, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)

def stoch_kd(high: pd.Series, low: pd.Series, close: pd.Series, k_n: int = 14, d_n: int = 3, smooth: int = 3):
    ll = low.rolling(k_n).min()
    hh = high.rolling(k_n).max()
    k = 100 * (close - ll) / (hh - ll).replace(0, np.nan)
    k = k.rolling(smooth).mean()
    d = k.rolling(d_n).mean()
    return k.fillna(50.0), d.fillna(50.0)

def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr_val = tr.ewm(alpha=1/n, adjust=False).mean()
    # replaces deprecated fillna(method="bfill")
    return atr_val.bfill()

def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = close.rolling(n).mean()
    std = close.rolling(n).std(ddof=0)
    upper = mid + k * std
    lower = mid - k * std
    return upper.bfill(), mid.bfill(), lower.bfill()

def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14):
    # Minimal ADX implementation (good enough for regime gating)
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat([
        (high - low),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    atr_n = tr.ewm(alpha=1/n, adjust=False).mean().replace(0, np.nan)
    plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(alpha=1/n, adjust=False).mean() / atr_n
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(alpha=1/n, adjust=False).mean() / atr_n
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0.0)
    adx_v = dx.ewm(alpha=1/n, adjust=False).mean().bfill()
    return adx_v, plus_di.bfill(), minus_di.bfill()
