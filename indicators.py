# indicators.py
import pandas as pd
import numpy as np


# -----------------------------------------------------------------------------
# Moving averages
# -----------------------------------------------------------------------------

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


# -----------------------------------------------------------------------------
# RSI
# -----------------------------------------------------------------------------

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val.fillna(50)


# -----------------------------------------------------------------------------
# Bollinger Bands
# -----------------------------------------------------------------------------

def bollinger_bands(
    close: pd.Series,
    period: int = 20,
    std_factor: float = 2.0,
):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()

    upper = mid + std_factor * std
    lower = mid - std_factor * std
    return upper, mid, lower


# -----------------------------------------------------------------------------
# Stochastic Oscillator
# -----------------------------------------------------------------------------

def stochastic_oscillator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
):
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()

    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(d_period).mean()

    return k.fillna(50), d.fillna(50)


# -----------------------------------------------------------------------------
# ATR (Average True Range)
# -----------------------------------------------------------------------------

def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_val = tr.rolling(period).mean()
    return atr_val.fillna(method="bfill")


# -----------------------------------------------------------------------------
# ADX (+DI / -DI)
# -----------------------------------------------------------------------------

def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
):
    up_move = high.diff()
    down_move = low.diff().abs()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = atr(high, low, close, period)

    plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / tr
    minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / tr

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx_val = dx.rolling(period).mean()

    return (
        adx_val.fillna(0),
        plus_di.fillna(0),
        minus_di.fillna(0),
    )


# -----------------------------------------------------------------------------
# Candlestick patterns
# -----------------------------------------------------------------------------

def bullish_engulfing(open_: pd.Series, close: pd.Series) -> pd.Series:
    prev_open = open_.shift(1)
    prev_close = close.shift(1)

    return (
        (close > open_) &
        (prev_close < prev_open) &
        (close >= prev_open) &
        (open_ <= prev_close)
    )


def bearish_engulfing(open_: pd.Series, close: pd.Series) -> pd.Series:
    prev_open = open_.shift(1)
    prev_close = close.shift(1)

    return (
        (close < open_) &
        (prev_close > prev_open) &
        (close <= prev_open) &
        (open_ >= prev_close)
    )