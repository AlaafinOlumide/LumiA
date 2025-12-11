import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.clip(lower=0)).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val


def bollinger_bands(series: pd.Series, period: int = 20, std_factor: float = 2.0):
    ma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = ma + std_factor * std
    lower = ma - std_factor * std
    return upper, ma, lower


def stochastic_oscillator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
):
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=d_period).mean()
    return k, d


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    """
    Simplified ADX implementation.
    """
    plus_dm = high.diff()
    minus_dm = low.diff().abs() * -1
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr_val = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).sum() / atr_val)
    minus_di = 100 * (minus_dm.abs().rolling(window=period).sum() / atr_val)
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di)) * 100
    adx_series = dx.rolling(window=period).mean()
    return adx_series, plus_di, minus_di


def bullish_engulfing(open_: pd.Series, close: pd.Series):
    prev_open = open_.shift(1)
    prev_close = close.shift(1)
    cond_prev_bear = prev_close < prev_open
    cond_curr_bull = close > open_
    cond_engulf = (close >= prev_open) & (open_ <= prev_close)
    return cond_prev_bear & cond_curr_bull & cond_engulf


def bearish_engulfing(open_: pd.Series, close: pd.Series):
    prev_open = open_.shift(1)
    prev_close = close.shift(1)
    cond_prev_bull = prev_close > prev_open
    cond_curr_bear = close < open_
    cond_engulf = (close <= prev_open) & (open_ >= prev_close)
    return cond_prev_bull & cond_curr_bear & cond_engulf


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Average True Range (ATR).
    """
    high_low = high - low
    high_close_prev = (high - close.shift()).abs()
    low_close_prev = (low - close.shift()).abs()
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()
