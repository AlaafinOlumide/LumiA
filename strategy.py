from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any
import pandas as pd
import datetime as dt

from indicators import ema, rsi, bollinger_bands, stochastic_oscillator, adx, bullish_engulfing, bearish_engulfing

Direction = Literal["LONG", "SHORT"]

@dataclass
class Signal:
    direction: Direction
    price: float
    time: dt.datetime
    reason: str
    extra: Dict[str, Any]

def is_within_sessions(now_utc: dt.datetime, s1_start: int, s1_end: int, s2_start: int, s2_end: int) -> bool:
    hhmm = now_utc.hour * 100 + now_utc.minute
    return (s1_start <= hhmm <= s1_end) or (s2_start <= hhmm <= s2_end)

def detect_trend_h1(h1_df: pd.DataFrame) -> Optional[Direction]:
    # Use EMA + ADX + DI to detect main trend on H1.
    df = h1_df.copy()
    df["ema_fast"] = ema(df["close"], 50)
    adx_series, plus_di, minus_di = adx(df["high"], df["low"], df["close"], period=14)
    last = df.iloc[-1]
    last_adx = adx_series.iloc[-1]
    last_plus = plus_di.iloc[-1]
    last_minus = minus_di.iloc[-1]

    if last_adx < 20:
        return None  # weak trend

    if last["close"] > last["ema_fast"] and last_plus > last_minus:
        return "LONG"
    if last["close"] < last["ema_fast"] and last_minus > last_plus:
        return "SHORT"
    return None

def confirm_trend_m15(m15_df: pd.DataFrame, direction: Direction) -> bool:
    df = m15_df.copy()
    df["ema_fast"] = ema(df["close"], 50)
    df["rsi"] = rsi(df["close"], 14)
    last = df.iloc[-1]

    if direction == "LONG":
        return last["close"] > last["ema_fast"] and last["rsi"] > 50
    else:
        return last["close"] < last["ema_fast"] and last["rsi"] < 50

def trigger_signal_m5(m5_df: pd.DataFrame, direction: Direction) -> Optional[Signal]:
    df = m5_df.copy()
    upper, mid, lower = bollinger_bands(df["close"], 20, 2.0)
    df["bb_upper"] = upper
    df["bb_mid"] = mid
    df["bb_lower"] = lower
    df["rsi"] = rsi(df["close"], 14)
    k, d = stochastic_oscillator(df["high"], df["low"], df["close"], 14, 3)
    df["stoch_k"] = k
    df["stoch_d"] = d
    adx_series, plus_di, minus_di = adx(df["high"], df["low"], df["close"], 14)
    df["adx"] = adx_series
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di
    df["bullish_engulf"] = bullish_engulfing(df["open"], df["close"])
    df["bearish_engulf"] = bearish_engulfing(df["open"], df["close"])

    last = df.iloc[-1]

    # Require at least moderate trend strength on M5 as well
    if last["adx"] < 18:
        return None

    reason = None

    if direction == "LONG":
        cond_bb = last["close"] <= last["bb_mid"]  # pullback
        cond_rsi = last["rsi"] < 45
        cond_stoch = last["stoch_k"] < 30
        cond_candle = last["bullish_engulf"]
        if cond_bb and cond_rsi and cond_stoch and cond_candle:
            reason = "M5 pullback in uptrend (BB mid/lower), RSI oversold, Stoch oversold, bullish engulfing."
    else:
        cond_bb = last["close"] >= last["bb_mid"]
        cond_rsi = last["rsi"] > 55
        cond_stoch = last["stoch_k"] > 70
        cond_candle = last["bearish_engulf"]
        if cond_bb and cond_rsi and cond_stoch and cond_candle:
            reason = "M5 pullback in downtrend (BB mid/upper), RSI overbought, Stoch overbought, bearish engulfing."

    if not reason:
        return None

    return Signal(
        direction=direction,
        price=float(last["close"]),
        time=pd.to_datetime(last["datetime"]).to_pydatetime(),
        reason=reason,
        extra={
            "m5_rsi": float(last["rsi"]),
            "m5_stoch_k": float(last["stoch_k"]),
            "m5_stoch_d": float(last["stoch_d"]),
            "bb_upper": float(last["bb_upper"]),
            "bb_mid": float(last["bb_mid"]),
            "bb_lower": float(last["bb_lower"]),
            "adx_m5": float(last["adx"]),
            "plus_di_m5": float(last["plus_di"]),
            "minus_di_m5": float(last["minus_di"]),
        },
    )
