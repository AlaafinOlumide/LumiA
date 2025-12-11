from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import datetime as dt

import pandas as pd

from indicators import (
    ema,
    rsi,
    bollinger_bands,
    stochastic_oscillator,
    adx,
    bullish_engulfing,
    bearish_engulfing,
)


@dataclass
class Signal:
    direction: str  # "LONG" or "SHORT"
    price: float
    time: dt.datetime
    reason: str
    extra: Dict[str, Any] = field(default_factory=dict)


def is_within_sessions(
    now_utc: dt.datetime,
    session_1_start: int,
    session_1_end: int,
    session_2_start: Optional[int],
    session_2_end: Optional[int],
) -> bool:
    """
    Check if current time (UTC) falls within either session window.

    session_*_* are HHMM integers, e.g. 700, 2000.
    """
    hhmm = now_utc.hour * 100 + now_utc.minute

    in_session_1 = session_1_start <= hhmm <= session_1_end

    in_session_2 = False
    if session_2_start is not None and session_2_end is not None:
        in_session_2 = session_2_start <= hhmm <= session_2_end

    return in_session_1 or in_session_2


# ---------------------------------------------------------------------------
# H1 TREND DETECTION (PRIMARY)
# ---------------------------------------------------------------------------

def detect_trend_h1(h1_df: pd.DataFrame) -> Optional[str]:
    """
    Detect H1 trend direction.

    More forgiving logic:
    - Uses EMA(50) vs EMA(200) for direction.
    - Uses ADX(14) for strength, but only blocks when both:
        * EMAs are flat/tangled, AND
        * ADX is very low.
    Returns:
        "LONG", "SHORT", or None if genuinely directionless.
    """
    if h1_df is None or h1_df.empty or len(h1_df) < 50:
        return None

    close = h1_df["close"]
    ema_fast = ema(close, 50)
    ema_slow = ema(close, 200)

    ema_fast_last = float(ema_fast.iloc[-1])
    ema_slow_last = float(ema_slow.iloc[-1])
    close_last = float(close.iloc[-1])

    # ADX on H1
    adx_series, plus_di, minus_di = adx(
        h1_df["high"], h1_df["low"], h1_df["close"], period=14
    )
    adx_last = float(adx_series.iloc[-1])

    # How "separated" are the EMAs?
    ema_diff = ema_fast_last - ema_slow_last
    ema_diff_ratio = abs(ema_diff) / close_last if close_last > 0 else 0.0

    # If EMAs are basically on top of each other AND ADX is very low -> no trend.
    if ema_diff_ratio < 0.001 and adx_last < 12:  # <0.1% apart & very weak ADX
        return None

    # Direction from EMA relationship
    if ema_fast_last > ema_slow_last:
        return "LONG"
    elif ema_fast_last < ema_slow_last:
        return "SHORT"
    else:
        # EMAs truly equal & ADX not terrible? allow a weak bias from +DI / -DI
        plus_last = float(plus_di.iloc[-1])
        minus_last = float(minus_di.iloc[-1])
        if adx_last >= 15:
            if plus_last > minus_last:
                return "LONG"
            elif minus_last > plus_last:
                return "SHORT"
        # Still nothing convincing
        return None


# ---------------------------------------------------------------------------
# M15 TREND DETECTION (FALLBACK WHEN H1 IS RANGING)
# ---------------------------------------------------------------------------

def detect_trend_m15_direction(m15_df: pd.DataFrame) -> Optional[str]:
    """
    Detect M15 trend direction, used as a fallback when H1 has no clear trend.

    Uses EMA(20) vs EMA(50) + ADX(14).
    Returns:
        "LONG", "SHORT", or None.
    """
    if m15_df is None or m15_df.empty or len(m15_df) < 50:
        return None

    close = m15_df["close"]
    ema_fast = ema(close, 20)
    ema_slow = ema(close, 50)

    ema_fast_last = float(ema_fast.iloc[-1])
    ema_slow_last = float(ema_slow.iloc[-1])
    close_last = float(close.iloc[-1])

    adx_series, _, _ = adx(
        m15_df["high"], m15_df["low"], m15_df["close"], period=14
    )
    adx_last = float(adx_series.iloc[-1])

    ema_diff = ema_fast_last - ema_slow_last
    ema_diff_ratio = abs(ema_diff) / close_last if close_last > 0 else 0.0

    # If EMAs are very flat/tight AND ADX is very weak -> no real trend here either.
    if ema_diff_ratio < 0.0008 and adx_last < 10:
        return None

    if ema_fast_last > ema_slow_last:
        return "LONG"
    elif ema_fast_last < ema_slow_last:
        return "SHORT"

    return None


# ---------------------------------------------------------------------------
# M15 CONFIRMATION (USED WHEN H1 IS PRIMARY)
# ---------------------------------------------------------------------------

def confirm_trend_m15(m15_df: pd.DataFrame, trend_h1: str) -> bool:
    """
    Confirm H1 trend using M15 structure.

    Basic idea:
    - Use EMA(20) and EMA(50) on M15.
    - For LONG trend: EMA20 > EMA50 and close > EMA20.
    - For SHORT trend: EMA20 < EMA50 and close < EMA20.
    - ADX(14) on M15 should not be extremely weak.
    """
    if m15_df is None or m15_df.empty or len(m15_df) < 50:
        return False

    close = m15_df["close"]
    ema_fast = ema(close, 20)
    ema_slow = ema(close, 50)

    ema_fast_last = float(ema_fast.iloc[-1])
    ema_slow_last = float(ema_slow.iloc[-1])
    close_last = float(close.iloc[-1])

    adx_series, _, _ = adx(
        m15_df["high"], m15_df["low"], m15_df["close"], period=14
    )
    adx_last = float(adx_series.iloc[-1])

    # Avoid ultra-weak conditions
    if adx_last < 10:
        return False

    if trend_h1 == "LONG":
        return (ema_fast_last > ema_slow_last) and (close_last > ema_fast_last)
    elif trend_h1 == "SHORT":
        return (ema_fast_last < ema_slow_last) and (close_last < ema_fast_last)

    return False


# ---------------------------------------------------------------------------
# M5 TRIGGER (LOOSER: MORE SIGNALS, STILL IN TREND DIRECTION)
# ---------------------------------------------------------------------------

def trigger_signal_m5(m5_df: pd.DataFrame, trend_for_signal: str) -> Optional[Signal]:
    """
    Generate a LONG/SHORT signal on M5 in the direction of the higher-timeframe bias.

    Loosened conditions for more signals:
    - ADX floor lowered from 8 -> 5
    - Wider "near band" definition
    - Momentum less strict
    - Candle filter: engulfing OR strong candle in trend direction.
    """
    if m5_df is None or m5_df.empty or len(m5_df) < 50:
        return None

    # Use the last COMPLETED candle (second-to-last row)
    row_idx = m5_df.index[-2]

    close_series = m5_df["close"]
    high_series = m5_df["high"]
    low_series = m5_df["low"]
    open_series = m5_df["open"]

    # Indicators
    rsi_series = rsi(close_series, period=14)
    stoch_k, stoch_d = stochastic_oscillator(
        high_series, low_series, close_series, k_period=14, d_period=3
    )
    bb_upper, bb_mid, bb_lower = bollinger_bands(
        close_series, period=20, std_factor=2.0
    )
    adx_series, plus_di, minus_di = adx(
        high_series, low_series, close_series, period=14
    )
    bull_engulf = bullish_engulfing(open_series, close_series)
    bear_engulf = bearish_engulfing(open_series, close_series)

    # Values on the last completed bar
    rsi_val = float(rsi_series.loc[row_idx])
    stoch_k_val = float(stoch_k.loc[row_idx])
    stoch_d_val = float(stoch_d.loc[row_idx])
    bb_upper_val = float(bb_upper.loc[row_idx])
    bb_mid_val = float(bb_mid.loc[row_idx])
    bb_lower_val = float(bb_lower.loc[row_idx])
    adx_val = float(adx_series.loc[row_idx])
    plus_di_val = float(plus_di.loc[row_idx])
    minus_di_val = float(minus_di.loc[row_idx])
    is_bull_engulf = bool(bull_engulf.loc[row_idx])
    is_bear_engulf = bool(bear_engulf.loc[row_idx])

    close_val = float(close_series.loc[row_idx])
    open_val = float(open_series.loc[row_idx])
    time_val = pd.to_datetime(m5_df.loc[row_idx, "datetime"]).to_pydatetime()

    extra = {
        "m5_rsi": rsi_val,
        "m5_stoch_k": stoch_k_val,
        "m5_stoch_d": stoch_d_val,
        "bb_upper": bb_upper_val,
        "bb_mid": bb_mid_val,
        "bb_lower": bb_lower_val,
        "adx_m5": adx_val,
        "plus_di_m5": plus_di_val,
        "minus_di_m5": minus_di_val,
    }

    # Slightly lower ADX floor to allow more trades, but still avoid pure noise
    if adx_val < 5:
        return None

    # Helper: strong candles in direction of trend
    is_strong_bull = close_val > open_val and close_val > bb_mid_val
    is_strong_bear = close_val < open_val and close_val < bb_mid_val

    # LONG setup: pullback in uptrend + bullish confirmation
    if trend_for_signal == "LONG":
        # Price near mid/lower band = pullback (wider tolerance)
        near_band = (close_val <= bb_mid_val * 1.01) or (close_val <= bb_lower_val * 1.02)
        # Momentum: allow slightly higher RSI / Stoch for more opportunities
        momentum_ok = (rsi_val < 65) and (stoch_k_val < 80)
        # Directional ADX bias
        adx_bias_up = plus_di_val >= minus_di_val
        # Candle confirmation: engulfing OR strong bullish bar
        candle_ok = is_bull_engulf or is_strong_bull

        if near_band and momentum_ok and adx_bias_up and candle_ok:
            reason = (
                "Looser LONG: M5 pullback in uptrend (BB mid/lower, wider tolerance), "
                "RSI < 65, StochK < 80, bullish engulfing or strong bullish candle."
            )
            return Signal(
                direction="LONG",
                price=close_val,
                time=time_val,
                reason=reason,
                extra=extra,
            )

    # SHORT setup: pullback in downtrend + bearish confirmation
    if trend_for_signal == "SHORT":
        # Price near mid/upper band = pullback (wider tolerance)
        near_band = (close_val >= bb_mid_val * 0.99) or (close_val >= bb_upper_val * 0.98)
        momentum_ok = (rsi_val > 35) and (stoch_k_val > 20)
        adx_bias_down = minus_di_val >= plus_di_val
        candle_ok = is_bear_engulf or is_strong_bear

        if near_band and momentum_ok and adx_bias_down and candle_ok:
            reason = (
                "Looser SHORT: M5 pullback in downtrend (BB mid/upper, wider tolerance), "
                "RSI > 35, StochK > 20, bearish engulfing or strong bearish candle."
            )
            return Signal(
                direction="SHORT",
                price=close_val,
                time=time_val,
                reason=reason,
                extra=extra,
            )

    return None
