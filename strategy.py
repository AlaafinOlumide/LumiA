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
    - Uses ADX(14) only to block when EMAs are very flat & ADX very low.

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
# M5 TRIGGER: PULLBACKS + MODERATE BREAKOUTS
# ---------------------------------------------------------------------------

def trigger_signal_m5(m5_df: pd.DataFrame, trend_for_signal: str) -> Optional[Signal]:
    """
    Generate a LONG/SHORT signal on M5 in the direction of the higher-timeframe bias.

    Supports TWO types of entries:
    - PULLBACK: price retraces to BB mid/lower (uptrend) / mid/upper (downtrend)
    - BREAKOUT: strong momentum candle through BB upper/lower in trend direction

    "Moderate" breakout mode:
    - Global ADX floor = 5 to avoid complete noise
    - Breakout-only ADX requirement ≈ 14+
    - Candle must close outside band in trend direction
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

    # Positional previous close for breakout body check
    prev_close = float(close_series.iloc[-3])

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

    # Slightly lower global ADX floor to allow more trades, but still avoid pure noise
    if adx_val < 5:
        return None

    # Helper: strong candles in direction of trend (for pullbacks)
    is_strong_bull = (close_val > open_val) and (close_val >= bb_mid_val)
    is_strong_bear = (close_val < open_val) and (close_val <= bb_mid_val)

    # -----------------------------------------------------------------------
    # LONG SIDE: PULLBACK + BREAKOUT
    # -----------------------------------------------------------------------
    if trend_for_signal == "LONG":
        # --- Pullback conditions ---
        pullback_near_band = (
            (close_val <= bb_mid_val * 1.01)  # slightly under / around mid
            or (close_val <= bb_lower_val * 1.03)  # or near lower band
        )
        pullback_momentum_ok = (rsi_val < 65) and (stoch_k_val < 80)
        pullback_adx_bias = plus_di_val >= minus_di_val
        pullback_candle_ok = is_bull_engulf or is_strong_bull

        if pullback_near_band and pullback_momentum_ok and pullback_adx_bias and pullback_candle_ok:
            reason = (
                "PULLBACK LONG: M5 pullback in uptrend (BB mid/lower, wider tolerance), "
                "RSI < 65, StochK < 80, bullish engulfing or strong bullish candle."
            )
            extra["setup_type"] = "PULLBACK_LONG"
            return Signal(
                direction="LONG",
                price=close_val,
                time=time_val,
                reason=reason,
                extra=extra,
            )

        # --- Moderate breakout conditions ---
        breakout_bb = close_val >= bb_upper_val  # close outside upper band
        breakout_momentum = (rsi_val >= 65) and (stoch_k_val >= 60)
        breakout_adx_bias = (plus_di_val > minus_di_val) and (adx_val >= 14)
        breakout_candle_ok = (
            (close_val > open_val)
            and (close_val >= bb_upper_val)
            and (close_val > prev_close)
        )

        if breakout_bb and breakout_momentum and breakout_adx_bias and breakout_candle_ok:
            reason = (
                "BREAKOUT LONG: M5 momentum candle through upper Bollinger band "
                "in line with uptrend (RSI >= 65, StochK >= 60, ADX >= 14, +DI > -DI)."
            )
            extra["setup_type"] = "BREAKOUT_LONG"
            return Signal(
                direction="LONG",
                price=close_val,
                time=time_val,
                reason=reason,
                extra=extra,
            )

    # -----------------------------------------------------------------------
    # SHORT SIDE: PULLBACK + BREAKOUT
    # -----------------------------------------------------------------------
    if trend_for_signal == "SHORT":
        # --- Pullback conditions ---
        pullback_near_band = (
            (close_val >= bb_mid_val * 0.99)  # slightly above / around mid
            or (close_val >= bb_upper_val * 0.97)  # near upper band
        )
        pullback_momentum_ok = (rsi_val > 35) and (stoch_k_val > 20)
        pullback_adx_bias = minus_di_val >= plus_di_val
        pullback_candle_ok = is_bear_engulf or is_strong_bear

        if pullback_near_band and pullback_momentum_ok and pullback_adx_bias and pullback_candle_ok:
            reason = (
                "PULLBACK SHORT: M5 pullback in downtrend (BB mid/upper, wider tolerance), "
                "RSI > 35, StochK > 20, bearish engulfing or strong bearish candle."
            )
            extra["setup_type"] = "PULLBACK_SHORT"
            return Signal(
                direction="SHORT",
                price=close_val,
                time=time_val,
                reason=reason,
                extra=extra,
            )

        # --- Moderate breakout conditions ---
        breakout_bb = close_val <= bb_lower_val  # close outside lower band
        breakout_momentum = (rsi_val <= 40) and (stoch_k_val <= 40)
        breakout_adx_bias = (minus_di_val > plus_di_val) and (adx_val >= 14)
        breakout_candle_ok = (
            (close_val < open_val)
            and (close_val <= bb_lower_val)
            and (close_val < prev_close)
        )

        if breakout_bb and breakout_momentum and breakout_adx_bias and breakout_candle_ok:
            reason = (
                "BREAKOUT SHORT: M5 momentum candle through lower Bollinger band "
                "in line with downtrend (RSI <= 40, StochK <= 40, ADX >= 14, -DI > +DI)."
            )
            extra["setup_type"] = "BREAKOUT_SHORT"
            return Signal(
                direction="SHORT",
                price=close_val,
                time=time_val,
                reason=reason,
                extra=extra,
            )

    # Nothing matched
    return None
