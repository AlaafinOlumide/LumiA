import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from indicators import bollinger_bands, rsi, stochastic_oscillator, adx, bullish_engulfing, bearish_engulfing


@dataclass
class Signal:
    direction: str          # "LONG" or "SHORT"
    price: float
    time: dt.datetime
    reason: str
    extra: Dict[str, Any]


# ---------------------------------------------------------------------------
# Session helper
# ---------------------------------------------------------------------------

def is_within_sessions(
    now_utc: dt.datetime,
    session_1_start: int,
    session_1_end: int,
    session_2_start: Optional[int],
    session_2_end: Optional[int],
) -> bool:
    """
    now_utc:      datetime in UTC
    session_*:    HHMM, e.g. 700 for 07:00, 2000 for 20:00
    """
    hhmm = now_utc.hour * 100 + now_utc.minute

    in_s1 = session_1_start <= hhmm <= session_1_end

    if session_2_start is not None and session_2_end is not None:
        in_s2 = session_2_start <= hhmm <= session_2_end
    else:
        in_s2 = False

    return in_s1 or in_s2


# ---------------------------------------------------------------------------
# Trend detection on H1 / M15
# ---------------------------------------------------------------------------

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def detect_trend_h1(h1_df: pd.DataFrame) -> Optional[str]:
    """
    Detect higher-timeframe trend on H1 using EMA(20/50), RSI, and ADX.
    Returns "LONG", "SHORT", or None (ranging / unclear).
    """
    if h1_df is None or len(h1_df) < 60:
        return None

    close = h1_df["close"]
    high = h1_df["high"]
    low = h1_df["low"]

    ema20 = _ema(close, 20)
    ema50 = _ema(close, 50)
    rsi_series = rsi(close, period=14)
    adx_series, _, _ = adx(high, low, close, period=14)

    c = float(close.iloc[-1])
    e20 = float(ema20.iloc[-1])
    e50 = float(ema50.iloc[-1])
    r = float(rsi_series.iloc[-1])
    a = float(adx_series.iloc[-1]) if pd.notna(adx_series.iloc[-1]) else 0.0

    # basic filters (if ADX weak -> ranging)
    if a < 15:
        return None

    # Uptrend
    if c > e20 > e50 and r > 55:
        return "LONG"

    # Downtrend
    if c < e20 < e50 and r < 45:
        return "SHORT"

    return None


def detect_trend_m15_direction(m15_df: pd.DataFrame) -> Optional[str]:
    """
    Fallback trend detection on M15 when H1 is ranging.
    Slightly looser thresholds than H1.
    """
    if m15_df is None or len(m15_df) < 60:
        return None

    close = m15_df["close"]
    high = m15_df["high"]
    low = m15_df["low"]

    ema20 = _ema(close, 20)
    ema50 = _ema(close, 50)
    rsi_series = rsi(close, period=14)
    adx_series, _, _ = adx(high, low, close, period=14)

    c = float(close.iloc[-1])
    e20 = float(ema20.iloc[-1])
    e50 = float(ema50.iloc[-1])
    r = float(rsi_series.iloc[-1])
    a = float(adx_series.iloc[-1]) if pd.notna(adx_series.iloc[-1]) else 0.0

    if a < 12:
        return None

    if c > e20 > e50 and r > 52:
        return "LONG"
    if c < e20 < e50 and r < 48:
        return "SHORT"

    return None


def confirm_trend_m15(m15_df: pd.DataFrame, trend: str) -> bool:
    """
    Confirm H1 trend using M15 structure.
    - For LONG: EMA20 > EMA50 and price above EMA20
    - For SHORT: EMA20 < EMA50 and price below EMA20
    """
    if m15_df is None or len(m15_df) < 40:
        return False

    close = m15_df["close"]
    ema20 = _ema(close, 20)
    ema50 = _ema(close, 50)

    c = float(close.iloc[-1])
    e20 = float(ema20.iloc[-1])
    e50 = float(ema50.iloc[-1])

    if trend == "LONG":
        return e20 > e50 and c > e20
    if trend == "SHORT":
        return e20 < e50 and c < e20

    return False


# ---------------------------------------------------------------------------
# M5 trigger logic: Pullback + Breakout + Breakout Continuation
# ---------------------------------------------------------------------------

def trigger_signal_m5(m5_df: pd.DataFrame, trend_dir: str) -> Optional[Signal]:
    """
    Main trigger engine on M5.
    Priority:
        1) PULLBACK
        2) BREAKOUT
        3) BREAKOUT_CONT
    """
    if m5_df is None or len(m5_df) < 60:
        return None

    high = m5_df["high"]
    low = m5_df["low"]
    close = m5_df["close"]
    open_ = m5_df["open"]

    # Indicators (MATCH indicators.py signatures!)
    bb_upper, bb_mid, bb_lower = bollinger_bands(close, period=20, std_factor=2.0)
    rsi_series = rsi(close, period=14)
    stoch_k, stoch_d = stochastic_oscillator(high, low, close, k_period=14, d_period=3)
    adx_series, plus_di, minus_di = adx(high, low, close, period=14)

    bull_engulf = bullish_engulfing(open_, close)
    bear_engulf = bearish_engulfing(open_, close)

    # last & previous values
    c = float(close.iloc[-1])
    o = float(open_.iloc[-1])
    c_prev = float(close.iloc[-2])

    bb_u = float(bb_upper.iloc[-1])
    bb_m = float(bb_mid.iloc[-1])
    bb_l = float(bb_lower.iloc[-1])

    bb_u_prev = float(bb_upper.iloc[-2])
    bb_l_prev = float(bb_lower.iloc[-2])

    r = float(rsi_series.iloc[-1])
    k = float(stoch_k.iloc[-1])
    d = float(stoch_d.iloc[-1])
    k_prev = float(stoch_k.iloc[-2])
    d_prev = float(stoch_d.iloc[-2])

    adx_m5 = float(adx_series.iloc[-1]) if pd.notna(adx_series.iloc[-1]) else 0.0
    plus_di_m5 = float(plus_di.iloc[-1]) if pd.notna(plus_di.iloc[-1]) else 0.0
    minus_di_m5 = float(minus_di.iloc[-1]) if pd.notna(minus_di.iloc[-1]) else 0.0

    bar_time = m5_df["datetime"].iloc[-1]
    if isinstance(bar_time, pd.Timestamp):
        bar_time = bar_time.to_pydatetime()

    extra_common = {
        "m5_rsi": r,
        "m5_stoch_k": k,
        "m5_stoch_d": d,
        "bb_upper": bb_u,
        "bb_mid": bb_m,
        "bb_lower": bb_l,
        "adx_m5": adx_m5,
        "plus_di_m5": plus_di_m5,
        "minus_di_m5": minus_di_m5,
    }

    # ------------------------------------------------------------------
    # 1) PULLBACK setups
    # ------------------------------------------------------------------
    def _pullback_long() -> Optional[Signal]:
        # pullback to mid/lower + stoch cross up + candle confirmation
        near_mid_or_lower = (c <= bb_m * 1.004) and (c >= bb_l * 0.996)
        stoch_cross_up = (k > d) and (k_prev <= d_prev)
        candle_ok = (c > o) or bool(bull_engulf.iloc[-1])
        rsi_ok = 38 <= r <= 60

        if near_mid_or_lower and stoch_cross_up and candle_ok and rsi_ok:
            extra = dict(extra_common)
            extra["setup_type"] = "PULLBACK_LONG"
            reason = "Pullback LONG: BB mid/lower + stoch cross up + candle confirm."
            return Signal("LONG", c, bar_time, reason, extra)
        return None

    def _pullback_short() -> Optional[Signal]:
        near_mid_or_upper = (c >= bb_m * 0.996) and (c <= bb_u * 1.004)
        stoch_cross_down = (k < d) and (k_prev >= d_prev)
        candle_ok = (c < o) or bool(bear_engulf.iloc[-1])
        rsi_ok = 40 <= r <= 62

        if near_mid_or_upper and stoch_cross_down and candle_ok and rsi_ok:
            extra = dict(extra_common)
            extra["setup_type"] = "PULLBACK_SHORT"
            reason = "Pullback SHORT: BB mid/upper + stoch cross down + candle confirm."
            return Signal("SHORT", c, bar_time, reason, extra)
        return None

    # ------------------------------------------------------------------
    # 2) FRESH BREAKOUT setups
    # ------------------------------------------------------------------
    def _breakout_long() -> Optional[Signal]:
        prev_inside = c_prev <= bb_u_prev * 0.999
        now_break = c > bb_u * 1.001
        rsi_ok = r >= 52
        stoch_ok = k >= 55

        if prev_inside and now_break and rsi_ok and stoch_ok:
            extra = dict(extra_common)
            extra["setup_type"] = "BREAKOUT_LONG"
            reason = "Breakout LONG: close breaks above upper BB with momentum."
            return Signal("LONG", c, bar_time, reason, extra)
        return None

    def _breakout_short() -> Optional[Signal]:
        prev_inside = c_prev >= bb_l_prev * 1.001
        now_break = c < bb_l * 0.999
        rsi_ok = r <= 48
        stoch_ok = k <= 45

        if prev_inside and now_break and rsi_ok and stoch_ok:
            extra = dict(extra_common)
            extra["setup_type"] = "BREAKOUT_SHORT"
            reason = "Breakout SHORT: close breaks below lower BB with momentum."
            return Signal("SHORT", c, bar_time, reason, extra)
        return None

    # ------------------------------------------------------------------
    # 3) BREAKOUT CONTINUATION setups
    # ------------------------------------------------------------------
    def _breakout_cont_long() -> Optional[Signal]:
        riding_upper = c >= bb_u * 0.999
        adx_ok = adx_m5 >= 20 and plus_di_m5 > minus_di_m5
        if riding_upper and adx_ok:
            extra = dict(extra_common)
            extra["setup_type"] = "BREAKOUT_CONT_LONG"
            reason = "Breakout continuation LONG: riding upper BB with ADX support."
            return Signal("LONG", c, bar_time, reason, extra)
        return None

    def _breakout_cont_short() -> Optional[Signal]:
        riding_lower = c <= bb_l * 1.001
        adx_ok = adx_m5 >= 20 and minus_di_m5 > plus_di_m5
        if riding_lower and adx_ok:
            extra = dict(extra_common)
            extra["setup_type"] = "BREAKOUT_CONT_SHORT"
            reason = "Breakout continuation SHORT: riding lower BB with ADX support."
            return Signal("SHORT", c, bar_time, reason, extra)
        return None

    # ------------------------------------------------------------------
    # Priority order
    # ------------------------------------------------------------------
    if trend_dir == "LONG":
        sig = _pullback_long()
        if sig:
            return sig
        sig = _breakout_long()
        if sig:
            return sig
        return _breakout_cont_long()

    if trend_dir == "SHORT":
        sig = _pullback_short()
        if sig:
            return sig
        sig = _breakout_short()
        if sig:
            return sig
        return _breakout_cont_short()

    return None
