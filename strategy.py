import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from indicators import bollinger_bands, rsi, stochastic, adx


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
    a = float(adx_series.iloc[-1])

    # basic filters
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
    a = float(adx_series.iloc[-1])

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
    We merge 3 setup types, with priority:

        1) PULLBACK      (best quality)
        2) BREAKOUT      (fresh BB breakout)
        3) BREAKOUT_CONT (momentum continuation along the band)

    All of them respect the higher-TF trend_dir ("LONG" / "SHORT").
    Returns a Signal or None.
    """
    if m5_df is None or len(m5_df) < 50:
        return None

    high = m5_df["high"]
    low = m5_df["low"]
    close = m5_df["close"]
    open_ = m5_df["open"]

    # Indicators
    bb_upper, bb_mid, bb_lower = bollinger_bands(close, period=20, std_dev=2)
    rsi_series = rsi(close, period=14)
    stoch_k, stoch_d = stochastic(high, low, close, k_period=5, d_period=3)
    adx_series, plus_di, minus_di = adx(high, low, close, period=14)

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
    r_prev = float(rsi_series.iloc[-2])

    k = float(stoch_k.iloc[-1])
    d = float(stoch_d.iloc[-1])
    k_prev = float(stoch_k.iloc[-2])
    d_prev = float(stoch_d.iloc[-2])

    adx_m5 = float(adx_series.iloc[-1])
    plus_di_m5 = float(plus_di.iloc[-1])
    minus_di_m5 = float(minus_di.iloc[-1])

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
    # 1) PULLBACK setups (highest quality, first priority)
    # ------------------------------------------------------------------
    def _pullback_long() -> Optional[Signal]:
        # Price pulls back toward mid/lower band in an uptrend,
        # momentum recovers (stoch cross up), RSI not too weak.
        near_mid_or_lower = (c <= bb_m * 1.003) and (c >= bb_l * 0.997)
        stoch_cross_up = (k > d) and (k_prev <= d_prev)
        bullish_candle = c > o
        rsi_ok = 40 <= r <= 60

        if near_mid_or_lower and stoch_cross_up and bullish_candle and rsi_ok:
            extra = dict(extra_common)
            extra["setup_type"] = "PULLBACK_LONG"
            reason = (
                "Pullback LONG: price near BB mid/lower in uptrend, "
                "stoch turning up, bullish candle."
            )
            return Signal(
                direction="LONG",
                price=c,
                time=bar_time,
                reason=reason,
                extra=extra,
            )
        return None

    def _pullback_short() -> Optional[Signal]:
        near_mid_or_upper = (c >= bb_m * 0.997) and (c <= bb_u * 1.003)
        stoch_cross_down = (k < d) and (k_prev >= d_prev)
        bearish_candle = c < o
        rsi_ok = 40 <= r <= 60

        if near_mid_or_upper and stoch_cross_down and bearish_candle and rsi_ok:
            extra = dict(extra_common)
            extra["setup_type"] = "PULLBACK_SHORT"
            reason = (
                "Pullback SHORT: price near BB mid/upper in downtrend, "
                "stoch turning down, bearish candle."
            )
            return Signal(
                direction="SHORT",
                price=c,
                time=bar_time,
                reason=reason,
                extra=extra,
            )
        return None

    # ------------------------------------------------------------------
    # 2) FRESH BREAKOUT setups (second priority)
    # ------------------------------------------------------------------
    def _breakout_long() -> Optional[Signal]:
        prev_inside_band = c_prev <= bb_u_prev * 0.999
        now_above_band = c > bb_u * 1.001
        rsi_ok = 55 <= r <= 75
        momentum_ok = k > 50

        if prev_inside_band and now_above_band and rsi_ok and momentum_ok:
            extra = dict(extra_common)
            extra["setup_type"] = "BREAKOUT_LONG"
            reason = (
                "Fresh breakout LONG: close breaks above upper BB from inside, "
                "RSI and Stoch confirm momentum."
            )
            return Signal(
                direction="LONG",
                price=c,
                time=bar_time,
                reason=reason,
                extra=extra,
            )
        return None

    def _breakout_short() -> Optional[Signal]:
        prev_inside_band = c_prev >= bb_l_prev * 1.001
        now_below_band = c < bb_l * 0.999
        rsi_ok = 25 <= r <= 45
        momentum_ok = k < 50

        if prev_inside_band and now_below_band and rsi_ok and momentum_ok:
            extra = dict(extra_common)
            extra["setup_type"] = "BREAKOUT_SHORT"
            reason = (
                "Fresh breakout SHORT: close breaks below lower BB from inside, "
                "RSI and Stoch confirm momentum."
            )
            return Signal(
                direction="SHORT",
                price=c,
                time=bar_time,
                reason=reason,
                extra=extra,
            )
        return None

    # ------------------------------------------------------------------
    # 3) BREAKOUT CONTINUATION setups (third priority, more aggressive)
    # ------------------------------------------------------------------
    def _breakout_cont_long() -> Optional[Signal]:
        # Price riding upper band with strong ADX – aggressive continuation.
        riding_upper = c > bb_u * 0.999
        adx_strong = adx_m5 >= 25 and plus_di_m5 > minus_di_m5

        if riding_upper and adx_strong:
            extra = dict(extra_common)
            extra["setup_type"] = "BREAKOUT_CONT_LONG"
            reason = (
                "Momentum breakout continuation LONG: price riding upper BB "
                "with strong ADX."
            )
            return Signal(
                direction="LONG",
                price=c,
                time=bar_time,
                reason=reason,
                extra=extra,
            )
        return None

    def _breakout_cont_short() -> Optional[Signal]:
        riding_lower = c < bb_l * 1.001
        adx_strong = adx_m5 >= 25 and minus_di_m5 > plus_di_m5

        if riding_lower and adx_strong:
            extra = dict(extra_common)
            extra["setup_type"] = "BREAKOUT_CONT_SHORT"
            reason = (
                "Momentum breakout continuation SHORT: price riding lower BB "
                "with strong ADX."
            )
            return Signal(
                direction="SHORT",
                price=c,
                time=bar_time,
                reason=reason,
                extra=extra,
            )
        return None

    # ------------------------------------------------------------------
    # Priority by setup quality:
    #   Pullback > Fresh Breakout > Breakout Continuation
    # ------------------------------------------------------------------
    if trend_dir == "LONG":
        sig = _pullback_long()
        if sig:
            return sig

        sig = _breakout_long()
        if sig:
            return sig

        sig = _breakout_cont_long()
        return sig

    elif trend_dir == "SHORT":
        sig = _pullback_short()
        if sig:
            return sig

        sig = _breakout_short()
        if sig:
            return sig

        sig = _breakout_cont_short()
        return sig

    return None
