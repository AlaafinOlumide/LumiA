# strategy.py

import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from indicators import (
    bollinger_bands,
    rsi,
    stochastic_oscillator,
    adx,
    bullish_engulfing,
    bearish_engulfing,
    ema,
)


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

    if a < 15:
        return None

    if c > e20 > e50 and r > 55:
        return "LONG"
    if c < e20 < e50 and r < 45:
        return "SHORT"

    return None


def detect_trend_m15_direction(m15_df: pd.DataFrame) -> Optional[str]:
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
# M15 Structure (NEW)
# ---------------------------------------------------------------------------

def _swing_low(series: pd.Series, i: int) -> bool:
    # strict swing low: low[i] < low[i-1] and low[i] < low[i+1]
    if i <= 0 or i >= len(series) - 1:
        return False
    return series.iloc[i] < series.iloc[i - 1] and series.iloc[i] < series.iloc[i + 1]


def _swing_high(series: pd.Series, i: int) -> bool:
    if i <= 0 or i >= len(series) - 1:
        return False
    return series.iloc[i] > series.iloc[i - 1] and series.iloc[i] > series.iloc[i + 1]


def m15_structure_ok(m15_df: pd.DataFrame, trend_dir: str, lookback: int = 12) -> Tuple[bool, Dict[str, Any]]:
    """
    Structure filter:
      LONG: recent HL confirmed + current price above EMA20
      SHORT: recent LH confirmed + current price below EMA20

    Returns (ok, debug_extra)
    """
    extra: Dict[str, Any] = {"m15_structure_ok": False}

    if m15_df is None or len(m15_df) < max(lookback, 20):
        extra["m15_structure_reason"] = "not_enough_m15_bars"
        return False, extra

    tmp = m15_df.tail(lookback).copy()
    low = tmp["low"].reset_index(drop=True)
    high = tmp["high"].reset_index(drop=True)
    close = tmp["close"].reset_index(drop=True)

    ema20 = _ema(m15_df["close"], 20)
    e20 = float(ema20.iloc[-1])
    c = float(m15_df["close"].iloc[-1])

    # find last 2 swing lows / highs in the window
    swing_lows = [i for i in range(1, len(low) - 1) if _swing_low(low, i)]
    swing_highs = [i for i in range(1, len(high) - 1) if _swing_high(high, i)]

    if trend_dir == "LONG":
        if len(swing_lows) < 2:
            extra["m15_structure_reason"] = "no_two_swing_lows"
            return False, extra

        i1, i2 = swing_lows[-2], swing_lows[-1]
        l1, l2 = float(low.iloc[i1]), float(low.iloc[i2])

        # higher low + price above EMA20
        ok = (l2 > l1) and (c > e20)
        extra.update({
            "m15_last_swing_low_1": l1,
            "m15_last_swing_low_2": l2,
            "m15_ema20": e20,
            "m15_close": c,
            "m15_structure_reason": "HL_and_above_ema20" if ok else "failed_HL_or_below_ema20",
            "m15_structure_ok": ok,
        })
        return ok, extra

    if trend_dir == "SHORT":
        if len(swing_highs) < 2:
            extra["m15_structure_reason"] = "no_two_swing_highs"
            return False, extra

        i1, i2 = swing_highs[-2], swing_highs[-1]
        h1, h2 = float(high.iloc[i1]), float(high.iloc[i2])

        ok = (h2 < h1) and (c < e20)
        extra.update({
            "m15_last_swing_high_1": h1,
            "m15_last_swing_high_2": h2,
            "m15_ema20": e20,
            "m15_close": c,
            "m15_structure_reason": "LH_and_below_ema20" if ok else "failed_LH_or_above_ema20",
            "m15_structure_ok": ok,
        })
        return ok, extra

    extra["m15_structure_reason"] = "invalid_trend_dir"
    return False, extra


# ---------------------------------------------------------------------------
# Score-based Entry Gate (NEW)
# ---------------------------------------------------------------------------

def score_gate(score: int, setup_type: str) -> bool:
    """
    Per-setup thresholds.
    Pullbacks need stricter score than continuation.
    """
    if setup_type.startswith("PULLBACK"):
        return score >= 75
    if setup_type.startswith("BREAKOUT"):
        return score >= 65
    if setup_type.startswith("BREAKOUT_CONT"):
        return score >= 60
    return score >= 70


# ---------------------------------------------------------------------------
# M5 Trigger logic (UPGRADED PULLBACK + M15 STRUCTURE + SCORE GATE)
# ---------------------------------------------------------------------------

def trigger_signal_m5(m5_df: pd.DataFrame, trend_dir: str, m15_df: Optional[pd.DataFrame] = None) -> Optional[Signal]:
    """
    Priority:
      1) PULLBACK (now strict + structure + score gate)
      2) BREAKOUT
      3) BREAKOUT_CONT

    NOTE: pass m15_df from main.py for the structure filter.
    """
    if m5_df is None or len(m5_df) < 60:
        return None

    high = m5_df["high"]
    low = m5_df["low"]
    close = m5_df["close"]
    open_ = m5_df["open"]

    # Indicators
    bb_upper, bb_mid, bb_lower = bollinger_bands(close, period=20, std_factor=2.0)
    rsi_series = rsi(close, period=14)
    stoch_k, stoch_d = stochastic_oscillator(high, low, close, k_period=14, d_period=3)
    adx_series, plus_di, minus_di = adx(high, low, close, period=14)

    # trend context on M5
    ema20_m5 = _ema(close, 20)
    ema50_m5 = _ema(close, 50)

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

    e20 = float(ema20_m5.iloc[-1])
    e50 = float(ema50_m5.iloc[-1])

    bar_time = m5_df["datetime"].iloc[-1]
    if isinstance(bar_time, pd.Timestamp):
        bar_time = bar_time.to_pydatetime()

    # M15 structure check
    m15_ok, m15_extra = (True, {}) if m15_df is None else m15_structure_ok(m15_df, trend_dir)

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
        "ema20_m5": e20,
        "ema50_m5": e50,
        **m15_extra,
    }

    # ------------------------------------------------------------------
    # 1) PULLBACK (PROPER) — strict trend pullback, not late chasing
    # ------------------------------------------------------------------
    def _pullback_long() -> Optional[Signal]:
        # Trend must exist on M5 (micro alignment)
        trend_ok = e20 > e50 and c > e50

        # Pullback zone: between EMA20 and BB mid/lower (controlled retrace)
        in_pullback_zone = (c <= e20 * 1.002) and (c >= bb_l * 0.997)

        # "Reclaim" trigger: close back above EMA20 OR bullish engulfing
        reclaim = (c > e20 and c_prev <= e20) or bool(bull_engulf.iloc[-1])

        # Momentum reset: stochastic crosses up from below 50
        stoch_cross_up = (k > d) and (k_prev <= d_prev) and (k < 55 or k_prev < 50)

        # RSI filter to avoid buying exhaustion
        rsi_ok = 42 <= r <= 62

        # Structure filter on M15 (hard gate)
        structure_ok = bool(m15_ok)

        # SCORE
        score = 0
        score += 20 if trend_ok else 0
        score += 20 if in_pullback_zone else 0
        score += 15 if reclaim else 0
        score += 15 if stoch_cross_up else 0
        score += 10 if rsi_ok else 0
        score += 20 if structure_ok else 0  # structure is important

        extra = dict(extra_common)
        extra["setup_type"] = "PULLBACK_LONG"
        extra["setup_score"] = score
        extra["setup_class"] = "Pullback"

        if not score_gate(score, "PULLBACK_LONG"):
            return None

        reason = "Pullback LONG: trend ok + pullback zone + reclaim + stoch reset + M15 HL structure."
        return Signal("LONG", c, bar_time, reason, extra)

    def _pullback_short() -> Optional[Signal]:
        trend_ok = e20 < e50 and c < e50

        in_pullback_zone = (c >= e20 * 0.998) and (c <= bb_u * 1.003)

        reclaim = (c < e20 and c_prev >= e20) or bool(bear_engulf.iloc[-1])

        stoch_cross_down = (k < d) and (k_prev >= d_prev) and (k > 45 or k_prev > 50)

        rsi_ok = 38 <= r <= 58

        structure_ok = bool(m15_ok)

        score = 0
        score += 20 if trend_ok else 0
        score += 20 if in_pullback_zone else 0
        score += 15 if reclaim else 0
        score += 15 if stoch_cross_down else 0
        score += 10 if rsi_ok else 0
        score += 20 if structure_ok else 0

        extra = dict(extra_common)
        extra["setup_type"] = "PULLBACK_SHORT"
        extra["setup_score"] = score
        extra["setup_class"] = "Pullback"

        if not score_gate(score, "PULLBACK_SHORT"):
            return None

        reason = "Pullback SHORT: trend ok + pullback zone + reclaim + stoch reset + M15 LH structure."
        return Signal("SHORT", c, bar_time, reason, extra)

    # ------------------------------------------------------------------
    # 2) FRESH BREAKOUT (kept, but scored)
    # ------------------------------------------------------------------
    def _breakout_long() -> Optional[Signal]:
        prev_inside = c_prev <= bb_u_prev * 0.999
        now_break = c > bb_u * 1.001
        rsi_ok = r >= 52
        stoch_ok = k >= 55

        score = 0
        score += 25 if prev_inside else 0
        score += 25 if now_break else 0
        score += 10 if rsi_ok else 0
        score += 10 if stoch_ok else 0
        score += 10 if adx_m5 >= 18 else 0

        extra = dict(extra_common)
        extra["setup_type"] = "BREAKOUT_LONG"
        extra["setup_score"] = score
        extra["setup_class"] = "Breakout"

        if not score_gate(score, "BREAKOUT_LONG"):
            return None

        reason = "Breakout LONG: close breaks above upper BB with momentum."
        return Signal("LONG", c, bar_time, reason, extra)

    def _breakout_short() -> Optional[Signal]:
        prev_inside = c_prev >= bb_l_prev * 1.001
        now_break = c < bb_l * 0.999
        rsi_ok = r <= 48
        stoch_ok = k <= 45

        score = 0
        score += 25 if prev_inside else 0
        score += 25 if now_break else 0
        score += 10 if rsi_ok else 0
        score += 10 if stoch_ok else 0
        score += 10 if adx_m5 >= 18 else 0

        extra = dict(extra_common)
        extra["setup_type"] = "BREAKOUT_SHORT"
        extra["setup_score"] = score
        extra["setup_class"] = "Breakout"

        if not score_gate(score, "BREAKOUT_SHORT"):
            return None

        reason = "Breakout SHORT: close breaks below lower BB with momentum."
        return Signal("SHORT", c, bar_time, reason, extra)

    # ------------------------------------------------------------------
    # 3) BREAKOUT CONTINUATION (kept, scored)
    # ------------------------------------------------------------------
    def _breakout_cont_long() -> Optional[Signal]:
        riding_upper = c >= bb_u * 0.999
        di_ok = plus_di_m5 > minus_di_m5
        adx_ok = adx_m5 >= 20

        score = 0
        score += 30 if riding_upper else 0
        score += 15 if adx_ok else 0
        score += 15 if di_ok else 0
        score += 10 if r >= 50 else 0

        extra = dict(extra_common)
        extra["setup_type"] = "BREAKOUT_CONT_LONG"
        extra["setup_score"] = score
        extra["setup_class"] = "Breakout Continuation"

        if not score_gate(score, "BREAKOUT_CONT_LONG"):
            return None

        reason = "Breakout continuation LONG: riding upper BB with ADX support."
        return Signal("LONG", c, bar_time, reason, extra)

    def _breakout_cont_short() -> Optional[Signal]:
        riding_lower = c <= bb_l * 1.001
        di_ok = minus_di_m5 > plus_di_m5
        adx_ok = adx_m5 >= 20

        score = 0
        score += 30 if riding_lower else 0
        score += 15 if adx_ok else 0
        score += 15 if di_ok else 0
        score += 10 if r <= 50 else 0

        extra = dict(extra_common)
        extra["setup_type"] = "BREAKOUT_CONT_SHORT"
        extra["setup_score"] = score
        extra["setup_class"] = "Breakout Continuation"

        if not score_gate(score, "BREAKOUT_CONT_SHORT"):
            return None

        reason = "Breakout continuation SHORT: riding lower BB with ADX support."
        return Signal("SHORT", c, bar_time, reason, extra)

    # ------------------------------------------------------------------
    # Priority order (you can flip this later if you want)
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
