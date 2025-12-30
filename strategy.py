# strategy.py
from __future__ import annotations

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
    atr,  # assumes you have atr() in indicators.py (you referenced ATR in messages)
)


@dataclass
class Signal:
    direction: str          # "LONG" or "SHORT"
    price: float
    time: dt.datetime
    reason: str
    extra: Dict[str, Any]


# ---------------------------------------------------------------------------
# Session helper (PATCHED: supports weekends + strict signature)
# ---------------------------------------------------------------------------

def is_within_sessions(
    now_utc: dt.datetime,
    session_1_start: int,
    session_1_end: int,
    session_2_start: Optional[int] = None,
    session_2_end: Optional[int] = None,
    trade_weekends: bool = False,
) -> bool:
    """
    now_utc: datetime in UTC
    session_*: HHMM, e.g. 700 for 07:00, 2000 for 20:00

    PATCH:
    - Weekend filtering (unless trade_weekends=True)
    - Explicit signature prevents silent bypass from bad kwargs
    """
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=dt.timezone.utc)

    # weekend filter
    if not trade_weekends:
        # Saturday=5, Sunday=6
        if now_utc.weekday() >= 5:
            return False

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

    if c > e20 > e50 and r > 55:
        return "LONG"
    if c < e20 < e50 and r < 45:
        return "SHORT"

    return None


def detect_trend_m15_direction(m15_df: pd.DataFrame) -> Optional[str]:
    """
    Fallback trend detection on M15 when H1 is ranging.
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
# Market regime + adaptive TP/SL (keep your existing API)
# ---------------------------------------------------------------------------

def market_regime(h1_df: pd.DataFrame) -> str:
    """
    Simple regime label:
    - High Volatility: ATR above its 50-period median
    - Low Volatility / Compression: ATR below its 50-period median
    """
    if h1_df is None or len(h1_df) < 70:
        return "Normal Volatility"

    atr14 = atr(h1_df["high"], h1_df["low"], h1_df["close"], period=14)
    if len(atr14) < 60:
        return "Normal Volatility"
    cur = float(atr14.iloc[-1])
    med = float(atr14.iloc[-60:].median())

    if cur > 1.15 * med:
        return "High Volatility"
    if cur < 0.85 * med:
        return "Low Volatility / Compression"
    return "Normal Volatility"


def dynamic_tp_sl(signal: Signal, h1_df: pd.DataFrame, regime: str) -> None:
    """
    Write TP/SL into signal.extra:
      - SL via ATR(H1)
      - TP1 / TP2 scaled by regime
    """
    if h1_df is None or len(h1_df) < 30:
        return

    atr14 = atr(h1_df["high"], h1_df["low"], h1_df["close"], period=14)
    atr_v = float(atr14.iloc[-1]) if pd.notna(atr14.iloc[-1]) else 8.0

    entry = float(signal.price)

    # base multipliers (tuned for scalp-ish behaviour)
    sl_mult = 0.9
    tp1_mult = 1.4
    tp2_mult = 2.2

    if regime == "High Volatility":
        sl_mult = 1.1
        tp1_mult = 1.6
        tp2_mult = 2.6
    elif regime == "Low Volatility / Compression":
        sl_mult = 0.8
        tp1_mult = 1.2
        tp2_mult = 1.9

    sl_dist = atr_v * sl_mult
    tp1_dist = atr_v * tp1_mult
    tp2_dist = atr_v * tp2_mult

    if signal.direction == "LONG":
        sl = entry - sl_dist
        tp1 = entry + tp1_dist
        tp2 = entry + tp2_dist
    else:
        sl = entry + sl_dist
        tp1 = entry - tp1_dist
        tp2 = entry - tp2_dist

    signal.extra["atr_h1_14"] = atr_v
    signal.extra["sl"] = float(sl)
    signal.extra["tp1"] = float(tp1)
    signal.extra["tp2"] = float(tp2)


# ---------------------------------------------------------------------------
# Confidence helpers (keep your existing API)
# ---------------------------------------------------------------------------

def compute_confidence(
    trend_source: str,
    setup_type: str,
    adx_h1: float,
    adx_m5: float,
    high_news: bool,
    entry_score: int,
) -> int:
    score = 50

    score += min(20, max(0, entry_score - 70))  # reward strong gate passes

    if trend_source == "H1":
        score += 5

    if adx_h1 >= 35:
        score += 10
    elif adx_h1 >= 25:
        score += 6
    elif adx_h1 >= 20:
        score += 3

    if adx_m5 >= 30:
        score += 6
    elif adx_m5 >= 20:
        score += 3

    if "IMPULSE" in setup_type:
        score += 4

    if high_news:
        score -= 12

    return int(max(0, min(99, score)))


def confidence_label(conf: int) -> str:
    if conf >= 80:
        return "High"
    if conf >= 65:
        return "Medium"
    return "Low"


# ---------------------------------------------------------------------------
# Score-based entry gate + patched trigger engine
# ---------------------------------------------------------------------------

def _entry_score(
    *,
    trend_dir: str,
    setup_type: str,
    rsi_m5: float,
    stoch_k: float,
    stoch_d: float,
    adx_m5: float,
    plus_di_m5: float,
    minus_di_m5: float,
    adx_h1: float,
    m15_structure_ok: bool,
    high_news: bool,
    range_ratio_ok: bool,
) -> Tuple[int, Dict[str, int]]:
    """
    Returns (score, breakdown).
    Score is 0..100, minimum threshold enforced by main.py (min_entry_score).
    """
    b: Dict[str, int] = {}

    # 1) Base
    score = 50
    b["base"] = 50

    # 2) Structure alignment
    if m15_structure_ok:
        score += 12
        b["m15_structure"] = 12
    else:
        score -= 12
        b["m15_structure"] = -12

    # 3) Trend strength (H1)
    if adx_h1 >= 35:
        score += 10
        b["adx_h1"] = 10
    elif adx_h1 >= 25:
        score += 6
        b["adx_h1"] = 6
    elif adx_h1 >= 20:
        score += 3
        b["adx_h1"] = 3
    else:
        score -= 6
        b["adx_h1"] = -6

    # 4) Momentum / participation (M5 ADX + DI)
    if adx_m5 >= 30:
        score += 8
        b["adx_m5"] = 8
    elif adx_m5 >= 20:
        score += 4
        b["adx_m5"] = 4
    else:
        score -= 6
        b["adx_m5"] = -6

    if trend_dir == "LONG" and plus_di_m5 > minus_di_m5:
        score += 4
        b["di_align"] = 4
    elif trend_dir == "SHORT" and minus_di_m5 > plus_di_m5:
        score += 4
        b["di_align"] = 4
    else:
        score -= 4
        b["di_align"] = -4

    # 5) RSI sanity (avoid buying overbought / selling oversold for pullbacks)
    if "PULLBACK" in setup_type:
        if trend_dir == "LONG" and 38 <= rsi_m5 <= 60:
            score += 6
            b["rsi_zone"] = 6
        elif trend_dir == "SHORT" and 40 <= rsi_m5 <= 62:
            score += 6
            b["rsi_zone"] = 6
        else:
            score -= 6
            b["rsi_zone"] = -6

    # 6) Stoch cross quality
    if (stoch_k > stoch_d and trend_dir == "LONG") or (stoch_k < stoch_d and trend_dir == "SHORT"):
        score += 3
        b["stoch_bias"] = 3
    else:
        score -= 3
        b["stoch_bias"] = -3

    # 7) Liquidity / range filter (critical)
    if range_ratio_ok:
        score += 6
        b["range_ok"] = 6
    else:
        score -= 10
        b["range_ok"] = -10

    # 8) News penalty
    if high_news:
        score -= 12
        b["news"] = -12

    return int(max(0, min(100, score))), b


def trigger_signal_m5(
    *,
    m5_df: pd.DataFrame,
    trend_dir: str,
    m15_df: pd.DataFrame,
    h1_df: pd.DataFrame,
    high_news: bool,
    min_score: int,
    range_filter_lookback: int = 20,
    range_filter_min_ratio: float = 0.60,
) -> Optional[Signal]:
    """
    PATCHES INCLUDED:
    - Liquidity/range filter
    - M15 structure gate baked in
    - Score-based entry gate (min_score)
    - Impulse continuation setup (captures big sell-offs)
    - Suppress pullbacks when ADX(H1) is very strong (>=35)
    """
    if m5_df is None or len(m5_df) < 120:
        return None
    if m15_df is None or len(m15_df) < 80:
        return None
    if h1_df is None or len(h1_df) < 80:
        return None

    # --- H1 ADX for regime gating ---
    adx_h1_series, _, _ = adx(h1_df["high"], h1_df["low"], h1_df["close"], period=14)
    adx_h1 = float(adx_h1_series.iloc[-1]) if pd.notna(adx_h1_series.iloc[-1]) else 0.0

    # --- M15 structure confirmation ---
    m15_structure_ok = confirm_trend_m15(m15_df, trend_dir)

    # --- Liquidity / candle-range filter on M5 ---
    rng = (m5_df["high"] - m5_df["low"])
    last_range = float(rng.iloc[-1])
    avg_range = float(rng.rolling(range_filter_lookback).mean().iloc[-1]) if len(rng) >= range_filter_lookback else last_range
    range_ratio_ok = (avg_range > 0) and (last_range >= range_filter_min_ratio * avg_range)

    # --- Indicators (M5) ---
    high = m5_df["high"]
    low = m5_df["low"]
    close = m5_df["close"]
    open_ = m5_df["open"]

    bb_upper, bb_mid, bb_lower = bollinger_bands(close, period=20, std_factor=2.0)
    rsi_series = rsi(close, period=14)
    stoch_k, stoch_d = stochastic_oscillator(high, low, close, k_period=14, d_period=3)
    adx_series, plus_di, minus_di = adx(high, low, close, period=14)

    bull_engulf = bullish_engulfing(open_, close)
    bear_engulf = bearish_engulfing(open_, close)

    # last/prev
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

    extra_common: Dict[str, Any] = {
        "m5_rsi": r,
        "m5_stoch_k": k,
        "m5_stoch_d": d,
        "bb_upper": bb_u,
        "bb_mid": bb_m,
        "bb_lower": bb_l,
        "adx_m5": adx_m5,
        "plus_di_m5": plus_di_m5,
        "minus_di_m5": minus_di_m5,
        "adx_h1": adx_h1,
        "m15_structure_ok": bool(m15_structure_ok),
        "range_last": last_range,
        "range_avg": avg_range,
        "range_ratio_ok": bool(range_ratio_ok),
    }

    # ------------------------------------------------------------------
    # Setup definitions
    # ------------------------------------------------------------------

    def _gate_or_none(setup_type: str, reason: str) -> Optional[Signal]:
        score, breakdown = _entry_score(
            trend_dir=trend_dir,
            setup_type=setup_type,
            rsi_m5=r,
            stoch_k=k,
            stoch_d=d,
            adx_m5=adx_m5,
            plus_di_m5=plus_di_m5,
            minus_di_m5=minus_di_m5,
            adx_h1=adx_h1,
            m15_structure_ok=m15_structure_ok,
            high_news=high_news,
            range_ratio_ok=range_ratio_ok,
        )
        if score < min_score:
            return None

        ex = dict(extra_common)
        ex["setup_type"] = setup_type
        ex["entry_score"] = int(score)
        ex["entry_score_breakdown"] = breakdown
        return Signal(trend_dir, c, bar_time, reason, ex)

    # 1) Pullback (PATCHED: stricter + structure required, and blocked in very strong ADX(H1))
    def _pullback_long() -> Optional[Signal]:
        if adx_h1 >= 35:
            return None  # PATCH: suppress pullbacks during very strong trend (avoid fading)
        near_mid_or_lower = (c <= bb_m * 1.003) and (c >= bb_l * 0.997)
        stoch_cross_up = (k > d) and (k_prev <= d_prev)
        candle_ok = (c > o) or bool(bull_engulf.iloc[-1])
        rsi_ok = 38 <= r <= 60

        # structure requirement
        if not m15_structure_ok:
            return None

        if near_mid_or_lower and stoch_cross_up and candle_ok and rsi_ok:
            return _gate_or_none(
                "PULLBACK_LONG",
                "Pullback LONG: BB mid/lower + stoch cross up + candle confirm + M15 structure.",
            )
        return None

    def _pullback_short() -> Optional[Signal]:
        if adx_h1 >= 35:
            return None
        near_mid_or_upper = (c >= bb_m * 0.997) and (c <= bb_u * 1.003)
        stoch_cross_down = (k < d) and (k_prev >= d_prev)
        candle_ok = (c < o) or bool(bear_engulf.iloc[-1])
        rsi_ok = 40 <= r <= 62

        if not m15_structure_ok:
            return None

        if near_mid_or_upper and stoch_cross_down and candle_ok and rsi_ok:
            return _gate_or_none(
                "PULLBACK_SHORT",
                "Pullback SHORT: BB mid/upper + stoch cross down + candle confirm + M15 structure.",
            )
        return None

    # 2) Fresh breakout
    def _breakout_long() -> Optional[Signal]:
        prev_inside = c_prev <= bb_u_prev * 0.999
        now_break = c > bb_u * 1.001
        rsi_ok = r >= 52
        stoch_ok = k >= 55
        if prev_inside and now_break and rsi_ok and stoch_ok and m15_structure_ok:
            return _gate_or_none(
                "BREAKOUT_LONG",
                "Breakout LONG: close breaks above upper BB with momentum + M15 structure.",
            )
        return None

    def _breakout_short() -> Optional[Signal]:
        prev_inside = c_prev >= bb_l_prev * 1.001
        now_break = c < bb_l * 0.999
        rsi_ok = r <= 48
        stoch_ok = k <= 45
        if prev_inside and now_break and rsi_ok and stoch_ok and m15_structure_ok:
            return _gate_or_none(
                "BREAKOUT_SHORT",
                "Breakout SHORT: close breaks below lower BB with momentum + M15 structure.",
            )
        return None

    # 3) BB ride continuation
    def _breakout_cont_long() -> Optional[Signal]:
        riding_upper = c >= bb_u * 0.999
        adx_ok = adx_m5 >= 20 and plus_di_m5 > minus_di_m5
        if riding_upper and adx_ok and m15_structure_ok:
            return _gate_or_none(
                "BREAKOUT_CONT_LONG",
                "Breakout continuation LONG: riding upper BB with ADX/DI + M15 structure.",
            )
        return None

    def _breakout_cont_short() -> Optional[Signal]:
        riding_lower = c <= bb_l * 1.001
        adx_ok = adx_m5 >= 20 and minus_di_m5 > plus_di_m5
        if riding_lower and adx_ok and m15_structure_ok:
            return _gate_or_none(
                "BREAKOUT_CONT_SHORT",
                "Breakout continuation SHORT: riding lower BB with ADX/DI + M15 structure.",
            )
        return None

    # 4) IMPULSE continuation (NEW: catches big sell-offs / buy squeezes)
    def _impulse_cont() -> Optional[Signal]:
        if adx_h1 < 35:
            return None

        # Use M15 BB as impulse detector
        m15_close = float(m15_df["close"].iloc[-1])
        m15_bb_u, m15_bb_m, m15_bb_l = bollinger_bands(m15_df["close"], period=20, std_factor=2.0)
        m15_u = float(m15_bb_u.iloc[-1])
        m15_l = float(m15_bb_l.iloc[-1])

        if trend_dir == "SHORT":
            impulse_m15 = m15_close < m15_l * 0.999
            ride_m5 = c <= bb_l * 1.001
            rsi_ok = r <= 50
            if impulse_m15 and ride_m5 and rsi_ok:
                return _gate_or_none(
                    "IMPULSE_CONT_SHORT",
                    "Impulse continuation SHORT: strong H1 trend + M15 impulse + M5 lower-band ride.",
                )
        else:
            impulse_m15 = m15_close > m15_u * 1.001
            ride_m5 = c >= bb_u * 0.999
            rsi_ok = r >= 50
            if impulse_m15 and ride_m5 and rsi_ok:
                return _gate_or_none(
                    "IMPULSE_CONT_LONG",
                    "Impulse continuation LONG: strong H1 trend + M15 impulse + M5 upper-band ride.",
                )
        return None

    # ------------------------------------------------------------------
    # Priority order (patched)
    # - In strong trend (ADX(H1) >= 35), prefer impulse/continuation, suppress pullbacks.
    # ------------------------------------------------------------------
    if adx_h1 >= 35:
        sig = _impulse_cont()
        if sig:
            return sig
        # continuation next
        if trend_dir == "LONG":
            sig = _breakout_cont_long()
            if sig:
                return sig
            return _breakout_long()
        else:
            sig = _breakout_cont_short()
            if sig:
                return sig
            return _breakout_short()

    # normal priority
    if trend_dir == "LONG":
        sig = _pullback_long()
        if sig:
            return sig
        sig = _breakout_long()
        if sig:
            return sig
        sig = _breakout_cont_long()
        return sig

    if trend_dir == "SHORT":
        sig = _pullback_short()
        if sig:
            return sig
        sig = _breakout_short()
        if sig:
            return sig
        sig = _breakout_cont_short()
        return sig

    return None