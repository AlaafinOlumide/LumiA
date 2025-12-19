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
    atr,
    ema,
)

@dataclass
class Signal:
    direction: str          # "LONG" or "SHORT"
    price: float
    time: dt.datetime
    reason: str
    extra: Dict[str, Any]

# -------------------------
# Sessions + weekend filter
# -------------------------
def is_within_sessions(
    now_utc: dt.datetime,
    session_1_start: int,
    session_1_end: int,
    session_2_start: Optional[int],
    session_2_end: Optional[int],
    trade_weekends: bool = False,
) -> bool:
    # Weekend block (Sat=5, Sun=6)
    if not trade_weekends and now_utc.weekday() >= 5:
        return False

    hhmm = now_utc.hour * 100 + now_utc.minute
    in_s1 = session_1_start <= hhmm <= session_1_end
    if session_2_start is not None and session_2_end is not None and session_2_start > 0 and session_2_end > 0:
        in_s2 = session_2_start <= hhmm <= session_2_end
    else:
        in_s2 = False
    return in_s1 or in_s2

# -------------------------
# Trend detection (H1 / M15)
# -------------------------
def detect_trend_h1(h1_df: pd.DataFrame) -> Optional[str]:
    if h1_df is None or len(h1_df) < 60:
        return None

    close = h1_df["close"]
    high = h1_df["high"]
    low = h1_df["low"]

    ema20 = ema(close, 20)
    ema50 = ema(close, 50)
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

    ema20 = ema(close, 20)
    ema50 = ema(close, 50)
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
    ema20 = ema(close, 20)
    ema50 = ema(close, 50)
    c = float(close.iloc[-1])
    e20 = float(ema20.iloc[-1])
    e50 = float(ema50.iloc[-1])

    if trend == "LONG":
        return e20 > e50 and c > e20
    if trend == "SHORT":
        return e20 < e50 and c < e20
    return False

# -------------------------
# M15 structure (NEW)
# -------------------------
def m15_structure_ok(m15_df: pd.DataFrame, direction: str) -> bool:
    """
    Lightweight structure filter:
    LONG  -> last swing is HH & HL + close above EMA20
    SHORT -> last swing is LL & LH + close below EMA20
    """
    if m15_df is None or len(m15_df) < 12:
        return False

    highs = m15_df["high"]
    lows = m15_df["low"]
    close = m15_df["close"]
    e20 = ema(close, 20)

    # Compare recent pivots (simple, stable heuristic)
    if direction == "LONG":
        hh = highs.iloc[-3] < highs.iloc[-1]
        hl = lows.iloc[-3] < lows.iloc[-1]
        above = close.iloc[-1] > e20.iloc[-1]
        return bool(hh and hl and above)

    if direction == "SHORT":
        ll = lows.iloc[-3] > lows.iloc[-1]
        lh = highs.iloc[-3] > highs.iloc[-1]
        below = close.iloc[-1] < e20.iloc[-1]
        return bool(ll and lh and below)

    return False

# -------------------------
# Scoring gate (NEW)
# -------------------------
def compute_entry_score(context: Dict[str, Any]) -> int:
    """
    0..10 gate. Keep it explainable.
    """
    score = 0
    score += 2 if context.get("htf_trend_aligned") else 0
    score += 2 if context.get("m15_structure_ok") else 0
    score += 2 if context.get("m5_displacement") else 0
    score += 1 if context.get("adx_rising") else 0
    score += 1 if context.get("rsi_reset") else 0
    score += 1 if context.get("stoch_cross") else 0
    score += 1 if context.get("news_clear") else 0
    return int(max(0, min(10, score)))

# -------------------------
# Market regime (for TP/SL tuning)
# -------------------------
def market_regime(h1_df: pd.DataFrame) -> str:
    if h1_df is None or len(h1_df) < 60:
        return "Unknown"

    close = h1_df["close"]
    high = h1_df["high"]
    low = h1_df["low"]

    upper, mid, lower = bollinger_bands(close, period=20, std_factor=2.0)
    bw = (upper - lower) / mid.replace(0, pd.NA)
    a = atr(high, low, close, period=14)

    bw_last = float(bw.iloc[-1]) if pd.notna(bw.iloc[-1]) else 0.0
    atr_last = float(a.iloc[-1]) if pd.notna(a.iloc[-1]) else 0.0

    if bw_last <= 0.008 and atr_last <= 10:
        return "Low Volatility / Compression"
    if bw_last >= 0.02 or atr_last >= 20:
        return "High Volatility"
    return "Normal Volatility"

# -------------------------
# Dynamic TP/SL (improved)
# -------------------------
def dynamic_tp_sl(signal: Signal, h1_df: pd.DataFrame, regime: str) -> None:
    if h1_df is None or len(h1_df) < 60:
        return

    a = atr(h1_df["high"], h1_df["low"], h1_df["close"], period=14)
    atr_h1 = float(a.iloc[-1]) if pd.notna(a.iloc[-1]) else None
    if atr_h1 is None or atr_h1 <= 0:
        return

    entry = float(signal.price)

    # Multipliers adapt to regime (this is what fixes “TP not hit” in compression)
    if regime == "Low Volatility / Compression":
        sl_mult, tp1_mult, tp2_mult = (0.85, 1.00, None)  # TP2 disabled
    elif regime == "High Volatility":
        sl_mult, tp1_mult, tp2_mult = (0.95, 1.50, 3.00)
    else:
        sl_mult, tp1_mult, tp2_mult = (0.90, 1.20, 2.20)

    if signal.direction == "LONG":
        sl = entry - (atr_h1 * sl_mult)
        tp1 = entry + (atr_h1 * tp1_mult)
        tp2 = entry + (atr_h1 * tp2_mult) if tp2_mult else None
    else:
        sl = entry + (atr_h1 * sl_mult)
        tp1 = entry - (atr_h1 * tp1_mult)
        tp2 = entry - (atr_h1 * tp2_mult) if tp2_mult else None

    signal.extra["atr_h1"] = atr_h1
    signal.extra["sl"] = sl
    signal.extra["tp1"] = tp1
    signal.extra["tp2"] = tp2
    signal.extra["regime"] = regime

# -------------------------
# M5 triggers (rewritten pullback + better continuation)
# -------------------------
def _displacement_candle(m5_df: pd.DataFrame, direction: str) -> bool:
    """
    “Intent” candle: big body + closes in direction + not a doji.
    """
    last = m5_df.iloc[-1]
    body = abs(last["close"] - last["open"])
    rng = max(1e-9, (last["high"] - last["low"]))
    strong = (body / rng) >= 0.60

    if direction == "LONG":
        return bool(strong and last["close"] > last["open"])
    if direction == "SHORT":
        return bool(strong and last["close"] < last["open"])
    return False

def _m5_pullback(m5_df: pd.DataFrame, direction: str) -> bool:
    """
    Proper pullback:
    - Price tags BB mid (or slightly beyond) toward trend side
    - Stoch crosses back in direction
    - Candle confirmation (bull/bear or engulfing)
    - RSI reset zone (not overextended)
    - Displacement candle present (intent)
    """
    close = m5_df["close"]
    high = m5_df["high"]
    low = m5_df["low"]
    open_ = m5_df["open"]

    bb_u, bb_m, bb_l = bollinger_bands(close, period=20, std_factor=2.0)
    rsi_s = rsi(close, 14)
    k, d = stochastic_oscillator(high, low, close, k_period=14, d_period=3)
    bull_eng = bullish_engulfing(open_, close)
    bear_eng = bearish_engulfing(open_, close)

    c = float(close.iloc[-1])
    o = float(open_.iloc[-1])
    k_now, d_now = float(k.iloc[-1]), float(d.iloc[-1])
    k_prev, d_prev = float(k.iloc[-2]), float(d.iloc[-2])
    r = float(rsi_s.iloc[-1])

    bbm = float(bb_m.iloc[-1])
    bbu = float(bb_u.iloc[-1])
    bbl = float(bb_l.iloc[-1])

    stoch_cross_up = (k_now > d_now) and (k_prev <= d_prev)
    stoch_cross_dn = (k_now < d_now) and (k_prev >= d_prev)
    disp_long = _displacement_candle(m5_df, "LONG")
    disp_short = _displacement_candle(m5_df, "SHORT")

    if direction == "LONG":
        # pullback should be toward mid/lower then reclaim
        touched = (c <= bbm * 1.002)  # tagged mid-ish
        rsi_reset = 40 <= r <= 58
        candle_ok = (c > o) or bool(bull_eng.iloc[-1])
        return bool(touched and stoch_cross_up and candle_ok and rsi_reset and disp_long)

    if direction == "SHORT":
        touched = (c >= bbm * 0.998)
        rsi_reset = 42 <= r <= 60
        candle_ok = (c < o) or bool(bear_eng.iloc[-1])
        return bool(touched and stoch_cross_dn and candle_ok and rsi_reset and disp_short)

    return False

def _m5_breakout(m5_df: pd.DataFrame, direction: str) -> bool:
    close = m5_df["close"]
    high = m5_df["high"]
    low = m5_df["low"]

    bb_u, bb_m, bb_l = bollinger_bands(close, period=20, std_factor=2.0)
    rsi_s = rsi(close, 14)
    k, d = stochastic_oscillator(high, low, close, k_period=14, d_period=3)

    c = float(close.iloc[-1])
    c_prev = float(close.iloc[-2])
    bbu = float(bb_u.iloc[-1]); bbu_prev = float(bb_u.iloc[-2])
    bbl = float(bb_l.iloc[-1]); bbl_prev = float(bb_l.iloc[-2])
    r = float(rsi_s.iloc[-1])
    k_now = float(k.iloc[-1])

    if direction == "LONG":
        prev_inside = c_prev <= bbu_prev * 0.999
        now_break = c > bbu * 1.001
        return bool(prev_inside and now_break and r >= 54 and k_now >= 55 and _displacement_candle(m5_df, "LONG"))

    if direction == "SHORT":
        prev_inside = c_prev >= bbl_prev * 1.001
        now_break = c < bbl * 0.999
        return bool(prev_inside and now_break and r <= 46 and k_now <= 45 and _displacement_candle(m5_df, "SHORT"))

    return False

def _m5_breakout_continuation(m5_df: pd.DataFrame, direction: str) -> bool:
    """
    Continuation should require:
    - Riding the band
    - ADX supportive
    - AND a fresh “re-acceleration” candle (displacement)
    This stops late entries.
    """
    close = m5_df["close"]
    high = m5_df["high"]
    low = m5_df["low"]

    bb_u, bb_m, bb_l = bollinger_bands(close, period=20, std_factor=2.0)
    adx_s, pdi, mdi = adx(high, low, close, period=14)

    c = float(close.iloc[-1])
    bbu = float(bb_u.iloc[-1])
    bbl = float(bb_l.iloc[-1])

    a = float(adx_s.iloc[-1]) if pd.notna(adx_s.iloc[-1]) else 0.0
    p = float(pdi.iloc[-1]) if pd.notna(pdi.iloc[-1]) else 0.0
    m = float(mdi.iloc[-1]) if pd.notna(mdi.iloc[-1]) else 0.0

    if direction == "LONG":
        riding = c >= bbu * 0.999
        ok = a >= 22 and p > m and _displacement_candle(m5_df, "LONG")
        return bool(riding and ok)

    if direction == "SHORT":
        riding = c <= bbl * 1.001
        ok = a >= 22 and m > p and _displacement_candle(m5_df, "SHORT")
        return bool(riding and ok)

    return False

def trigger_signal_m5(
    m5_df: pd.DataFrame,
    trend_dir: str,
    m15_df: pd.DataFrame,
    h1_df: pd.DataFrame,
    high_news: bool,
    min_score: int = 7,
) -> Optional[Signal]:
    """
    Priority:
      1) Pullback (structure + intent)
      2) Breakout (fresh + intent)
      3) Breakout continuation (re-acceleration only)
    AND score-gate required.
    """
    if m5_df is None or len(m5_df) < 120:
        return None

    close = m5_df["close"]
    high = m5_df["high"]
    low = m5_df["low"]

    # M5 indicators for message payload
    bb_u, bb_m, bb_l = bollinger_bands(close, period=20, std_factor=2.0)
    rsi_s = rsi(close, 14)
    k, d = stochastic_oscillator(high, low, close, k_period=14, d_period=3)
    adx_s, pdi, mdi = adx(high, low, close, period=14)

    c = float(close.iloc[-1])
    bar_time = m5_df["datetime"].iloc[-1]
    if isinstance(bar_time, pd.Timestamp):
        bar_time = bar_time.to_pydatetime()

    # Context
    m15_ok = m15_structure_ok(m15_df, trend_dir)
    # “ADX rising” simple proxy: last adx > previous
    adx_now = float(adx_s.iloc[-1]) if pd.notna(adx_s.iloc[-1]) else 0.0
    adx_prev = float(adx_s.iloc[-2]) if pd.notna(adx_s.iloc[-2]) else 0.0
    adx_rising = adx_now > adx_prev

    r_now = float(rsi_s.iloc[-1])
    rsi_reset = (40 <= r_now <= 58) if trend_dir == "LONG" else (42 <= r_now <= 60)

    k_now, d_now = float(k.iloc[-1]), float(d.iloc[-1])
    k_prev, d_prev = float(k.iloc[-2]), float(d.iloc[-2])
    stoch_cross = ((k_now > d_now) and (k_prev <= d_prev)) if trend_dir == "LONG" else ((k_now < d_now) and (k_prev >= d_prev))

    # Trigger candidates
    pullback = _m5_pullback(m5_df, trend_dir)
    breakout = _m5_breakout(m5_df, trend_dir)
    cont = _m5_breakout_continuation(m5_df, trend_dir)

    # Determine which setup would fire (priority)
    setup_type = None
    reason = None

    if pullback:
        setup_type = "PULLBACK_LONG" if trend_dir == "LONG" else "PULLBACK_SHORT"
        reason = "Pullback: M15 structure ok + BB mid tag + stoch reset + displacement candle."
    elif breakout:
        setup_type = "BREAKOUT_LONG" if trend_dir == "LONG" else "BREAKOUT_SHORT"
        reason = "Breakout: fresh BB break from inside + momentum + displacement candle."
    elif cont:
        setup_type = "BREAKOUT_CONT_LONG" if trend_dir == "LONG" else "BREAKOUT_CONT_SHORT"
        reason = "Continuation: riding band + ADX support + re-acceleration candle."
    else:
        return None

    # Score gate (blocks weak signals in chop/compression)
    context = {
        "htf_trend_aligned": True,
        "m15_structure_ok": m15_ok,
        "m5_displacement": _displacement_candle(m5_df, trend_dir),
        "adx_rising": adx_rising,
        "rsi_reset": rsi_reset,
        "stoch_cross": stoch_cross,
        "news_clear": (not high_news),
    }
    entry_score = compute_entry_score(context)
    if entry_score < min_score:
        return None

    extra = {
        "setup_type": setup_type,
        "entry_score": entry_score,
        "m15_structure_ok": m15_ok,
        "m5_rsi": r_now,
        "m5_stoch_k": k_now,
        "m5_stoch_d": d_now,
        "bb_upper": float(bb_u.iloc[-1]),
        "bb_mid": float(bb_m.iloc[-1]),
        "bb_lower": float(bb_l.iloc[-1]),
        "adx_m5": adx_now,
        "plus_di_m5": float(pdi.iloc[-1]) if pd.notna(pdi.iloc[-1]) else 0.0,
        "minus_di_m5": float(mdi.iloc[-1]) if pd.notna(mdi.iloc[-1]) else 0.0,
    }

    return Signal(direction=trend_dir, price=c, time=bar_time, reason=reason, extra=extra)

# -------------------------
# Confidence (kept but improved with score)
# -------------------------
def confidence_label(score: int) -> str:
    if score >= 75:
        return "High"
    if score >= 55:
        return "Medium"
    return "Low"

def compute_confidence(
    trend_source: str,
    setup_type: str,
    adx_h1: float,
    adx_m5: float,
    high_news: bool,
    entry_score: int,
) -> int:
    score = 50
    score += 10 if trend_source == "H1" else 6

    if setup_type.startswith("PULLBACK"):
        score += 16
    elif setup_type.startswith("BREAKOUT_CONT"):
        score += 7
    elif setup_type.startswith("BREAKOUT"):
        score += 10

    if adx_h1 >= 30:
        score += 12
    elif adx_h1 >= 20:
        score += 7
    else:
        score -= 6

    if adx_m5 >= 25:
        score += 6
    elif adx_m5 < 15:
        score -= 5

    if high_news:
        score -= 12

    # Entry gate contributes meaningfully (stops “high confidence but weak structure”)
    score += int((entry_score - 7) * 4)  # 7->0, 10->+12

    return max(0, min(100, score))
