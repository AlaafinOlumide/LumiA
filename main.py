# strategy.py
from __future__ import annotations

from dataclasses import dataclass, field
import datetime as dt
from typing import Optional, Dict, Any

import pandas as pd

import indicators


@dataclass
class Signal:
    direction: str  # "LONG" or "SHORT"
    entry: float
    sl: float
    tp1: float
    tp2: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------
# Sessions
# ---------------------------
def active_session(now_utc: dt.datetime, *, enable_asia: bool, trade_weekends: bool) -> Optional[str]:
    """
    Returns: "ASIA", "LONDON_NY", or None
    """
    if not trade_weekends and now_utc.weekday() >= 5:
        return None

    h = now_utc.hour

    if enable_asia and (0 <= h < 2):
        return "ASIA"

    if 7 <= h < 20:
        return "LONDON_NY"

    return None


# ---------------------------
# Trend detection
# ---------------------------
def detect_trend_h1(h1_df: pd.DataFrame) -> Optional[str]:
    """
    H1 trend: EMA(50) vs EMA(200) + price location.
    Returns "LONG"/"SHORT"/None
    """
    if len(h1_df) < 220:
        return None

    close = h1_df["close"]
    ema50 = indicators.ema(close, 50)
    ema200 = indicators.ema(close, 200)

    if ema50.iloc[-1] > ema200.iloc[-1] and close.iloc[-1] > ema50.iloc[-1]:
        return "LONG"
    if ema50.iloc[-1] < ema200.iloc[-1] and close.iloc[-1] < ema50.iloc[-1]:
        return "SHORT"
    return None


def m15_structure_ok(m15_df: pd.DataFrame, trend_dir: str) -> bool:
    """
    M15 structure filter:
      - EMA(20) slope in direction
      - Close above EMA(20) for LONG, below for SHORT
    """
    if len(m15_df) < 60:
        return False

    close = m15_df["close"]
    ema20 = indicators.ema(close, 20)

    slope = ema20.iloc[-1] - ema20.iloc[-5]  # rough slope
    if trend_dir == "LONG":
        return slope > 0 and close.iloc[-1] >= ema20.iloc[-1]
    else:
        return slope < 0 and close.iloc[-1] <= ema20.iloc[-1]


# ---------------------------
# Setup logic
# ---------------------------
def score_entry_gate(
    *,
    m5_df: pd.DataFrame,
    m15_df: pd.DataFrame,
    trend_dir: str,
    session: str,
) -> tuple[int, Dict[str, Any]]:
    """
    Score-based gate: returns (score, debug_details).
    """
    close = m5_df["close"]
    high = m5_df["high"]
    low = m5_df["low"]

    upper, mid, lower = indicators.bollinger(close, 20, 2.0)
    rsi = indicators.rsi(close, 14)
    k, d = indicators.stoch_kd(high, low, close, 14, 3, 3)
    adx_v, plus_di, minus_di = indicators.adx(high, low, close, 14)

    score = 0
    dbg: Dict[str, Any] = {}

    # 1) trend alignment (+2)
    score += 2
    dbg["trend_align"] = True

    # 2) M15 structure (+2)
    m15_ok = m15_structure_ok(m15_df, trend_dir)
    if m15_ok:
        score += 2
    dbg["m15_ok"] = m15_ok

    # 3) BB location (+1)
    last_close = float(close.iloc[-1])
    last_mid = float(mid.iloc[-1])
    last_upper = float(upper.iloc[-1])
    last_lower = float(lower.iloc[-1])

    if trend_dir == "LONG":
        # prefer pullback near mid/lower
        if last_close <= last_mid:
            score += 1
            dbg["bb_pullback_ok"] = True
        else:
            dbg["bb_pullback_ok"] = False
    else:
        if last_close >= last_mid:
            score += 1
            dbg["bb_pullback_ok"] = True
        else:
            dbg["bb_pullback_ok"] = False

    # 4) stochastic confirmation (+1)
    k0, d0 = float(k.iloc[-1]), float(d.iloc[-1])
    k1, d1 = float(k.iloc[-2]), float(d.iloc[-2])
    stoch_cross_up = (k1 < d1) and (k0 > d0)
    stoch_cross_down = (k1 > d1) and (k0 < d0)

    if trend_dir == "LONG" and stoch_cross_up:
        score += 1
        dbg["stoch_confirm"] = "cross_up"
    elif trend_dir == "SHORT" and stoch_cross_down:
        score += 1
        dbg["stoch_confirm"] = "cross_down"
    else:
        dbg["stoch_confirm"] = "none"

    # 5) candle confirmation (+1)
    o = float(m5_df["open"].iloc[-1])
    c = float(m5_df["close"].iloc[-1])
    bullish = c > o
    bearish = c < o
    if trend_dir == "LONG" and bullish:
        score += 1
        dbg["candle_confirm"] = "bull"
    elif trend_dir == "SHORT" and bearish:
        score += 1
        dbg["candle_confirm"] = "bear"
    else:
        dbg["candle_confirm"] = "none"

    # 6) avoid extreme RSI in ASIA (stricter) (-1 penalty)
    r = float(rsi.iloc[-1])
    if session == "ASIA":
        if trend_dir == "LONG" and r > 70:
            score -= 1
            dbg["asia_rsi_penalty"] = True
        elif trend_dir == "SHORT" and r < 30:
            score -= 1
            dbg["asia_rsi_penalty"] = True
        else:
            dbg["asia_rsi_penalty"] = False

    dbg["rsi_m5"] = r
    dbg["stoch_k_m5"] = k0
    dbg["adx_m5"] = float(adx_v.iloc[-1])
    dbg["bb_upper"] = last_upper
    dbg["bb_mid"] = last_mid
    dbg["bb_lower"] = last_lower

    return score, dbg


def make_sl_tp_from_rr(entry: float, sl: float, direction: str, rr: float) -> float:
    risk = abs(entry - sl)
    if risk <= 0:
        risk = 0.01
    if direction == "LONG":
        return entry + risk * rr
    return entry - risk * rr


def trigger_signal_m5(
    *,
    m5_df: pd.DataFrame,
    m15_df: pd.DataFrame,
    h1_df: pd.DataFrame,
    trend_dir: str,
    high_news: bool,
    min_score: int,
    session: str,
    asia_extra_buffer: int,
    tp1_rr: float,
    tp2_rr: float,
    asia_tp1_rr: float,
    sl_atr_mult: float,
) -> Optional[Signal]:
    """
    Produces pullback-based signals only (no continuation in ASIA).
    """
    if high_news:
        return None

    # ASIA rule: pullback only, stricter scoring
    required = min_score + (asia_extra_buffer if session == "ASIA" else 0)

    score, dbg = score_entry_gate(m5_df=m5_df, m15_df=m15_df, trend_dir=trend_dir, session=session)

    if score < required:
        return None

    # Build SL using ATR(H1) * mult
    atr_h1 = float(indicators.atr(h1_df["high"], h1_df["low"], h1_df["close"], 14).iloc[-1])
    if atr_h1 <= 0:
        atr_h1 = 5.0

    entry = float(m5_df["close"].iloc[-1])
    direction = trend_dir

    if direction == "LONG":
        sl = entry - atr_h1 * sl_atr_mult
    else:
        sl = entry + atr_h1 * sl_atr_mult

    # TP logic
    if session == "ASIA":
        tp1 = make_sl_tp_from_rr(entry, sl, direction, asia_tp1_rr)
        tp2 = None
        tp_mode = "TP1_ONLY"
    else:
        tp1 = make_sl_tp_from_rr(entry, sl, direction, tp1_rr)
        tp2 = make_sl_tp_from_rr(entry, sl, direction, tp2_rr)
        tp_mode = "TP1_TP2"

    sig = Signal(
        direction=direction,
        entry=entry,
        sl=sl,
        tp1=tp1,
        tp2=tp2,
        extra={
            "setup_type": f"PULLBACK_{direction}",
            "entry_score": int(score),
            "session": session,
            "tp_mode": tp_mode,
            **dbg,
        },
    )
    return sig