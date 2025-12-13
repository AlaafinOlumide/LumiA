# trade_manager.py
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, Optional
import uuid


def new_trade_id() -> str:
    return uuid.uuid4().hex[:10]


@dataclass
class ActiveTrade:
    trade_id: str
    opened_time: dt.datetime
    direction: str  # "LONG" / "SHORT"
    setup_type: str
    entry: float
    sl: float
    tp1: float
    tp2: float
    confidence: int
    trend_source: str  # "H1" or "M15"

    status: str = "OPEN"  # OPEN / TP1 / TP2 / SL / INVALIDATED / CLOSED
    exit_time: Optional[dt.datetime] = None
    exit_price: Optional[float] = None
    result_r: Optional[float] = None

    invalidation_deadline: Optional[dt.datetime] = None
    invalidated_reason: Optional[str] = None


def compute_r_result(direction: str, entry: float, sl: float, exit_price: float) -> float:
    """
    R = (exit - entry) / (entry - sl) for LONG
    R = (entry - exit) / (sl - entry) for SHORT
    """
    if direction == "LONG":
        risk = entry - sl
        if risk <= 0:
            return 0.0
        return (exit_price - entry) / risk

    risk = sl - entry
    if risk <= 0:
        return 0.0
    return (entry - exit_price) / risk


def check_tp_sl_hit(trade: ActiveTrade, last_closed_m5_row) -> Optional[Dict[str, Any]]:
    """
    Determines if the last closed candle hits SL/TP1/TP2.
    Uses OHLC high/low of that candle.
    """
    high = float(last_closed_m5_row["high"])
    low = float(last_closed_m5_row["low"])

    if trade.direction == "LONG":
        # Worst-case ordering assumption: SL hits before TP in same candle
        if low <= trade.sl:
            return {"status": "SL", "exit_price": trade.sl}
        if high >= trade.tp2:
            return {"status": "TP2", "exit_price": trade.tp2}
        if high >= trade.tp1:
            return {"status": "TP1", "exit_price": trade.tp1}
        return None

    # SHORT
    if high >= trade.sl:
        return {"status": "SL", "exit_price": trade.sl}
    if low <= trade.tp2:
        return {"status": "TP2", "exit_price": trade.tp2}
    if low <= trade.tp1:
        return {"status": "TP1", "exit_price": trade.tp1}
    return None


def should_invalidate(
    trade: ActiveTrade,
    last_closed_m5,
    bb_mid: float,
    bb_upper: float,
    bb_lower: float,
    rsi_m5: float,
) -> Optional[str]:
    """
    Signal invalidation rules (only valid BEFORE invalidation_deadline).

    LONG invalidation examples:
      - Candle CLOSES below BB mid by a margin (trend pullback failed)
      - RSI drops below 45 quickly
      - (Optional) candle closes back inside band after a breakout signal

    SHORT invalidation examples:
      - Candle CLOSES above BB mid
      - RSI rises above 55 quickly
    """
    close = float(last_closed_m5["close"])
    open_ = float(last_closed_m5["open"])

    # small buffers to avoid noise
    mid_break_buffer = 0.0015  # 0.15%

    if trade.direction == "LONG":
        # 1) Loss of structure: close < mid
        if close < bb_mid * (1 - mid_break_buffer):
            return "Invalidation: M5 close broke below BB mid (pullback failed)."

        # 2) Momentum flip
        if rsi_m5 < 45:
            return "Invalidation: RSI(M5) flipped bearish (<45)."

        # 3) Breakout-type failure: close back inside band after being outside
        if "BREAKOUT" in trade.setup_type and close < bb_upper:
            # for long breakout, falling back below upper band quickly is weakness
            return "Invalidation: Breakout failed (close back below upper BB)."

        # 4) Big bearish candle (simple proxy)
        if close < open_ and (open_ - close) > (bb_mid - bb_lower) * 0.6:
            return "Invalidation: strong bearish candle against LONG setup."

        return None

    # SHORT invalidation
    if close > bb_mid * (1 + mid_break_buffer):
        return "Invalidation: M5 close broke above BB mid (pullback failed)."

    if rsi_m5 > 55:
        return "Invalidation: RSI(M5) flipped bullish (>55)."

    if "BREAKOUT" in trade.setup_type and close > bb_lower:
        return "Invalidation: Breakout failed (close back above lower BB)."

    if close > open_ and (close - open_) > (bb_upper - bb_mid) * 0.6:
        return "Invalidation: strong bullish candle against SHORT setup."

    return None
