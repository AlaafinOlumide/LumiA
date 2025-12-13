# trade_manager.py
import csv
import os
import uuid
import datetime as dt
from dataclasses import dataclass
from typing import Optional, Dict, Any

import pandas as pd


@dataclass
class ActiveTrade:
    trade_id: str
    opened_time: dt.datetime
    direction: str
    setup_type: str
    entry: float
    sl: float
    tp1: float
    tp2: float
    confidence: int
    trend_source: str
    status: str = "OPEN"  # OPEN | TP1 | TP2 | SL | INVALIDATED
    invalidation_deadline: Optional[dt.datetime] = None
    invalidated_reason: Optional[str] = None
    exit_time: Optional[dt.datetime] = None
    exit_price: Optional[float] = None
    result_r: Optional[float] = None


class TradeJournal:
    """
    CSV journal.
    - Append on open
    - Update rows on close/invalidate
    """
    def __init__(self, path: str = "trades.csv"):
        self.path = path
        self._ensure_file()

    def _ensure_file(self) -> None:
        if os.path.exists(self.path):
            return
        with open(self.path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "trade_id", "opened_time_utc", "direction", "setup_type",
                "entry", "sl", "tp1", "tp2",
                "confidence", "trend_source",
                "status",
                "exit_time_utc", "exit_price",
                "result_r",
                "invalidated_reason"
            ])

    def append_open(self, t: ActiveTrade) -> None:
        with open(self.path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                t.trade_id, t.opened_time.isoformat(),
                t.direction, t.setup_type,
                f"{t.entry:.2f}", f"{t.sl:.2f}", f"{t.tp1:.2f}", f"{t.tp2:.2f}",
                t.confidence, t.trend_source,
                t.status,
                "", "", "", ""
            ])

    def update_trade(self, t: ActiveTrade) -> None:
        # Load all, update matching ID, rewrite
        if not os.path.exists(self.path):
            return

        rows = []
        with open(self.path, "r", newline="") as f:
            rows = list(csv.reader(f))

        header = rows[0]
        out = [header]

        for r in rows[1:]:
            if not r:
                continue
            if r[0] != t.trade_id:
                out.append(r)
                continue

            # Update row
            out.append([
                t.trade_id, t.opened_time.isoformat(),
                t.direction, t.setup_type,
                f"{t.entry:.2f}", f"{t.sl:.2f}", f"{t.tp1:.2f}", f"{t.tp2:.2f}",
                t.confidence, t.trend_source,
                t.status,
                t.exit_time.isoformat() if t.exit_time else "",
                f"{t.exit_price:.2f}" if t.exit_price is not None else "",
                f"{t.result_r:.2f}" if t.result_r is not None else "",
                t.invalidated_reason or ""
            ])

        with open(self.path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerows(out)


def new_trade_id() -> str:
    return uuid.uuid4().hex


# ---------------------------
# Invalidation + outcome checks
# ---------------------------

def compute_r_result(direction: str, entry: float, sl: float, exit_price: float) -> float:
    risk = (entry - sl) if direction == "LONG" else (sl - entry)
    if risk <= 0:
        return 0.0
    pnl = (exit_price - entry) if direction == "LONG" else (entry - exit_price)
    return pnl / risk


def should_invalidate(
    trade: ActiveTrade,
    last_closed_m5: pd.Series,
    bb_mid: float,
    bb_upper: float,
    bb_lower: float,
    rsi_m5: float,
) -> Optional[str]:
    """
    Simple + effective invalidation:
    - Pullback invalidation:
        LONG: close back under BB mid + RSI < 45 within 2 candles => invalidate
        SHORT: close back over BB mid + RSI > 55 within 2 candles => invalidate
    - Breakout invalidation:
        LONG: close back inside band (close < upper) AND RSI < 50 within 2 candles
        SHORT: close back inside band (close > lower) AND RSI > 50 within 2 candles
    - Continuation invalidation:
        LONG: close < mid within 1 candle
        SHORT: close > mid within 1 candle
    """
    c = float(last_closed_m5["close"])

    st = trade.setup_type

    if st.startswith("PULLBACK"):
        if trade.direction == "LONG":
            if c < bb_mid and rsi_m5 < 45:
                return "Invalidated: pullback failed (close < BB mid + RSI<45)"
        else:
            if c > bb_mid and rsi_m5 > 55:
                return "Invalidated: pullback failed (close > BB mid + RSI>55)"

    if st.startswith("BREAKOUT") and not st.startswith("BREAKOUT_CONT"):
        if trade.direction == "LONG":
            if c < bb_upper and rsi_m5 < 50:
                return "Invalidated: breakout failed (back inside BB + RSI<50)"
        else:
            if c > bb_lower and rsi_m5 > 50:
                return "Invalidated: breakout failed (back inside BB + RSI>50)"

    if st.startswith("BREAKOUT_CONT"):
        if trade.direction == "LONG" and c < bb_mid:
            return "Invalidated: continuation lost (close < BB mid)"
        if trade.direction == "SHORT" and c > bb_mid:
            return "Invalidated: continuation lost (close > BB mid)"

    return None


def check_tp_sl_hit(trade: ActiveTrade, candle: pd.Series) -> Optional[Dict[str, Any]]:
    """
    Uses HIGH/LOW of closed candle to see if SL/TP hit.
    Priority: SL first (conservative).
    """
    h = float(candle["high"])
    l = float(candle["low"])

    if trade.direction == "LONG":
        if l <= trade.sl:
            return {"status": "SL", "exit_price": trade.sl}
        if h >= trade.tp2:
            return {"status": "TP2", "exit_price": trade.tp2}
        if h >= trade.tp1:
            return {"status": "TP1", "exit_price": trade.tp1}

    else:
        if h >= trade.sl:
            return {"status": "SL", "exit_price": trade.sl}
        if l <= trade.tp2:
            return {"status": "TP2", "exit_price": trade.tp2}
        if l <= trade.tp1:
            return {"status": "TP1", "exit_price": trade.tp1}

    return None
