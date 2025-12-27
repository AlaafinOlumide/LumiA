from dataclasses import dataclass, asdict
import pandas as pd
import uuid
import csv
import os
from datetime import timedelta

@dataclass
class JournalEntry:
    trade_id: str
    symbol: str
    tf: str
    setup: str
    direction: str
    signal_time: str
    entry: float
    sl: float
    tp1: float
    tp2: float
    expiry_time: str
    status: str = "OPEN"     # OPEN, TP1, TP2, SL, EXPIRED
    exit_time: str = ""
    exit_price: float = 0.0
    r_multiple: float = 0.0

def new_trade_id() -> str:
    return uuid.uuid4().hex[:8]

def _risk(entry: float, sl: float, min_risk_points: float) -> float:
    return max(min_risk_points, abs(entry - sl))

def compute_r_multiple(entry: float, sl: float, direction: str, exit_price: float, min_risk_points: float, max_abs_r: float) -> float:
    risk = _risk(entry, sl, min_risk_points)
    if direction == "BUY":
        r = (exit_price - entry) / risk
    else:
        r = (entry - exit_price) / risk
    # clamp to stop nonsensical extremes
    if r > max_abs_r:
        r = max_abs_r
    if r < -max_abs_r:
        r = -max_abs_r
    return float(r)

def append_csv(path: str, row: dict) -> None:
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)

def open_journal_entry(signal, expiry_minutes: int) -> JournalEntry:
    expiry = (signal.candle_time.to_pydatetime() + timedelta(minutes=expiry_minutes)).isoformat()
    return JournalEntry(
        trade_id=new_trade_id(),
        symbol=signal.symbol,
        tf=signal.tf,
        setup=signal.setup,
        direction=signal.direction,
        signal_time=signal.candle_time.to_pydatetime().isoformat(),
        entry=float(signal.entry),
        sl=float(signal.sl),
        tp1=float(signal.tp1),
        tp2=float(signal.tp2),
        expiry_time=expiry,
    )

def update_journal_status(
    entry: JournalEntry,
    last_close_time: pd.Timestamp,
    last_close: float,
    min_risk_points: float,
    max_abs_r: float
) -> JournalEntry:
    """
    Called every new candle. Determines if TP/SL hit; otherwise EXPIRED on expiry_time.
    """
    if entry.status != "OPEN":
        return entry

    # Compare vs last_close only (signal bot, not a live execution engine)
    # If you later want OHLC-based hit detection, feed candle high/low too.
    if entry.direction == "BUY":
        if last_close >= entry.tp2:
            entry.status = "TP2"
        elif last_close >= entry.tp1:
            entry.status = "TP1"
        elif last_close <= entry.sl:
            entry.status = "SL"
    else:
        if last_close <= entry.tp2:
            entry.status = "TP2"
        elif last_close <= entry.tp1:
            entry.status = "TP1"
        elif last_close >= entry.sl:
            entry.status = "SL"

    # expiry check
    expiry_ts = pd.to_datetime(entry.expiry_time, utc=True)
    if entry.status == "OPEN" and last_close_time >= expiry_ts:
        entry.status = "EXPIRED"

    if entry.status != "OPEN":
        entry.exit_time = last_close_time.to_pydatetime().isoformat()
        entry.exit_price = float(last_close)
        entry.r_multiple = compute_r_multiple(entry.entry, entry.sl, entry.direction, entry.exit_price, min_risk_points, max_abs_r)

    return entry
