# journal.py
from __future__ import annotations

import os
import uuid
import datetime as dt
from typing import Any, Dict, Optional, Tuple, List

import pandas as pd


COLUMNS = [
    "signal_id",
    "symbol",
    "direction",
    "setup_type",
    "trend_source",
    "trend_dir",
    "time_utc",
    "entry",
    "sl",
    "tp1",
    "tp2",
    "entry_score",
    "confidence",
    "adx_h1",
    "adx_m5",
    "regime",
    "status",          # OPEN/CLOSED
    "result",          # TP1/TP2/SL/EXPIRED/NONE
    "hit_time_utc",    # when first level hit (or expired time)
    "exit_price",      # price at expiry/none
    "pnl_r",           # result in R (approx)
    "notes",
]


def _ensure_csv(path: str) -> None:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    df = pd.DataFrame(columns=COLUMNS)
    df.to_csv(path, index=False)


def new_signal_id() -> str:
    return uuid.uuid4().hex[:12]


def append_open_signal(path: str, row: Dict[str, Any]) -> None:
    _ensure_csv(path)
    df = pd.read_csv(path)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False)


def load_open_signals(path: str) -> pd.DataFrame:
    _ensure_csv(path)
    df = pd.read_csv(path)
    if df.empty:
        return df
    df_open = df[df["status"] == "OPEN"].copy()
    return df_open


def update_signal_close(
    path: str,
    signal_id: str,
    *,
    result: str,
    hit_time_utc: Optional[str],
    exit_price: Optional[float],
    pnl_r: Optional[float],
    notes: str = "",
) -> None:
    _ensure_csv(path)
    df = pd.read_csv(path)
    if df.empty:
        return
    idx = df.index[df["signal_id"] == signal_id]
    if len(idx) == 0:
        return
    i = idx[0]
    df.at[i, "status"] = "CLOSED"
    df.at[i, "result"] = result
    df.at[i, "hit_time_utc"] = hit_time_utc or ""
    df.at[i, "exit_price"] = "" if exit_price is None else float(exit_price)
    df.at[i, "pnl_r"] = "" if pnl_r is None else float(pnl_r)
    df.at[i, "notes"] = notes
    df.to_csv(path, index=False)


def _parse_iso_utc(s: str) -> dt.datetime:
    # stored as ISO; ensure tz-aware UTC
    t = dt.datetime.fromisoformat(s)
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    return t.astimezone(dt.timezone.utc)


def evaluate_outcome_on_m5(
    m5_df: pd.DataFrame,
    *,
    direction: str,
    entry_time_utc_iso: str,
    entry: float,
    sl: float,
    tp1: float,
    tp2: float,
) -> Tuple[str, Optional[dt.datetime], Optional[float], Optional[float], str]:
    """
    Walk forward candle-by-candle from entry_time and find first hit:
      LONG: SL if low<=sl, TP if high>=tp
      SHORT: SL if high>=sl, TP if low<=tp

    Conservative assumption:
      if SL and TP are both crossed in same candle => count SL first.
    Returns: (result, hit_time, exit_price, pnl_r, notes)
    """
    if m5_df is None or m5_df.empty:
        return "NONE", None, None, None, "No data"

    entry_time = _parse_iso_utc(entry_time_utc_iso)

    df = m5_df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df[df["datetime"] >= entry_time].sort_values("datetime")

    if df.empty:
        return "NONE", None, None, None, "No candles after entry time"

    risk = abs(entry - sl)
    if risk <= 1e-9:
        return "NONE", None, None, None, "Bad SL risk distance"

    for _, row in df.iterrows():
        t = row["datetime"].to_pydatetime()
        hi = float(row["high"])
        lo = float(row["low"])

        if direction == "LONG":
            sl_hit = lo <= sl
            tp2_hit = hi >= tp2
            tp1_hit = hi >= tp1

            if sl_hit:
                pnl_r = -1.0
                return "SL", t, sl, pnl_r, "SL hit"
            if tp2_hit:
                pnl_r = (tp2 - entry) / risk
                return "TP2", t, tp2, pnl_r, "TP2 hit"
            if tp1_hit:
                pnl_r = (tp1 - entry) / risk
                return "TP1", t, tp1, pnl_r, "TP1 hit"

        else:  # SHORT
            sl_hit = hi >= sl
            tp2_hit = lo <= tp2
            tp1_hit = lo <= tp1

            if sl_hit:
                pnl_r = -1.0
                return "SL", t, sl, pnl_r, "SL hit"
            if tp2_hit:
                pnl_r = (entry - tp2) / risk
                return "TP2", t, tp2, pnl_r, "TP2 hit"
            if tp1_hit:
                pnl_r = (entry - tp1) / risk
                return "TP1", t, tp1, pnl_r, "TP1 hit"

    # nothing hit in available candles
    last_close = float(df["close"].iloc[-1])
    pnl_r = ((last_close - entry) / risk) if direction == "LONG" else ((entry - last_close) / risk)
    return "NONE", None, last_close, pnl_r, "No SL/TP hit in window"