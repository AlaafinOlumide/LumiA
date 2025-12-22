# data_logger.py
import csv
import os
import datetime as dt
from typing import Dict, Any, Optional, List


JOURNAL_HEADERS = [
    "signal_id",
    "symbol",
    "direction",
    "setup_type",
    "trend_source",
    "trend_bias",
    "entry_time_utc",
    "entry",
    "sl",
    "tp1",
    "tp2",
    "confidence",
    "entry_score",
    "status",            # OPEN / TP1 / TP2 / SL / EXPIRED / INVALIDATED
    "closed_time_utc",
    "exit_price",
    "pnl_r",
    "notes",
]


def _ensure_journal(path: str) -> None:
    if os.path.exists(path):
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(JOURNAL_HEADERS)


def append_signal_open(path: str, row: Dict[str, Any]) -> None:
    _ensure_journal(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=JOURNAL_HEADERS)
        out = {k: row.get(k, "") for k in JOURNAL_HEADERS}
        w.writerow(out)


def read_all(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        _ensure_journal(path)
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r)


def write_all(path: str, rows: List[Dict[str, str]]) -> None:
    _ensure_journal(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=JOURNAL_HEADERS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in JOURNAL_HEADERS})


def update_signal_close(
    path: str,
    signal_id: str,
    result: str,
    hit_time_utc: str,
    exit_price: float,
    pnl_r: float,
    notes: str = "",
) -> None:
    rows = read_all(path)
    changed = False
    for row in rows:
        if row.get("signal_id") == signal_id and row.get("status") == "OPEN":
            row["status"] = result
            row["closed_time_utc"] = hit_time_utc
            row["exit_price"] = f"{exit_price:.5f}"
            row["pnl_r"] = f"{pnl_r:.4f}"
            row["notes"] = notes
            changed = True
            break
    if changed:
        write_all(path, rows)


def get_open_signals(path: str) -> List[Dict[str, str]]:
    rows = read_all(path)
    return [r for r in rows if r.get("status") == "OPEN"]


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def safe_dt_iso_to_utc(s: str) -> Optional[dt.datetime]:
    if not s:
        return None
    try:
        t = dt.datetime.fromisoformat(s)
        if t.tzinfo is None:
            t = t.replace(tzinfo=dt.timezone.utc)
        return t.astimezone(dt.timezone.utc)
    except Exception:
        return None