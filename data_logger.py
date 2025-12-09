import csv
import os
from typing import Dict, Any
from datetime import datetime

LOG_FILE = "signals_log.csv"

FIELDNAMES = [
    "timestamp_utc",
    "symbol",
    "direction",
    "price",
    "reason",
    "trend_h1",
    "session_window",
    "m5_rsi",
    "m5_stoch_k",
    "m5_stoch_d",
    "bb_upper",
    "bb_mid",
    "bb_lower",
    "adx_m5",
    "plus_di_m5",
    "minus_di_m5",
    "high_impact_news",
]

def log_signal(row: Dict[str, Any]) -> None:
    # Append a signal row to CSV. Extra keys beyond FIELDNAMES are ignored.
    exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        clean_row = {k: row.get(k, "") for k in FIELDNAMES}
        if not clean_row.get("timestamp_utc"):
            clean_row["timestamp_utc"] = datetime.utcnow().isoformat()
        writer.writerow(clean_row)
