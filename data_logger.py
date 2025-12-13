# data_logger.py
from __future__ import annotations

import csv
import os
import datetime as dt
from dataclasses import asdict
from typing import Dict, Optional, Tuple
from collections import defaultdict, deque

from trade_manager import ActiveTrade


class TradeJournal:
    """
    Lightweight CSV journal:
      - Append on OPEN
      - Update on CLOSE/TP/SL/INVALIDATED

    Also keeps rolling performance stats per (setup_type, trend_source)
    so we can adapt confidence.
    """

    def __init__(self, csv_path: str = "trades.csv", rolling_n: int = 60):
        self.csv_path = csv_path
        self.rolling_n = rolling_n

        self._rolling: Dict[Tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=self.rolling_n))
        self._ensure_header()

        # Rebuild rolling stats from existing CSV if present
        self._load_existing_into_rolling()

    def _ensure_header(self):
        if os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0:
            return

        header = [
            "trade_id", "opened_time", "direction", "setup_type", "entry", "sl", "tp1", "tp2",
            "confidence", "trend_source", "status",
            "exit_time", "exit_price", "result_r",
            "invalidation_deadline", "invalidated_reason",
        ]
        with open(self.csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)

    def _load_existing_into_rolling(self):
        if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
            return

        with open(self.csv_path, "r", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                status = row.get("status", "")
                setup = row.get("setup_type", "GENERIC")
                src = row.get("trend_source", "H1")
                rr = row.get("result_r")

                if status in ("TP1", "TP2", "SL", "INVALIDATED", "CLOSED") and rr not in (None, "", "None"):
                    try:
                        rr_val = float(rr)
                    except Exception:
                        continue
                    self._rolling[(setup, src)].append(rr_val)

    def append_open(self, trade: ActiveTrade) -> None:
        row = self._trade_to_row(trade)
        self._append_row(row)

    def update_trade(self, trade: ActiveTrade) -> None:
        """
        Rewrite CSV by updating the trade_id row.
        (Ok for small journals; later we can switch to SQLite.)
        """
        if not os.path.exists(self.csv_path):
            self.append_open(trade)
            return

        updated_rows = []
        found = False

        with open(self.csv_path, "r", newline="") as f:
            r = csv.DictReader(f)
            fieldnames = r.fieldnames or []
            for row in r:
                if row.get("trade_id") == trade.trade_id:
                    found = True
                    updated_rows.append(self._trade_to_row(trade))
                else:
                    updated_rows.append(row)

        if not found:
            updated_rows.append(self._trade_to_row(trade))

        with open(self.csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=updated_rows[0].keys())
            w.writeheader()
            w.writerows(updated_rows)

        # update rolling
        if trade.result_r is not None and trade.status in ("TP1", "TP2", "SL", "INVALIDATED", "CLOSED"):
            self._rolling[(trade.setup_type, trade.trend_source)].append(float(trade.result_r))

    def adaptive_confidence_adjustment(self, setup_type: str, trend_source: str) -> int:
        """
        Returns an adjustment (-10 .. +10) based on rolling R outcomes.
        Simple and stable:
          - If avg R is good -> + adjustment
          - If avg R is bad -> - adjustment
        """
        data = list(self._rolling.get((setup_type, trend_source), []))
        if len(data) < 10:
            return 0  # not enough evidence

        avg_r = sum(data) / len(data)

        # map avg_r to adjustment
        if avg_r >= 0.60:
            return +10
        if avg_r >= 0.25:
            return +6
        if avg_r >= 0.05:
            return +2
        if avg_r <= -0.60:
            return -10
        if avg_r <= -0.25:
            return -6
        if avg_r <= -0.05:
            return -2
        return 0

    def _append_row(self, row: Dict[str, str]) -> None:
        with open(self.csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=row.keys())
            w.writerow(row)

    def _trade_to_row(self, trade: ActiveTrade) -> Dict[str, str]:
        def iso(v):
            if v is None:
                return ""
            if isinstance(v, dt.datetime):
                return v.isoformat()
            return str(v)

        return {
            "trade_id": trade.trade_id,
            "opened_time": iso(trade.opened_time),
            "direction": trade.direction,
            "setup_type": trade.setup_type,
            "entry": f"{trade.entry:.5f}",
            "sl": f"{trade.sl:.5f}",
            "tp1": f"{trade.tp1:.5f}",
            "tp2": f"{trade.tp2:.5f}",
            "confidence": str(trade.confidence),
            "trend_source": trade.trend_source,
            "status": trade.status,
            "exit_time": iso(trade.exit_time),
            "exit_price": "" if trade.exit_price is None else f"{trade.exit_price:.5f}",
            "result_r": "" if trade.result_r is None else f"{trade.result_r:.4f}",
            "invalidation_deadline": iso(trade.invalidation_deadline),
            "invalidated_reason": trade.invalidated_reason or "",
        }
