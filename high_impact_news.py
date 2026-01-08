# high_impact_news.py
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

import requests


def _default_now_utc() -> datetime:
    return datetime.now(timezone.utc)


def is_high_impact_now(
    *,
    now_utc: Optional[datetime] = None,   # ✅ ADD THIS
    minutes_before: int = 10,
    minutes_after: int = 10,
    currencies: Tuple[str, ...] = ("USD", "EUR", "GBP"),
) -> bool:
    """
    Returns True if current time is within high impact news window.
    """

    if os.getenv("HIGH_IMPACT_NEWS", "1").strip().lower() in ("0", "false", "no"):
        return False

    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        return False

    now = (now_utc or _default_now_utc()).astimezone(timezone.utc)

    base_url = "https://financialmodelingprep.com/api/v3/economic_calendar"
    start = (now - timedelta(days=1)).date().isoformat()
    end = (now + timedelta(days=1)).date().isoformat()

    try:
        r = requests.get(
            base_url,
            params={"from": start, "to": end, "apikey": api_key},
            timeout=20,
        )
        r.raise_for_status()
        events = r.json() if isinstance(r.json(), list) else []
    except Exception:
        return False

    window_start = now - timedelta(minutes=minutes_before)
    window_end = now + timedelta(minutes=minutes_after)

    for ev in events:
        cur = (ev.get("country") or ev.get("currency") or "").upper()
        if cur and cur not in currencies:
            continue

        impact = str(ev.get("impact") or ev.get("importance") or "").lower()
        # keep simple: treat "high" as high impact
        if impact and "high" not in impact:
            continue

        dt_str = ev.get("date")
        if not dt_str:
            continue

        ev_dt = _parse_event_dt(dt_str)
        if not ev_dt:
            continue

        if window_start <= ev_dt <= window_end:
            return True

    return False


def _parse_event_dt(dt_str: str) -> Optional[datetime]:
    dt_str = dt_str.strip()
    fmts = (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    )
    for fmt in fmts:
        try:
            dt = datetime.strptime(dt_str, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            pass
    return None