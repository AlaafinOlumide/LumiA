# high_impact_news.py
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

import requests


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def is_high_impact_now(
    *,
    minutes_before: int = 10,
    minutes_after: int = 10,
    currencies: Tuple[str, ...] = ("USD", "EUR", "GBP"),
) -> bool:
    """
    Returns True if we are within a 'high impact news' window.

    Default: blocks trading 10 minutes before and after high-impact events
    on USD/EUR/GBP.

    Requires env:
      - FMP_API_KEY (Financial Modeling Prep economic calendar)
        OR set HIGH_IMPACT_NEWS=0 to disable this feature safely.
    """
    # Allow hard-disable without breaking bot
    if os.getenv("HIGH_IMPACT_NEWS", "1").strip() in ("0", "false", "False", "no", "NO"):
        return False

    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        # If no key, don't crash the bot. Just assume "no news block".
        # (If you prefer stricter safety: return True instead.)
        return False

    # FMP economic calendar endpoint
    # We'll request today's and tomorrow's events to catch late/early UTC overlaps.
    base_url = "https://financialmodelingprep.com/api/v3/economic_calendar"
    now = _now_utc()
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
        # Never crash trading because calendar fetch failed
        return False

    # Define what counts as "high impact"
    high_impact_keywords = ("high",)  # many calendars use "High" impact label

    window_start = now - timedelta(minutes=minutes_before)
    window_end = now + timedelta(minutes=minutes_after)

    for ev in events:
        # Currency filter
        cur = (ev.get("country") or ev.get("currency") or "").upper()
        if cur and cur not in currencies:
            continue

        # Impact filter (varies by API)
        impact = str(ev.get("impact") or ev.get("importance") or "").lower()
        if impact and not any(k in impact for k in high_impact_keywords):
            continue

        # Parse datetime (FMP often uses "date" as 'YYYY-MM-DD HH:MM:SS')
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
    """
    Best-effort parsing for economic calendar datetime strings.
    Assumes UTC if timezone missing.
    """
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