import datetime as dt
import logging
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# ForexFactory weekly calendar JSON feed.
# No API key required, but there are rate limits, so we cache aggressively.
FF_JSON_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

# For XAUUSD we mostly care about USD events, optionally global 'ALL'.
RELEVANT_CURRENCIES = {"USD", "ALL", ""}

# Cache structure
_CACHE: Dict[str, Any] = {
    "last_fetch": None,
    "events": [],
}


def _fetch_calendar_events(cache_seconds: int = 300) -> List[Dict[str, Any]]:
    """
    Fetch calendar events from ForexFactory JSON and cache the result.
    Returns a list of event dicts.
    """
    global _CACHE

    now = dt.datetime.utcnow()
    last_fetch = _CACHE.get("last_fetch")

    if isinstance(last_fetch, dt.datetime) and (now - last_fetch).total_seconds() < cache_seconds:
        return _CACHE.get("events", [])

    try:
        resp = requests.get(FF_JSON_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("Failed to fetch ForexFactory calendar JSON: %s", e)
        return _CACHE.get("events", [])

    # The JSON is typically a list of events. If it's a dict, try common keys.
    if isinstance(data, list):
        events = data
    elif isinstance(data, dict):
        if "events" in data and isinstance(data["events"], list):
            events = data["events"]
        elif "calendar" in data and isinstance(data["calendar"], list):
            events = data["calendar"]
        else:
            possible: List[Dict[str, Any]] = []
            for v in data.values():
                if isinstance(v, list):
                    possible.extend([x for x in v if isinstance(x, dict)])
            events = possible
    else:
        events = []

    _CACHE["last_fetch"] = now
    _CACHE["events"] = events
    return events


def _event_matches_xauusd(ev: Dict[str, Any]) -> bool:
    """
    Return True if an event is relevant for XAUUSD (USD / ALL currencies, High impact).
    """
    currency = str(ev.get("currency", "") or ev.get("country", "") or "").upper().strip()
    if currency not in RELEVANT_CURRENCIES:
        return False

    impact = str(ev.get("impact", "")).lower()
    return "high" in impact


def _parse_epoch(ts_int: int) -> Optional[dt.datetime]:
    """
    Handle epoch seconds vs epoch milliseconds.
    """
    try:
        # milliseconds are usually 13 digits
        if ts_int > 10_000_000_000:  # > ~2286-11-20 in seconds => likely ms
            ts_int = ts_int // 1000
        return dt.datetime.utcfromtimestamp(ts_int)
    except Exception:
        return None


def _event_time(ev: Dict[str, Any]) -> Optional[dt.datetime]:
    """
    Extract event time as UTC datetime from different possible fields.
    ForexFactory feed may contain:
      - "timestamp" (seconds or milliseconds)
      - "time" / "date" as epoch-like string OR ISO string
    """
    ts = ev.get("timestamp") or ev.get("time") or ev.get("date")
    if ts is None:
        return None

    # Numeric epoch
    if isinstance(ts, (int, float)):
        return _parse_epoch(int(ts))

    s = str(ts).strip()
    if not s:
        return None

    # Epoch-like string
    if s.isdigit():
        return _parse_epoch(int(s))

    # ISO-like string (try robust parsing)
    try:
        # If it's ISO without timezone, treat as UTC
        t = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
        if t.tzinfo is None:
            return t.replace(tzinfo=dt.timezone.utc).astimezone(dt.timezone.utc).replace(tzinfo=None)
        return t.astimezone(dt.timezone.utc).replace(tzinfo=None)
    except Exception:
        return None


def has_high_impact_news_near(symbol: str, now_utc: dt.datetime, window_minutes: int = 60) -> bool:
    """
    Return True if there is a nearby high-impact news event that could move XAUUSD.

    - Uses ForexFactory weekly JSON feed (no key needed).
    - Filters by:
      - High impact events only
      - Relevant currencies (USD / ALL)
    - Considers events within +/- `window_minutes` of `now_utc`.

    If the API call fails or parsing looks weird, returns False so that
    trading is not blocked by a network error.
    """
    try:
        events = _fetch_calendar_events()
    except Exception as e:
        logger.warning("Error while fetching calendar events: %s", e)
        return False

    if not events:
        return False

    # normalize now_utc
    if now_utc.tzinfo is not None:
        now_utc = now_utc.astimezone(dt.timezone.utc).replace(tzinfo=None)

    now_ts = int(now_utc.timestamp())
    window = window_minutes * 60

    for ev in events:
        if not isinstance(ev, dict):
            continue
        if not _event_matches_xauusd(ev):
            continue

        t = _event_time(ev)
        if t is None:
            continue

        if abs(int(t.timestamp()) - now_ts) <= window:
            return True

    return False


# --------------------------------------------------------------------
# Compatibility wrapper: MAIN IMPORTS THIS NAME OFTEN
# --------------------------------------------------------------------
def has_high_impact_news_nearby(now_utc: dt.datetime, window_minutes: int = 60, symbol: str = "XAUUSD") -> bool:
    """
    Backwards-compatible wrapper so main.py can do:
      from high_impact_news import has_high_impact_news_nearby

    Internally uses has_high_impact_news_near(...).
    """
    return has_high_impact_news_near(symbol=symbol, now_utc=now_utc, window_minutes=window_minutes)
