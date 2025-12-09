import datetime as dt
import logging
from typing import Any, Dict, List

import requests

logger = logging.getLogger(__name__)

# ForexFactory weekly calendar JSON feed.
# No API key required, but there are rate limits (approx 2 calls / 5 minutes),
# so we cache aggressively in this module.
FF_JSON_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

# For XAUUSD we mostly care about USD events, optionally global 'ALL' if present.
RELEVANT_CURRENCIES = {"USD", "ALL", ""}

# Cache structure
_CACHE: Dict[str, Any] = {
    "last_fetch": None,
    "events": [],
}

def _fetch_calendar_events() -> List[Dict[str, Any]]:
    # Fetch calendar events from ForexFactory JSON and cache the result.
    # Returns a list of event dicts. The JSON structure can change, so we try to be
    # defensive and support both list and dict-with-key formats.
    global _CACHE

    now = dt.datetime.utcnow()
    last_fetch = _CACHE.get("last_fetch")
    if last_fetch and (now - last_fetch).total_seconds() < 300:
        # Reuse cache if we fetched less than 5 minutes ago
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
        if "events" in data:
            events = data["events"]
        elif "calendar" in data:
            events = data["calendar"]
        else:
            # Fallback: treat dict values as events if they look like dicts
            possible = []
            for v in data.values():
                if isinstance(v, list):
                    possible.extend(v)
            events = possible
    else:
        events = []

    _CACHE["last_fetch"] = now
    _CACHE["events"] = events
    return events

def _event_matches_xauusd(ev: Dict[str, Any]) -> bool:
    # Return True if an event is relevant for XAUUSD (USD / ALL currencies, High impact).
    currency = str(ev.get("currency", "") or ev.get("country", "") or "").upper()
    if currency not in RELEVANT_CURRENCIES:
        return False

    impact = str(ev.get("impact", "")).lower()
    # ForexFactory typically uses 'High' (and sometimes with brackets).
    return "high" in impact

def _event_time(ev: Dict[str, Any]) -> dt.datetime | None:
    # Extract event time as UTC datetime from different possible fields.
    ts = ev.get("timestamp") or ev.get("time") or ev.get("date")
    if ts is None:
        return None

    # ForexFactory JSON normally exposes a unix timestamp (seconds).
    try:
        if isinstance(ts, (int, float)):
            return dt.datetime.utcfromtimestamp(int(ts))
        # Strings: try parse as int first
        return dt.datetime.utcfromtimestamp(int(str(ts)))
    except Exception:
        # As a last resort, try ISO parse
        try:
            return dt.datetime.fromisoformat(str(ts))
        except Exception:
            return None

def has_high_impact_news_near(symbol: str, now_utc: dt.datetime, window_minutes: int = 60) -> bool:
    # Return True if there is a nearby high-impact news event that could move XAUUSD.
    #
    # - Uses ForexFactory weekly JSON feed (no key needed).
    # - Filters by:
    #   - High impact events only
    #   - Relevant currencies (USD / ALL)
    # - Considers events within +/- `window_minutes` of `now_utc`.
    #
    # This function is intentionally conservative: if the API call fails or parsing
    # looks weird, it returns False so that trading is not blocked by a network error.
    try:
        events = _fetch_calendar_events()
    except Exception as e:
        logger.warning("Error while fetching calendar events: %s", e)
        return False

    if not events:
        return False

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
