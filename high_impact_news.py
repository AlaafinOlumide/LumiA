# high_impact_news.py
# (kept your implementation; only export the name main.py uses)
import datetime as dt
import logging
from typing import Any, Dict, List
import requests

logger = logging.getLogger(__name__)
FF_JSON_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
RELEVANT_CURRENCIES = {"USD", "ALL", ""}

_CACHE: Dict[str, Any] = {"last_fetch": None, "events": []}

def _fetch_calendar_events() -> List[Dict[str, Any]]:
    global _CACHE
    now = dt.datetime.utcnow()
    last_fetch = _CACHE.get("last_fetch")
    if last_fetch and (now - last_fetch).total_seconds() < 300:
        return _CACHE.get("events", [])

    try:
        resp = requests.get(FF_JSON_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("Failed to fetch ForexFactory calendar JSON: %s", e)
        return _CACHE.get("events", [])

    if isinstance(data, list):
        events = data
    elif isinstance(data, dict):
        events = data.get("events") or data.get("calendar") or []
        if not events:
            possible: List[Dict[str, Any]] = []
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
    currency = str(ev.get("currency", "") or ev.get("country", "") or "").upper()
    if currency not in RELEVANT_CURRENCIES:
        return False
    impact = str(ev.get("impact", "")).lower()
    return "high" in impact

def _event_time(ev: Dict[str, Any]):
    ts = ev.get("timestamp") or ev.get("time") or ev.get("date")
    if ts is None:
        return None
    try:
        if isinstance(ts, (int, float)):
            return dt.datetime.utcfromtimestamp(int(ts))
        return dt.datetime.utcfromtimestamp(int(str(ts)))
    except Exception:
        try:
            return dt.datetime.fromisoformat(str(ts))
        except Exception:
            return None

def has_high_impact_news_near(symbol: str, now_utc: dt.datetime, window_minutes: int = 60) -> bool:
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
