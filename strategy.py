# strategy.py
from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
from typing import Optional, Literal, Dict, Any

import numpy as np
import pandas as pd


Direction = Literal["BUY", "SELL"]
SessionName = Literal["ASIA", "LONDON_NY"]


@dataclass
class Signal:
    direction: Direction
    entry: float
    sl: float
    tp1: float
    tp2: float
    extra: Dict[str, Any]


# -------------------------
# Sessions
# -------------------------
def active_session(
    now_utc: dt.datetime,
    enable_asia: bool,
    trade_weekends: bool,
    asia_start_hour_utc: int = 0,
    asia_end_hour_utc: int = 2,
    london_start_hour_utc: int = 7,
    london_end_hour_utc: int = 20,
) -> Optional[str]:
    """
    Returns "ASIA", "LONDON_NY", or None.
    Times are UTC hours.
    """
    if not trade_weekends and now_utc.weekday() in (5, 6):  # Sat=5 Sun=6
        return None

    h = now_utc.hour

    if enable_asia and asia_start_hour_utc <= h < asia_end_hour_utc:
        return "ASIA"

    if london_start_hour_utc <= h < london_end_hour_utc:
        return "LONDON_NY"

    return None


# -------------------------
# Small indicator helpers (pure pandas)
# -------------------------
def _ema(series: pd.Series, span: int) -> pd.Series:
    series = series.astype(float)
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI with stable numeric dtype + no deprecated fillna(method=...).
    """
    close = close.astype(float)
    delta = close.diff()

    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    # avoid object dtype by using np.nan (not pd.NA)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # backfill early NaNs
    return rsi.bfill()


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = _true_range(high, low, close)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


# -------------------------
# Trend detection (H1)
# -------------------------
def detect_trend_h1(h1_df: pd.DataFrame) -> Optional[str]:
    """
    Simple and stable H1 trend filter:
    - BUY trend if EMA50 > EMA200 and last close > EMA50 and EMA50 slope up
    - SELL trend if EMA50 < EMA200 and last close < EMA50 and EMA50 slope down
    else None
    """
    if h1_df is None or len(h1_df) < 220:
        return None

    close = h1_df["close"].astype(float)

    ema50 = _ema(close, 50)
    ema200 = _ema(close, 200)

    last_close = float(close.iloc[-1])
    last_ema50 = float(ema50.iloc[-1])
    last_ema200 = float(ema200.iloc[-1])

    # slope check (avoid flat/noisy)
    ema50_slope = float(ema50.iloc[-1] - ema50.iloc[-6])  # ~5 hours slope

    if last_ema50 > last_ema200 and last_close > last_ema50 and ema50_slope > 0:
        return "BULL"
    if last_ema50 < last_ema200 and last_close < last_ema50 and ema50_slope < 0:
        return "BEAR"

    return None


# -------------------------
# Entry trigger (M5)
# -------------------------
def trigger_signal_m5(
    m5_df: pd.DataFrame,
    m15_df: pd.DataFrame,
    h1_df: pd.DataFrame,  # kept for signature compatibility (not used here)
    trend_dir: str,
    high_news: bool,
    min_score: int,
    session: str,
    asia_extra_buffer: int,
    tp1_rr: float,
    tp2_rr: float,
    asia_tp1_rr: float,
    sl_atr_mult: float,
) -> Optional[Signal]:
    """
    Scoring model (0–90) to support your config:
      - min_score = 65
      - asia_extra_buffer = 2

    Score weights:
      - M5 alignment: 20
      - M15 alignment: 20
      - Price vs EMA20: 20
      - RSI band: 20
      - EMA50 slope support: 10
      Max score: 90
    """
    if m5_df is None or len(m5_df) < 200:
        return None
    if m15_df is None or len(m15_df) < 120:
        return None
    if high_news:
        return None

    # Session threshold
    required_score = int(min_score + (asia_extra_buffer if session == "ASIA" else 0))

    # Prepare series
    m5 = m5_df.copy()
    m15 = m15_df.copy()

    for df in (m5, m15):
        for c in ("open", "high", "low", "close"):
            df[c] = df[c].astype(float)

    m5_close = m5["close"]
    m15_close = m15["close"]

    # Indicators
    m5_ema20 = _ema(m5_close, 20)
    m5_ema50 = _ema(m5_close, 50)
    m5_rsi14 = _rsi(m5_close, 14)
    m5_atr14 = _atr(m5["high"], m5["low"], m5_close, 14)

    m15_ema20 = _ema(m15_close, 20)
    m15_ema50 = _ema(m15_close, 50)

    last_price = float(m5_close.iloc[-1])
    last_atr = float(m5_atr14.iloc[-1]) if pd.notna(m5_atr14.iloc[-1]) else 0.0
    if last_atr <= 0:
        return None

    # -------------------------
    # FIX: normalize direction input
    # -------------------------
    td = (trend_dir or "").upper().strip()
    if td in ("BULL", "LONG", "BUY"):
        direction: Direction = "BUY"
    elif td in ("BEAR", "SHORT", "SELL"):
        direction = "SELL"
    else:
        # unknown trend input -> skip
        return None

    # -------------------------
    # Scoring (0–90)
    # -------------------------
    score = 0

    cond_m5_align = False
    cond_m15_align = False
    cond_price_momo = False
    cond_rsi_ok = False
    cond_slope_ok = False

    # 1) M5 alignment (20)
    if direction == "BUY" and m5_ema20.iloc[-1] > m5_ema50.iloc[-1]:
        score += 20
        cond_m5_align = True
    if direction == "SELL" and m5_ema20.iloc[-1] < m5_ema50.iloc[-1]:
        score += 20
        cond_m5_align = True

    # 2) M15 alignment (20)
    if direction == "BUY" and m15_ema20.iloc[-1] > m15_ema50.iloc[-1]:
        score += 20
        cond_m15_align = True
    if direction == "SELL" and m15_ema20.iloc[-1] < m15_ema50.iloc[-1]:
        score += 20
        cond_m15_align = True

    # 3) Price position vs M5 EMA20 (20)
    if direction == "BUY" and last_price > float(m5_ema20.iloc[-1]):
        score += 20
        cond_price_momo = True
    if direction == "SELL" and last_price < float(m5_ema20.iloc[-1]):
        score += 20
        cond_price_momo = True

    # 4) RSI sanity (20)
    last_rsi = float(m5_rsi14.iloc[-1]) if pd.notna(m5_rsi14.iloc[-1]) else 50.0
    if direction == "BUY" and 45.0 <= last_rsi <= 70.0:
        score += 20
        cond_rsi_ok = True
    if direction == "SELL" and 30.0 <= last_rsi <= 55.0:
        score += 20
        cond_rsi_ok = True

    # 5) EMA50 slope supports direction (10)
    ema50_slope = float(m5_ema50.iloc[-1] - m5_ema50.iloc[-6])
    if direction == "BUY" and ema50_slope > 0:
        score += 10
        cond_slope_ok = True
    if direction == "SELL" and ema50_slope < 0:
        score += 10
        cond_slope_ok = True

    if score < required_score:
        # keep rich diagnostics so logs can explain “No signal”
        return None

    # -------------------------
    # SL/TP
    # -------------------------
    sl_dist = float(sl_atr_mult) * last_atr

    if direction == "BUY":
        sl = last_price - sl_dist
        rr1 = asia_tp1_rr if session == "ASIA" else tp1_rr
        tp1 = last_price + rr1 * sl_dist
        tp2 = last_price + tp2_rr * sl_dist
    else:
        sl = last_price + sl_dist
        rr1 = asia_tp1_rr if session == "ASIA" else tp1_rr
        tp1 = last_price - rr1 * sl_dist
        tp2 = last_price - tp2_rr * sl_dist

    extra = {
        "entry_score": int(score),
        "required_score": int(required_score),
        "tp_mode": ("ASIA_TIGHT" if session == "ASIA" else "NORMAL"),
        "trend_input": td,
        "direction": direction,
        "rsi": float(round(last_rsi, 2)),
        "atr": float(round(last_atr, 4)),
        "ema50_slope": float(round(ema50_slope, 6)),
        # condition breakdown (super useful for debugging)
        "cond_m5_align": cond_m5_align,
        "cond_m15_align": cond_m15_align,
        "cond_price_momo": cond_price_momo,
        "cond_rsi_ok": cond_rsi_ok,
        "cond_slope_ok": cond_slope_ok,
    }

    return Signal(
        direction=direction,
        entry=float(round(last_price, 2)),
        sl=float(round(sl, 2)),
        tp1=float(round(tp1, 2)),
        tp2=float(round(tp2, 2)),
        extra=extra,
    )