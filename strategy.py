# strategy.py
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import pandas as pd

import indicators
from indicators import bollinger_bands, rsi, stochastic_oscillator, adx, bullish_engulfing, bearish_engulfing


@dataclass
class Signal:
    direction: str          # "LONG" or "SHORT"
    price: float
    time: dt.datetime
    reason: str
    extra: Dict[str, Any]


# -----------------------------------------------------------------------------
# Session helper (includes weekend control)
# -----------------------------------------------------------------------------
def is_within_sessions(
    now_utc: dt.datetime,
    session_1_start: int,
    session_1_end: int,
    session_2_start: Optional[int],
    session_2_end: Optional[int],
    trade_weekends: bool,
) -> bool:
    if not trade_weekends and now_utc.weekday() in (5, 6):
        return False

    hhmm = now_utc.hour * 100 + now_utc.minute
    in_s1 = session_1_start <= hhmm <= session_1_end

    if session_2_start is not None and session_2_end is not None:
        in_s2 = session_2_start <= hhmm <= session_2_end
    else:
        in_s2 = False

    return in_s1 or in_s2


# -----------------------------------------------------------------------------
# EMA + trend
# -----------------------------------------------------------------------------
def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def detect_trend_h1(h1_df: pd.DataFrame) -> Optional[str]:
    if h1_df is None or len(h1_df) < 60:
        return None

    close = h1_df["close"]
    high = h1_df["high"]
    low = h1_df["low"]

    ema20 = _ema(close, 20)
    ema50 = _ema(close, 50)
    rsi_series = rsi(close, period=14)
    adx_series, _, _ = adx(high, low, close, period=14)

    c = float(close.iloc[-1])
    e20 = float(ema20.iloc[-1])
    e50 = float(ema50.iloc[-1])
    r = float(rsi_series.iloc[-1])
    a = float(adx_series.iloc[-1]) if pd.notna(adx_series.iloc[-1]) else 0.0

    if a < 15:
        return None

    if c > e20 > e50 and r > 55:
        return "LONG"
    if c < e20 < e50 and r < 45:
        return "SHORT"
    return None


def detect_trend_m15_direction(m15_df: pd.DataFrame) -> Optional[str]:
    if m15_df is None or len(m15_df) < 60:
        return None

    close = m15_df["close"]
    high = m15_df["high"]
    low = m15_df["low"]

    ema20 = _ema(close, 20)
    ema50 = _ema(close, 50)
    rsi_series = rsi(close, period=14)
    adx_series, _, _ = adx(high, low, close, period=14)

    c = float(close.iloc[-1])
    e20 = float(ema20.iloc[-1])
    e50 = float(ema50.iloc[-1])
    r = float(rsi_series.iloc[-1])
    a = float(adx_series.iloc[-1]) if pd.notna(adx_series.iloc[-1]) else 0.0

    if a < 12:
        return None

    if c > e20 > e50 and r > 52:
        return "LONG"
    if c < e20 < e50 and r < 48:
        return "SHORT"

    return None


def confirm_trend_m15(m15_df: pd.DataFrame, trend: str) -> bool:
    if m15_df is None or len(m15_df) < 40:
        return False

    close = m15_df["close"]
    ema20 = _ema(close, 20)
    ema50 = _ema(close, 50)

    c = float(close.iloc[-1])
    e20 = float(ema20.iloc[-1])
    e50 = float(ema50.iloc[-1])

    if trend == "LONG":
        return e20 > e50 and c > e20
    if trend == "SHORT":
        return e20 < e50 and c < e20
    return False


# -----------------------------------------------------------------------------
# M15 structure filter
# -----------------------------------------------------------------------------
def m15_structure_ok(m15_df: pd.DataFrame, direction: str) -> Tuple[bool, Dict[str, Any]]:
    if m15_df is None or len(m15_df) < 60:
        return False, {"m15_structure": "insufficient_data"}

    close = m15_df["close"]
    open_ = m15_df["open"]
    ema20 = _ema(close, 20)
    ema50 = _ema(close, 50)

    c = float(close.iloc[-1])
    e20 = float(ema20.iloc[-1])
    e50 = float(ema50.iloc[-1])

    bull_engulf = bool(bullish_engulfing(open_, close).iloc[-1])
    bear_engulf = bool(bearish_engulfing(open_, close).iloc[-1])

    info = {
        "m15_close": c,
        "m15_ema20": e20,
        "m15_ema50": e50,
        "m15_bull_engulf": bull_engulf,
        "m15_bear_engulf": bear_engulf,
    }

    if direction == "LONG":
        ok = (e20 > e50) and (c >= e20) and (not bear_engulf)
        info["m15_structure"] = "ok" if ok else "fail_long"
        return ok, info

    if direction == "SHORT":
        ok = (e20 < e50) and (c <= e20) and (not bull_engulf)
        info["m15_structure"] = "ok" if ok else "fail_short"
        return ok, info

    return False, {"m15_structure": "bad_direction"}


# -----------------------------------------------------------------------------
# Regime
# -----------------------------------------------------------------------------
def market_regime(h1_df: pd.DataFrame) -> str:
    if h1_df is None or len(h1_df) < 40:
        return "Normal Volatility"

    atr_series = indicators.atr(h1_df["high"], h1_df["low"], h1_df["close"], period=14)
    atr = float(atr_series.iloc[-1]) if pd.notna(atr_series.iloc[-1]) else 0.0
    price = float(h1_df["close"].iloc[-1])
    atr_pct = (atr / price * 100.0) if price > 0 else 0.0

    if atr_pct >= 0.35:
        return "High Volatility"
    if atr_pct <= 0.20:
        return "Low Volatility / Compression"
    return "Normal Volatility"


# -----------------------------------------------------------------------------
# Score gate
# -----------------------------------------------------------------------------
def _score_gate(
    *,
    m15_ok: bool,
    in_pullback_zone: bool,
    rsi_reset_ok: bool,
    stoch_reset_ok: bool,
    rejection_ok: bool,
    adx_ok: bool,
    no_news: bool,
    weights: Dict[str, int],
) -> Tuple[int, Dict[str, Any]]:
    score = 0
    breakdown: Dict[str, Any] = {}

    def add(key: str, condition: bool, weight_key: str):
        nonlocal score
        pts = weights.get(weight_key, 0) if condition else 0
        score += pts
        breakdown[key] = {"ok": condition, "pts": pts}

    add("m15_structure_ok", m15_ok, "score_m15_structure")
    add("pullback_zone", in_pullback_zone, "score_pullback_zone")
    add("rsi_reset", rsi_reset_ok, "score_rsi_reset")
    add("stoch_reset", stoch_reset_ok, "score_stoch_reset")
    add("rejection", rejection_ok, "score_rejection")
    add("adx_ok", adx_ok, "score_adx_ok")
    add("no_high_news", no_news, "score_no_news")

    return int(score), breakdown


# -----------------------------------------------------------------------------
# Trigger on M5 (Pullback rewritten properly)
# -----------------------------------------------------------------------------
def trigger_signal_m5(
    m5_df: pd.DataFrame,
    trend_dir: str,
    m15_df: pd.DataFrame,
    h1_df: pd.DataFrame,
    high_news: bool,
    min_score: int,
    score_weights: Optional[Dict[str, int]] = None,
) -> Optional[Signal]:
    if m5_df is None or len(m5_df) < 120:
        return None
    if score_weights is None:
        score_weights = {}

    high = m5_df["high"]
    low = m5_df["low"]
    close = m5_df["close"]
    open_ = m5_df["open"]

    bb_upper, bb_mid, bb_lower = bollinger_bands(close, period=20, std_factor=2.0)
    rsi_series = rsi(close, period=14)
    stoch_k, stoch_d = stochastic_oscillator(high, low, close, k_period=14, d_period=3)
    adx_series, plus_di, minus_di = adx(high, low, close, period=14)

    bull_engulf_m5 = bullish_engulfing(open_, close)
    bear_engulf_m5 = bearish_engulfing(open_, close)

    c = float(close.iloc[-1])
    o = float(open_.iloc[-1])
    h = float(high.iloc[-1])
    l = float(low.iloc[-1])

    r = float(rsi_series.iloc[-1])
    k = float(stoch_k.iloc[-1])
    d = float(stoch_d.iloc[-1])
    k_prev = float(stoch_k.iloc[-2])
    d_prev = float(stoch_d.iloc[-2])

    bb_u = float(bb_upper.iloc[-1])
    bb_m = float(bb_mid.iloc[-1])
    bb_l = float(bb_lower.iloc[-1])

    adx_m5 = float(adx_series.iloc[-1]) if pd.notna(adx_series.iloc[-1]) else 0.0
    plus_di_m5 = float(plus_di.iloc[-1]) if pd.notna(plus_di.iloc[-1]) else 0.0
    minus_di_m5 = float(minus_di.iloc[-1]) if pd.notna(minus_di.iloc[-1]) else 0.0

    bar_time = m5_df["datetime"].iloc[-1]
    if isinstance(bar_time, pd.Timestamp):
        bar_time = bar_time.to_pydatetime()

    # M15 structure filter
    m15_ok, m15_info = m15_structure_ok(m15_df, trend_dir)

    extra_common = {
        "setup_type": None,
        "m5_rsi": r,
        "m5_stoch_k": k,
        "m5_stoch_d": d,
        "bb_upper": bb_u,
        "bb_mid": bb_m,
        "bb_lower": bb_l,
        "adx_m5": adx_m5,
        "plus_di_m5": plus_di_m5,
        "minus_di_m5": minus_di_m5,
        **m15_info,
    }

    def _rejection_bullish() -> bool:
        body = abs(c - o)
        rng = max(1e-9, (h - l))
        lower_wick = min(o, c) - l
        lower_wick_ratio = lower_wick / rng
        return (c > o) or bool(bull_engulf_m5.iloc[-1]) or (lower_wick_ratio >= 0.40 and body / rng <= 0.45)

    def _rejection_bearish() -> bool:
        body = abs(c - o)
        rng = max(1e-9, (h - l))
        upper_wick = h - max(o, c)
        upper_wick_ratio = upper_wick / rng
        return (c < o) or bool(bear_engulf_m5.iloc[-1]) or (upper_wick_ratio >= 0.40 and body / rng <= 0.45)

    def _pullback_long() -> Optional[Signal]:
        in_zone = (c <= bb_m) or (l <= bb_m)
        rsi_reset_ok = r <= 45.0
        stoch_reset_ok = (k_prev < 30.0 or d_prev < 30.0) and (k > d) and (k_prev <= d_prev)
        rejection_ok = _rejection_bullish()
        adx_ok = adx_m5 >= 18.0 and plus_di_m5 >= minus_di_m5
        no_news = not high_news

        score, breakdown = _score_gate(
            m15_ok=m15_ok,
            in_pullback_zone=in_zone,
            rsi_reset_ok=rsi_reset_ok,
            stoch_reset_ok=stoch_reset_ok,
            rejection_ok=rejection_ok,
            adx_ok=adx_ok,
            no_news=no_news,
            weights=score_weights,
        )

        if score < min_score:
            return None

        if in_zone and rsi_reset_ok and stoch_reset_ok and rejection_ok and m15_ok:
            extra = dict(extra_common)
            extra["setup_type"] = "PULLBACK_LONG"
            extra["entry_score"] = score
            extra["score_breakdown"] = breakdown
            reason = "Pullback LONG: BB mid touch/close + RSI reset <=45 + stoch <30 cross up + rejection + M15 structure."
            return Signal("LONG", c, bar_time, reason, extra)
        return None

    def _pullback_short() -> Optional[Signal]:
        in_zone = (c >= bb_m) or (h >= bb_m)
        rsi_reset_ok = r >= 55.0
        stoch_reset_ok = (k_prev > 70.0 or d_prev > 70.0) and (k < d) and (k_prev >= d_prev)
        rejection_ok = _rejection_bearish()
        adx_ok = adx_m5 >= 18.0 and minus_di_m5 >= plus_di_m5
        no_news = not high_news

        score, breakdown = _score_gate(
            m15_ok=m15_ok,
            in_pullback_zone=in_zone,
            rsi_reset_ok=rsi_reset_ok,
            stoch_reset_ok=stoch_reset_ok,
            rejection_ok=rejection_ok,
            adx_ok=adx_ok,
            no_news=no_news,
            weights=score_weights,
        )

        if score < min_score:
            return None

        if in_zone and rsi_reset_ok and stoch_reset_ok and rejection_ok and m15_ok:
            extra = dict(extra_common)
            extra["setup_type"] = "PULLBACK_SHORT"
            extra["entry_score"] = score
            extra["score_breakdown"] = breakdown
            reason = "Pullback SHORT: BB mid touch/close + RSI reset >=55 + stoch >70 cross down + rejection + M15 structure."
            return Signal("SHORT", c, bar_time, reason, extra)
        return None

    def _breakout_cont_long() -> Optional[Signal]:
        riding_upper = c >= bb_u * 0.998
        adx_ok = adx_m5 >= 22.0 and plus_di_m5 > minus_di_m5
        no_news = not high_news

        score, breakdown = _score_gate(
            m15_ok=m15_ok,
            in_pullback_zone=riding_upper,
            rsi_reset_ok=(r >= 52.0),
            stoch_reset_ok=(k >= 55.0),
            rejection_ok=True,
            adx_ok=adx_ok,
            no_news=no_news,
            weights=score_weights,
        )
        if score < min_score:
            return None

        if riding_upper and adx_ok and m15_ok and r >= 52.0:
            extra = dict(extra_common)
            extra["setup_type"] = "BREAKOUT_CONT_LONG"
            extra["entry_score"] = score
            extra["score_breakdown"] = breakdown
            reason = "Breakout continuation LONG: riding upper BB + ADX +DI dominance + M15 structure."
            return Signal("LONG", c, bar_time, reason, extra)
        return None

    def _breakout_cont_short() -> Optional[Signal]:
        riding_lower = c <= bb_l * 1.002
        adx_ok = adx_m5 >= 22.0 and minus_di_m5 > plus_di_m5
        no_news = not high_news

        score, breakdown = _score_gate(
            m15_ok=m15_ok,
            in_pullback_zone=riding_lower,
            rsi_reset_ok=(r <= 48.0),
            stoch_reset_ok=(k <= 45.0),
            rejection_ok=True,
            adx_ok=adx_ok,
            no_news=no_news,
            weights=score_weights,
        )
        if score < min_score:
            return None

        if riding_lower and adx_ok and m15_ok and r <= 48.0:
            extra = dict(extra_common)
            extra["setup_type"] = "BREAKOUT_CONT_SHORT"
            extra["entry_score"] = score
            extra["score_breakdown"] = breakdown
            reason = "Breakout continuation SHORT: riding lower BB + ADX -DI dominance + M15 structure."
            return Signal("SHORT", c, bar_time, reason, extra)
        return None

    if trend_dir == "LONG":
        return _pullback_long() or _breakout_cont_long()
    if trend_dir == "SHORT":
        return _pullback_short() or _breakout_cont_short()

    return None


# -----------------------------------------------------------------------------
# TP/SL + management (Drop 2 integrated)
# -----------------------------------------------------------------------------
def dynamic_tp_sl(signal: Signal, h1_df: pd.DataFrame, regime: str) -> None:
    if h1_df is None or len(h1_df) < 40:
        return

    atr_series = indicators.atr(h1_df["high"], h1_df["low"], h1_df["close"], period=14)
    atr = float(atr_series.iloc[-1]) if pd.notna(atr_series.iloc[-1]) else 0.0
    if atr <= 0:
        return

    entry = float(signal.price)

    # Regime-based targets (fixes “TP not hit” on compression days)
    if "Low Volatility" in regime:
        tp1_r, tp2_r = 0.8, 1.6
        sl_mult = 1.1
        trail_r = 0.6
    elif "High Volatility" in regime:
        tp1_r, tp2_r = 1.5, 2.5
        sl_mult = 1.4
        trail_r = 1.0
    else:
        tp1_r, tp2_r = 1.2, 2.0
        sl_mult = 1.2
        trail_r = 0.8

    sl_dist = atr * sl_mult

    if signal.direction == "LONG":
        sl = entry - sl_dist
        risk = entry - sl
        tp1 = entry + (tp1_r * risk)
        tp2 = entry + (tp2_r * risk)
    else:
        sl = entry + sl_dist
        risk = sl - entry
        tp1 = entry - (tp1_r * risk)
        tp2 = entry - (tp2_r * risk)

    signal.extra["atr_h1"] = atr
    signal.extra["sl"] = sl
    signal.extra["tp1"] = tp1
    signal.extra["tp2"] = tp2
    signal.extra["rr_tp1"] = f"{tp1_r:.2f}"
    signal.extra["rr_tp2"] = f"{tp2_r:.2f}"

    mgmt = []
    mgmt.append("Management:")
    mgmt.append(f"- TP1: take 50% off at {tp1_r:.2f}R, move SL to breakeven.")
    mgmt.append(f"- After TP1: trail using ~{trail_r:.2f}R behind price (or trail beyond M5 swing).")
    mgmt.append("- If strong close back through BB mid against position, consider early exit.")
    signal.extra["management_notes"] = "\n".join(mgmt)


# -----------------------------------------------------------------------------
# Confidence (uses entry_score)
# -----------------------------------------------------------------------------
def compute_confidence(
    trend_source: str,
    setup_type: str,
    adx_h1: float,
    adx_m5: float,
    high_news: bool,
    entry_score: int,
) -> int:
    base = 50

    base += 8 if trend_source == "H1" else 4
    if "PULLBACK" in setup_type:
        base += 6
    if "BREAKOUT_CONT" in setup_type:
        base += 4

    if adx_h1 >= 35:
        base += 10
    elif adx_h1 >= 25:
        base += 7
    elif adx_h1 >= 20:
        base += 4
    else:
        base -= 5

    if adx_m5 >= 30:
        base += 6
    elif adx_m5 >= 22:
        base += 3

    base += int(min(20, max(0, entry_score - 60) * 0.5))

    if high_news:
        base -= 15

    return max(1, min(99, int(base)))


def confidence_label(conf: int) -> str:
    if conf >= 80:
        return "High"
    if conf >= 65:
        return "Medium"
    return "Low"