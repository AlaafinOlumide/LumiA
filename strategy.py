# strategy.py
import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from indicators import (
    bollinger_bands,
    rsi,
    stochastic_oscillator,
    adx,
    atr,
    bullish_engulfing,
    bearish_engulfing,
)

# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------

@dataclass
class Signal:
    direction: str          # "LONG" or "SHORT"
    price: float
    time: dt.datetime
    reason: str
    extra: Dict[str, Any]


# -----------------------------------------------------------------------------
# Sessions / trading window
# -----------------------------------------------------------------------------

def is_within_sessions(
    *,
    now_utc: dt.datetime,
    session_1_start: int,
    session_1_end: int,
    session_2_start: Optional[int] = None,
    session_2_end: Optional[int] = None,
    trade_weekends: bool = False,
) -> bool:
    """
    Time window filter (UTC). Prevents weekends unless trade_weekends=True.

    session_* are HHMM integers. Example: 700 -> 07:00, 2000 -> 20:00
    """
    # Weekend block
    if not trade_weekends:
        # Monday=0 ... Friday=5 is wrong; Python: Monday=0 .. Sunday=6
        wd = now_utc.weekday()
        if wd >= 5:  # 5=Saturday, 6=Sunday
            return False

    hhmm = now_utc.hour * 100 + now_utc.minute

    in_s1 = session_1_start <= hhmm <= session_1_end

    in_s2 = False
    if session_2_start is not None and session_2_end is not None:
        in_s2 = session_2_start <= hhmm <= session_2_end

    return in_s1 or in_s2


# -----------------------------------------------------------------------------
# Trend detection H1 / direction fallback M15
# -----------------------------------------------------------------------------

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def detect_trend_h1(h1_df: pd.DataFrame) -> Optional[str]:
    """
    Higher-timeframe trend using EMA(20/50), RSI(14), and ADX(14).
    Returns "LONG", "SHORT", or None (ranging/unclear).
    """
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

    # Weak ADX => don't force a direction
    if a < 15:
        return None

    if c > e20 > e50 and r > 55:
        return "LONG"
    if c < e20 < e50 and r < 45:
        return "SHORT"

    return None


def detect_trend_m15_direction(m15_df: pd.DataFrame) -> Optional[str]:
    """
    Fallback direction using M15 (looser thresholds).
    """
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


def _m15_structure_bias(m15_df: pd.DataFrame) -> str:
    """
    Simple structure read on M15:
      - BULL: EMA20>EMA50, last close > EMA20
      - BEAR: EMA20<EMA50, last close < EMA20
      - otherwise RANGE
    """
    if m15_df is None or len(m15_df) < 50:
        return "RANGE"

    close = m15_df["close"]
    ema20 = _ema(close, 20)
    ema50 = _ema(close, 50)

    c = float(close.iloc[-1])
    e20 = float(ema20.iloc[-1])
    e50 = float(ema50.iloc[-1])

    if e20 > e50 and c > e20:
        return "BULL"
    if e20 < e50 and c < e20:
        return "BEAR"
    return "RANGE"


def confirm_trend_m15(m15_df: pd.DataFrame, trend: str) -> bool:
    """
    Confirm H1 trend using M15 structure (EMA alignment + price location).
    """
    bias = _m15_structure_bias(m15_df)
    if trend == "LONG":
        return bias == "BULL"
    if trend == "SHORT":
        return bias == "BEAR"
    return False


# -----------------------------------------------------------------------------
# Market regime + adaptive TP/SL
# -----------------------------------------------------------------------------

def market_regime(h1_df: pd.DataFrame) -> str:
    """
    Label regime from ATR and ADX:
      - High Volatility: ATR in top tercile or ADX>=25
      - Low Volatility:  ATR in bottom tercile and ADX<15
      - otherwise: Normal
    """
    if h1_df is None or len(h1_df) < 80:
        return "Normal"

    high = h1_df["high"]
    low = h1_df["low"]
    close = h1_df["close"]
    atr14 = atr(high, low, close, period=14)

    adx_series, _, _ = adx(high, low, close, period=14)
    a = float(adx_series.iloc[-1]) if pd.notna(adx_series.iloc[-1]) else 0.0

    # simple distribution split
    last_atr = float(atr14.iloc[-1]) if pd.notna(atr14.iloc[-1]) else 0.0
    if atr14.dropna().empty:
        return "Normal"

    q33 = float(atr14.quantile(0.33))
    q66 = float(atr14.quantile(0.66))

    if last_atr >= q66 or a >= 25:
        return "High Volatility"
    if last_atr <= q33 and a < 15:
        return "Low Volatility"
    return "Normal"


def dynamic_tp_sl(sig: Signal, h1_df: pd.DataFrame, regime: str) -> None:
    """
    Set SL/TP from H1 ATR. Uses R multiples, stored in sig.extra["sl"/"tp1"/"tp2"].
    """
    high = h1_df["high"]
    low = h1_df["low"]
    close = h1_df["close"]
    atr14 = atr(high, low, close, period=14)
    curr_atr = float(atr14.iloc[-1]) if pd.notna(atr14.iloc[-1]) else 10.0

    # ATR multipliers by regime (tighter in low vol, wider in high)
    if regime == "Low Volatility":
        sl_mult = 0.6
    elif regime == "High Volatility":
        sl_mult = 1.2
    else:
        sl_mult = 0.9

    sl_dist = max(curr_atr * sl_mult, 1e-4)  # safety floor

    if sig.direction.upper() == "LONG":
        sl = sig.price - sl_dist
        tp1 = sig.price + 1.2 * (sig.price - sl)  # 1.2R
        tp2 = sig.price + 2.0 * (sig.price - sl)  # 2.0R
    else:
        sl = sig.price + sl_dist
        tp1 = sig.price - 1.2 * (sl - sig.price)
        tp2 = sig.price - 2.0 * (sl - sig.price)

    sig.extra["sl"] = float(sl)
    sig.extra["tp1"] = float(tp1)
    sig.extra["tp2"] = float(tp2)
    sig.extra["atr_h1"] = curr_atr
    sig.extra["regime"] = regime


# -----------------------------------------------------------------------------
# Confidence scoring
# -----------------------------------------------------------------------------

def compute_confidence(
    *,
    trend_source: str,
    setup_type: str,
    adx_h1: float,
    adx_m5: float,
    high_news: bool,
    entry_score: int,
) -> int:
    """
    Convert entry_score + context to a final 0..100 confidence.
    """
    score = entry_score

    # Slight boost for H1-driven trend
    if trend_source == "H1":
        score += 5

    # ADX(H1) adds confidence
    if adx_h1 >= 30:
        score += 5
    elif adx_h1 >= 20:
        score += 3

    # ADX(M5) small contribution
    if adx_m5 >= 25:
        score += 3
    elif adx_m5 >= 18:
        score += 1

    # High impact news reduces confidence
    if high_news:
        score -= 8

    # Setup-type small stylistic nudges
    if setup_type.startswith("BREAKOUT_CONT"):
        score += 2

    return int(max(0, min(100, score)))


def confidence_label(score: int) -> str:
    if score >= 76:
        return "High"
    if score >= 56:
        return "Medium"
    return "Low"


# -----------------------------------------------------------------------------
# Trigger engine on M5
#   - Rewritten Pullback logic
#   - Fresh Breakout
#   - Breakout Continuation
#   - Score-based entry gate
# -----------------------------------------------------------------------------

def _score_gate_long(
    *,
    c: float,
    o: float,
    bb_m: float,
    bb_l: float,
    rsi_val: float,
    stoch_k: float,
    stoch_d: float,
    stoch_k_prev: float,
    stoch_d_prev: float,
    m5_close: pd.Series,
    adx_m5: float,
    m15_bias: str,
    trend_dir: str,
    candle_bull: bool,
) -> Tuple[int, str]:
    """
    Build a score for LONG pullback/continuation. Returns (score, reason_snippet).
    """
    score = 0
    reasons = []

    # 1) Trend alignment (M15 structure + overall trend)
    if trend_dir == "LONG":
        score += 12
    if m15_bias == "BULL":
        score += 10
    else:
        reasons.append("M15 not bullish")

    # 2) Location: near mid/lower band
    if bb_l * 0.995 <= c <= bb_m * 1.005:
        score += 12
        reasons.append("Near BB mid/lower")
    else:
        reasons.append("Not near BB mid/lower")

    # 3) Momentum turn: stoch cross up
    if (stoch_k > stoch_d) and (stoch_k_prev <= stoch_d_prev):
        score += 10
        reasons.append("Stoch cross up")
    elif stoch_k > stoch_d:
        score += 5
        reasons.append("Stoch > D")

    # 4) RSI healthy
    if 38 <= rsi_val <= 62:
        score += 8
    elif rsi_val > 62:
        reasons.append("RSI hot")
    else:
        reasons.append("RSI cold")

    # 5) Candle confirmation
    if candle_bull or c > o:
        score += 8
        reasons.append("Bullish candle/engulf")

    # 6) M5 micro-trend (EMA20>EMA50)
    ema20 = _ema(m5_close, 20)
    ema50 = _ema(m5_close, 50)
    if float(ema20.iloc[-1]) > float(ema50.iloc[-1]):
        score += 8
        reasons.append("M5 EMA20>EMA50")

    # 7) ADX support on M5
    if adx_m5 >= 23:
        score += 6
    elif adx_m5 >= 18:
        score += 3

    return score, "; ".join(reasons)


def _score_gate_short(
    *,
    c: float,
    o: float,
    bb_m: float,
    bb_u: float,
    rsi_val: float,
    stoch_k: float,
    stoch_d: float,
    stoch_k_prev: float,
    stoch_d_prev: float,
    m5_close: pd.Series,
    adx_m5: float,
    m15_bias: str,
    trend_dir: str,
    candle_bear: bool,
) -> Tuple[int, str]:
    score = 0
    reasons = []

    if trend_dir == "SHORT":
        score += 12
    if m15_bias == "BEAR":
        score += 10
    else:
        reasons.append("M15 not bearish")

    if bb_m * 0.995 <= c <= bb_u * 1.005:
        score += 12
        reasons.append("Near BB mid/upper")
    else:
        reasons.append("Not near BB mid/upper")

    if (stoch_k < stoch_d) and (stoch_k_prev >= stoch_d_prev):
        score += 10
        reasons.append("Stoch cross down")
    elif stoch_k < stoch_d:
        score += 5
        reasons.append("Stoch < D")

    if 38 <= rsi_val <= 62:
        score += 8
    elif rsi_val < 38:
        reasons.append("RSI cold ok")
    else:
        reasons.append("RSI hot")

    if candle_bear or c < o:
        score += 8
        reasons.append("Bearish candle/engulf")

    ema20 = _ema(m5_close, 20)
    ema50 = _ema(m5_close, 50)
    if float(ema20.iloc[-1]) < float(ema50.iloc[-1]):
        score += 8
        reasons.append("M5 EMA20<EMA50")

    if adx_m5 >= 23:
        score += 6
    elif adx_m5 >= 18:
        score += 3

    return score, "; ".join(reasons)


def trigger_signal_m5(
    *,
    m5_df: pd.DataFrame,
    trend_dir: str,
    m15_df: pd.DataFrame,
    h1_df: pd.DataFrame,
    high_news: bool,
    min_score: int,
) -> Optional[Signal]:
    """
    Main M5 trigger engine with prioritised setups:
      1) Pullback (rewritten)
      2) Fresh Breakout
      3) Breakout Continuation
    Uses a **score gate**; returns None if score < min_score.
    """
    if m5_df is None or len(m5_df) < 80:
        return None

    high = m5_df["high"]
    low = m5_df["low"]
    close = m5_df["close"]
    open_ = m5_df["open"]

    bb_upper, bb_mid, bb_lower = bollinger_bands(close, period=20, std_factor=2.0)
    rsi_series = rsi(close, period=14)
    stoch_k, stoch_d = stochastic_oscillator(high, low, close, k_period=14, d_period=3)
    adx_series, plus_di, minus_di = adx(high, low, close, period=14)

    bull_eng = bullish_engulfing(open_, close)
    bear_eng = bearish_engulfing(open_, close)

    # latest + previous
    c = float(close.iloc[-1])
    o = float(open_.iloc[-1])
    c_prev = float(close.iloc[-2])

    bb_u = float(bb_upper.iloc[-1]); bb_m = float(bb_mid.iloc[-1]); bb_l = float(bb_lower.iloc[-1])
    bb_u_prev = float(bb_upper.iloc[-2]); bb_l_prev = float(bb_lower.iloc[-2])

    r = float(rsi_series.iloc[-1])
    k = float(stoch_k.iloc[-1]); d = float(stoch_d.iloc[-1])
    k_prev = float(stoch_k.iloc[-2]); d_prev = float(stoch_d.iloc[-2])

    adx_m5 = float(adx_series.iloc[-1]) if pd.notna(adx_series.iloc[-1]) else 0.0
    plus_di_m5 = float(plus_di.iloc[-1]) if pd.notna(plus_di.iloc[-1]) else 0.0
    minus_di_m5 = float(minus_di.iloc[-1]) if pd.notna(minus_di.iloc[-1]) else 0.0

    bar_time = m5_df["datetime"].iloc[-1]
    if isinstance(bar_time, pd.Timestamp):
        bar_time = bar_time.to_pydatetime()

    m15_bias = _m15_structure_bias(m15_df)
    extra_common = {
        "m5_rsi": r,
        "m5_stoch_k": k,
        "m5_stoch_d": d,
        "bb_upper": bb_u,
        "bb_mid": bb_m,
        "bb_lower": bb_l,
        "adx_m5": adx_m5,
        "plus_di_m5": plus_di_m5,
        "minus_di_m5": minus_di_m5,
        "m15_bias": m15_bias,
    }

    # -------------------------------------------------------------------------
    # 1) PULLBACK (rewritten)
    # -------------------------------------------------------------------------
    def _pullback_long() -> Optional[Tuple[Signal, int]]:
        candle_ok = (c > o) or bool(bull_eng.iloc[-1])
        score, why = _score_gate_long(
            c=c, o=o, bb_m=bb_m, bb_l=bb_l, rsi_val=r,
            stoch_k=k, stoch_d=d, stoch_k_prev=k_prev, stoch_d_prev=d_prev,
            m5_close=close, adx_m5=adx_m5, m15_bias=m15_bias, trend_dir=trend_dir,
            candle_bull=candle_ok
        )
        if score >= min_score and trend_dir == "LONG":
            extra = dict(extra_common)
            extra["setup_type"] = "PULLBACK_LONG"
            extra["entry_score"] = score
            reason = f"Pullback LONG: confluence [{why}]"
            return Signal("LONG", c, bar_time, reason, extra), score
        return None

    def _pullback_short() -> Optional[Tuple[Signal, int]]:
        candle_ok = (c < o) or bool(bear_eng.iloc[-1])
        score, why = _score_gate_short(
            c=c, o=o, bb_m=bb_m, bb_u=bb_u, rsi_val=r,
            stoch_k=k, stoch_d=d, stoch_k_prev=k_prev, stoch_d_prev=d_prev,
            m5_close=close, adx_m5=adx_m5, m15_bias=m15_bias, trend_dir=trend_dir,
            candle_bear=candle_ok
        )
        if score >= min_score and trend_dir == "SHORT":
            extra = dict(extra_common)
            extra["setup_type"] = "PULLBACK_SHORT"
            extra["entry_score"] = score
            reason = f"Pullback SHORT: confluence [{why}]"
            return Signal("SHORT", c, bar_time, reason, extra), score
        return None

    # -------------------------------------------------------------------------
    # 2) FRESH BREAKOUT
    # -------------------------------------------------------------------------
    def _breakout_long() -> Optional[Tuple[Signal, int]]:
        prev_inside = c_prev <= bb_u_prev * 0.999
        now_break = c > bb_u * 1.001
        stoch_ok = k >= 55
        rsi_ok = r >= 52
        adx_ok = adx_m5 >= 18

        score = 0
        if trend_dir == "LONG": score += 10
        if m15_bias == "BULL": score += 8
        if prev_inside and now_break: score += 14
        if stoch_ok: score += 6
        if rsi_ok: score += 6
        if adx_ok: score += 5

        if score >= min_score:
            extra = dict(extra_common)
            extra["setup_type"] = "BREAKOUT_LONG"
            extra["entry_score"] = score
            reason = "Breakout LONG: upper BB break with momentum."
            return Signal("LONG", c, bar_time, reason, extra), score
        return None

    def _breakout_short() -> Optional[Tuple[Signal, int]]:
        prev_inside = c_prev >= bb_l_prev * 1.001
        now_break = c < bb_l * 0.999
        stoch_ok = k <= 45
        rsi_ok = r <= 48
        adx_ok = adx_m5 >= 18

        score = 0
        if trend_dir == "SHORT": score += 10
        if m15_bias == "BEAR": score += 8
        if prev_inside and now_break: score += 14
        if stoch_ok: score += 6
        if rsi_ok: score += 6
        if adx_ok: score += 5

        if score >= min_score:
            extra = dict(extra_common)
            extra["setup_type"] = "BREAKOUT_SHORT"
            extra["entry_score"] = score
            reason = "Breakout SHORT: lower BB break with momentum."
            return Signal("SHORT", c, bar_time, reason, extra), score
        return None

    # -------------------------------------------------------------------------
    # 3) BREAKOUT CONTINUATION
    # -------------------------------------------------------------------------
    def _breakout_cont_long() -> Optional[Tuple[Signal, int]]:
        riding_upper = c >= bb_u * 0.999
        adx_ok = adx_m5 >= 20 and plus_di_m5 > minus_di_m5

        score = 0
        if trend_dir == "LONG": score += 10
        if m15_bias == "BULL": score += 8
        if riding_upper: score += 10
        if adx_ok: score += 8

        if score >= min_score:
            extra = dict(extra_common)
            extra["setup_type"] = "BREAKOUT_CONT_LONG"
            extra["entry_score"] = score
            reason = "Breakout continuation LONG: riding upper band with ADX support."
            return Signal("LONG", c, bar_time, reason, extra), score
        return None

    def _breakout_cont_short() -> Optional[Tuple[Signal, int]]:
        riding_lower = c <= bb_l * 1.001
        adx_ok = adx_m5 >= 20 and minus_di_m5 > plus_di_m5

        score = 0
        if trend_dir == "SHORT": score += 10
        if m15_bias == "BEAR": score += 8
        if riding_lower: score += 10
        if adx_ok: score += 8

        if score >= min_score:
            extra = dict(extra_common)
            extra["setup_type"] = "BREAKOUT_CONT_SHORT"
            extra["entry_score"] = score
            reason = "Breakout continuation SHORT: riding lower band with ADX support."
            return Signal("SHORT", c, bar_time, reason, extra), score
        return None

    # Prioritise: Pullback → Breakout → Continuation, in the current trend_dir
    candidates: list[Tuple[Signal, int]] = []

    if trend_dir == "LONG":
        for f in (_pullback_long, _breakout_long, _breakout_cont_long):
            out = f()
            if out:
                candidates.append(out)
    elif trend_dir == "SHORT":
        for f in (_pullback_short, _breakout_short, _breakout_cont_short):
            out = f()
            if out:
                candidates.append(out)

    if not candidates:
        return None

    # Choose highest score
    candidates.sort(key=lambda x: x[1], reverse=True)
    best = candidates[0][0]

    # Light news filter (don’t block, but mark)
    best.extra["news_flag"] = bool(high_news)

    return best