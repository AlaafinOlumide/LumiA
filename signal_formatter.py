from typing import Tuple
from strategy import Signal


def rr(entry: float, sl: float, tp: float) -> float:
    risk = abs(entry - sl)
    reward = abs(tp - entry)
    return round(reward / risk, 2) if risk > 0 else 0.0


def confidence_label(score: int) -> str:
    if score >= 80:
        return "Very High"
    if score >= 65:
        return "High"
    if score >= 50:
        return "Medium"
    return "Low"


def format_signal_message(
    signal: Signal,
    sl: float,
    tp1: float,
    tp2: float,
    confidence: int,
    trend_bias: str,
    trend_source: str,
    session: str,
    market_state: str,
    trend_strength: str,
    market_regime: str,
    atr_h1: float,
    h1_adx: float,
    news_ok: bool,
) -> str:

    rr1 = rr(signal.price, sl, tp1)
    rr2 = rr(signal.price, sl, tp2)

    return f"""
XAUUSD Signal [SCALP]
{signal.direction} XAUUSD at {signal.price:.2f}

SL: {sl:.2f}
TP1: {tp1:.2f}
TP2: {tp2:.2f}

Setup: {signal.extra["setup_type"].replace("_", " ").title()} ({signal.extra["setup_type"]})
Confidence: {confidence} ({confidence_label(confidence)})

RR to TP1: {rr1}R | RR to TP2: {rr2}R

Time (UTC): {signal.time.isoformat()}
Trend Bias: {trend_bias} (source: {trend_source})
Session: {session}
Reason: {signal.reason}

Market State (H1): {market_state} (ADX {h1_adx:.2f})
Trend Strength (H1): {trend_strength}
Market Regime: {market_regime}
ATR(H1,14): {atr_h1:.2f}

{"No high-impact news flag near this time." if news_ok else "⚠ High-impact news nearby."}
RSI(M5): {signal.extra["m5_rsi"]:.2f} | StochK(M5): {signal.extra["m5_stoch_k"]:.2f}
ADX(M5): {signal.extra["adx_m5"]:.2f} (+DI: {signal.extra["plus_di_m5"]:.2f}, -DI: {signal.extra["minus_di_m5"]:.2f})
BB(M5): upper {signal.extra["bb_upper"]:.2f}, mid {signal.extra["bb_mid"]:.2f}, lower {signal.extra["bb_lower"]:.2f}
""".strip()
