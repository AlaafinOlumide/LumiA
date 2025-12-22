# signal_formatter.py
from __future__ import annotations

import datetime as dt
from strategy import Signal


def _fmt(x: float) -> str:
    return f"{x:.2f}"


def build_signal_message(
    *,
    symbol_label: str,
    signal: Signal,
    trend_label: str,
    trend_source: str,
    session_window: str,
    high_news: bool,
    market_state: str,
    market_regime_text: str,
    adx_h1: float,
    trend_strength: str,
    confidence_score: int,
    confidence_text: str,
) -> str:
    setup_type = str(signal.extra.get("setup_type", "GENERIC"))
    entry_score = int(signal.extra.get("entry_score", 0))

    sl = signal.extra.get("sl")
    tp1 = signal.extra.get("tp1")
    tp2 = signal.extra.get("tp2")

    rr1 = signal.extra.get("rr_tp1")
    rr2 = signal.extra.get("rr_tp2")

    direction_word = "BUY" if signal.direction == "LONG" else "SELL"

    lines = []
    lines.append(f"{symbol_label} Signal [SCALP]")
    lines.append(f"{direction_word} {symbol_label} at {_fmt(signal.price)}")
    lines.append("")
    if sl is not None:
        lines.append(f"SL: {_fmt(float(sl))}")
    if tp1 is not None:
        lines.append(f"TP1: {_fmt(float(tp1))}")
    if tp2 is not None:
        lines.append(f"TP2: {_fmt(float(tp2))}")

    lines.append("")
    lines.append(f"Setup: {setup_type.replace('_', ' ').title()} ({setup_type})")
    lines.append(f"Entry Score: {entry_score}")
    lines.append(f"Confidence: {confidence_score} ({confidence_text})")

    if rr1 is not None and rr2 is not None:
        lines.append("")
        lines.append(f"RR to TP1: {rr1}R | RR to TP2: {rr2}R")

    lines.append("")
    t = signal.time
    if isinstance(t, dt.datetime):
        lines.append(f"Time (UTC): {t.isoformat()}")
    else:
        lines.append(f"Time (UTC): {signal.time}")

    lines.append(f"Trend Bias: {trend_label} (source: {trend_source})")
    lines.append(f"Session: {session_window}")
    lines.append(f"Reason: {signal.reason}")

    lines.append("")
    lines.append(f"Market State (H1): {market_state} (ADX {adx_h1:.2f})")
    lines.append(f"Trend Strength (H1): {trend_strength}")
    lines.append(f"Market Regime: {market_regime_text}")

    atr_h1 = signal.extra.get("atr_h1")
    if atr_h1 is not None:
        lines.append(f"ATR(H1,14): {float(atr_h1):.2f}")

    lines.append("")
    lines.append("No high-impact news flag near this time." if not high_news else "⚠️ High-impact news nearby: be careful.")

    lines.append(
        f"RSI(M5): {float(signal.extra.get('m5_rsi', 0.0)):.2f} | "
        f"StochK(M5): {float(signal.extra.get('m5_stoch_k', 0.0)):.2f}"
    )
    lines.append(
        f"ADX(M5): {float(signal.extra.get('adx_m5', 0.0)):.2f} "
        f"(+DI: {float(signal.extra.get('plus_di_m5', 0.0)):.2f}, -DI: {float(signal.extra.get('minus_di_m5', 0.0)):.2f})"
    )
    lines.append(
        f"BB(M5): upper {float(signal.extra.get('bb_upper', 0.0)):.2f}, "
        f"mid {float(signal.extra.get('bb_mid', 0.0)):.2f}, "
        f"lower {float(signal.extra.get('bb_lower', 0.0)):.2f}"
    )

    if "m15_structure" in signal.extra:
        lines.append(
            f"M15 Structure: {signal.extra.get('m15_structure')} "
            f"(EMA20 {float(signal.extra.get('m15_ema20', 0.0)):.2f}, EMA50 {float(signal.extra.get('m15_ema50', 0.0)):.2f})"
        )

    mgmt = signal.extra.get("management_notes")
    if mgmt:
        lines.append("")
        lines.append(str(mgmt))

    return "\n".join(lines)