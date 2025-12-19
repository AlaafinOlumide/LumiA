# signal_formatter.py
from typing import Optional

def setup_label(setup_type: str) -> str:
    if not setup_type:
        return "Generic"
    if setup_type.startswith("PULLBACK"):
        return "Pullback"
    if setup_type.startswith("BREAKOUT_CONT"):
        return "Breakout Continuation"
    if setup_type.startswith("BREAKOUT"):
        return "Breakout"
    return "Generic"

def risk_tag_from_adx_m5(adx_m5: float) -> str:
    return "SCALP"

def build_signal_message(
    symbol_label: str,
    signal,
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
    entry = float(signal.price)
    setup_type = signal.extra.get("setup_type", "GENERIC")
    setup_text = setup_label(setup_type)

    adx_m5 = float(signal.extra.get("adx_m5", 0.0))
    risk_tag = risk_tag_from_adx_m5(adx_m5)

    sl = signal.extra.get("sl")
    tp1 = signal.extra.get("tp1")
    tp2 = signal.extra.get("tp2")
    atr_h1 = signal.extra.get("atr_h1")

    arrow = "BUY" if signal.direction == "LONG" else "SELL"

    # RR
    rr_line = ""
    if sl is not None and tp1 is not None:
        if signal.direction == "LONG":
            risk = entry - float(sl)
            rr1 = (float(tp1) - entry) / risk if risk > 0 else 0.0
            rr2 = (float(tp2) - entry) / risk if (risk > 0 and tp2 is not None) else None
        else:
            risk = float(sl) - entry
            rr1 = (entry - float(tp1)) / risk if risk > 0 else 0.0
            rr2 = (entry - float(tp2)) / risk if (risk > 0 and tp2 is not None) else None

        if rr2 is None:
            rr_line = f"RR to TP1: {rr1:.2f}R"
        else:
            rr_line = f"RR to TP1: {rr1:.2f}R | RR to TP2: {rr2:.2f}R"

    lines = []
    # Template exactly like you asked (spacing kept)
    lines.append(f"XAUUSD Signal [{risk_tag}]")
    lines.append(f"{arrow} {symbol_label} at {entry:.2f}")
    lines.append("")

    if sl is not None:
        lines.append(f"SL: {float(sl):.2f}")
    if tp1 is not None:
        lines.append(f"TP1: {float(tp1):.2f}")
    if tp2 is not None:
        lines.append(f"TP2: {float(tp2):.2f}")

    lines.append("")
    lines.append(f"Setup: {setup_text} ({setup_type})")
    lines.append(f"Confidence: {confidence_score} ({confidence_text})")

    entry_score = signal.extra.get("entry_score")
    if entry_score is not None:
        lines.append(f"Entry Score: {entry_score}/10")

    if rr_line:
        lines.append("")
        lines.append(rr_line)

    lines.append("")
    lines.append(f"Time (UTC): {signal.time.isoformat()}")
    lines.append(f"Trend Bias: {trend_label} (source: {trend_source})")
    lines.append(f"Session: {session_window}")
    lines.append(f"Reason: {signal.reason}")
    lines.append("")
    lines.append(f"Market State (H1): {market_state} (ADX {adx_h1:.2f})")
    lines.append(f"Trend Strength (H1): {trend_strength}")
    lines.append(f"Market Regime: {market_regime_text}")
    if atr_h1 is not None:
        lines.append(f"ATR(H1,14): {float(atr_h1):.2f}")
    lines.append("")
    lines.append("No high-impact news flag near this time." if not high_news else "HIGH-IMPACT NEWS NEARBY: expect extra volatility.")
    lines.append(
        f"RSI(M5): {float(signal.extra.get('m5_rsi', 0.0)):.2f} | "
        f"StochK(M5): {float(signal.extra.get('m5_stoch_k', 0.0)):.2f}"
    )
    lines.append(
        f"ADX(M5): {float(signal.extra.get('adx_m5', 0.0)):.2f} "
        f"(+DI: {float(signal.extra.get('plus_di_m5', 0.0)):.2f}, "
        f"-DI: {float(signal.extra.get('minus_di_m5', 0.0)):.2f})"
    )
    lines.append(
        "BB(M5): upper {0:.2f}, mid {1:.2f}, lower {2:.2f}".format(
            float(signal.extra.get("bb_upper", 0.0)),
            float(signal.extra.get("bb_mid", 0.0)),
            float(signal.extra.get("bb_lower", 0.0)),
        )
    )

    return "\n".join(lines)
