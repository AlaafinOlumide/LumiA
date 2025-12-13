# signal_formatter.py
import pandas as pd
import indicators


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


def confidence_label(score: int) -> str:
    if score >= 75:
        return "High"
    if score >= 55:
        return "Medium"
    return "Low"


def market_state_from_adx(adx_h1: float) -> str:
    return "TRENDING" if adx_h1 >= 20 else "RANGING"


def trend_strength_from_adx(adx_h1: float) -> str:
    if adx_h1 >= 35:
        return "Very Strong"
    if adx_h1 >= 25:
        return "Strong"
    if adx_h1 >= 20:
        return "Moderate"
    if adx_h1 >= 15:
        return "Weak"
    return "Very Weak"


def market_regime(h1_df: pd.DataFrame) -> str:
    close = h1_df["close"]
    high = h1_df["high"]
    low = h1_df["low"]

    upper, mid, lower = indicators.bollinger_bands(close, period=20, std_factor=2.0)
    bw = (upper - lower) / mid.replace(0, pd.NA)
    a = indicators.atr(high, low, close, period=14)

    bw_last = float(bw.iloc[-1]) if pd.notna(bw.iloc[-1]) else 0.0
    atr_last = float(a.iloc[-1]) if pd.notna(a.iloc[-1]) else 0.0

    if bw_last >= 0.02 or atr_last >= 20:
        return "High Volatility"
    if bw_last <= 0.008 and atr_last <= 10:
        return "Low Volatility / Compression"
    return "Normal Volatility"


def build_signal_message(
    symbol_label: str,
    signal,
    trend_source: str,
    session_window: str,
    high_news: bool,
    h1_df: pd.DataFrame,
) -> str:
    entry = float(signal.price)
    setup_type = signal.extra.get("setup_type", "GENERIC")
    setup_text = setup_label(setup_type)

    direction = "BUY" if signal.direction == "LONG" else "SELL"

    sl = float(signal.extra.get("sl", 0.0))
    tp1 = float(signal.extra.get("tp1", 0.0))
    tp2 = float(signal.extra.get("tp2", 0.0))

    conf = int(signal.extra.get("confidence", 0))
    conf_text = confidence_label(conf)

    # RR
    if signal.direction == "LONG":
        risk = entry - sl
        rr1 = (tp1 - entry) / risk if risk > 0 else 0.0
        rr2 = (tp2 - entry) / risk if risk > 0 else 0.0
    else:
        risk = sl - entry
        rr1 = (entry - tp1) / risk if risk > 0 else 0.0
        rr2 = (entry - tp2) / risk if risk > 0 else 0.0

    # H1 metrics
    adx_h1_series, _, _ = indicators.adx(h1_df["high"], h1_df["low"], h1_df["close"], period=14)
    adx_h1 = float(adx_h1_series.iloc[-1]) if pd.notna(adx_h1_series.iloc[-1]) else 0.0
    state = market_state_from_adx(adx_h1)
    strength = trend_strength_from_adx(adx_h1)
    regime = market_regime(h1_df)

    atr_h1 = signal.extra.get("atr_h1")
    atr_line = f"ATR(H1,14): {float(atr_h1):.2f}" if atr_h1 is not None else "ATR(H1,14): N/A"

    # M5 metrics
    rsi_m5 = float(signal.extra.get("m5_rsi", 0.0))
    stoch_k = float(signal.extra.get("m5_stoch_k", 0.0))
    adx_m5 = float(signal.extra.get("adx_m5", 0.0))
    pdi = float(signal.extra.get("plus_di_m5", 0.0))
    mdi = float(signal.extra.get("minus_di_m5", 0.0))
    bb_u = float(signal.extra.get("bb_upper", 0.0))
    bb_m = float(signal.extra.get("bb_mid", 0.0))
    bb_l = float(signal.extra.get("bb_lower", 0.0))

    trend_bias = "LONG" if signal.direction == "LONG" else "SHORT"

    lines = []
    lines.append("XAUUSD Signal [SCALP]")
    lines.append(f"{direction} {symbol_label} at {entry:.2f}")
    lines.append("")
    lines.append(f"SL: {sl:.2f}")
    lines.append(f"TP1: {tp1:.2f}")
    lines.append(f"TP2: {tp2:.2f}")
    lines.append("")
    lines.append(f"Setup: {setup_text} ({setup_type})")
    lines.append(f"Confidence: {conf} ({conf_text})")
    lines.append("")
    lines.append(f"RR to TP1: {rr1:.2f}R | RR to TP2: {rr2:.2f}R")
    lines.append("")
    lines.append(f"Time (UTC): {signal.time.isoformat()}")
    lines.append(f"Trend Bias: {trend_bias} (source: {trend_source})")
    lines.append(f"Session: {session_window}")
    lines.append(f"Reason: {signal.reason}")
    lines.append("")
    lines.append(f"Market State (H1): {state} (ADX {adx_h1:.2f})")
    lines.append(f"Trend Strength (H1): {strength}")
    lines.append(f"Market Regime: {regime}")
    lines.append(atr_line)
    lines.append("")
    lines.append("No high-impact news flag near this time." if not high_news else "HIGH-IMPACT NEWS NEARBY: expect extra volatility.")
    lines.append(f"RSI(M5): {rsi_m5:.2f} | StochK(M5): {stoch_k:.2f}")
    lines.append(f"ADX(M5): {adx_m5:.2f} (+DI: {pdi:.2f}, -DI: {mdi:.2f})")
    lines.append(f"BB(M5): upper {bb_u:.2f}, mid {bb_m:.2f}, lower {bb_l:.2f}")

    return "\n".join(lines)
