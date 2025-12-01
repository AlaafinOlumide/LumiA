from typing import Dict
import pandas as pd

def analyze(df: pd.DataFrame) -> Dict:
    """
    Detect only strong BOUNCE (reversal) or BREAKOUT patterns.

    Returns:
    {
        "mode": "Reversal" | "Breakout" | None,
        "direction": "BUY" | "SELL" | None,
        "confirmations": int
    }
    """

    if len(df) < 30:
        return {"mode": None, "direction": None, "confirmations": 0}

    last = df.iloc[-1]   # current closed candle
    prev = df.iloc[-2]   # previous candle

    sig = {"mode": None, "direction": None, "confirmations": 0}

    # ------------- 1) BOUNCE / REVERSAL SETUPS -------------

    # BUY bounce (reversal up) at lower band:
    # - previous candle closed at/under lower band
    # - current candle closes back above lower band (rejection)
    if prev.close <= prev.BB_LOWER and last.close > last.BB_LOWER:
        sig["mode"] = "Reversal"

        # RSI oversold
        if last.RSI < 30:
            sig["confirmations"] += 1

        # Stoch bullish behaviour
        if (
            (prev["%K"] < 20 and last["%K"] > last["%D"])  # crossover from low
            or (last["%K"] > prev["%K"] and last["%K"] < 40)  # turning up from low
        ):
            sig["confirmations"] += 1

        # Bullish candle
        if last.close > last.open:
            sig["confirmations"] += 1

        if sig["confirmations"] >= 2:
            sig["direction"] = "BUY"

    # SELL bounce (reversal down) at upper band:
    # - previous candle closed at/above upper band
    # - current candle closes back below upper band
    elif prev.close >= prev.BB_UPPER and last.close < last.BB_UPPER:
        sig["mode"] = "Reversal"

        # RSI overbought
        if last.RSI > 70:
            sig["confirmations"] += 1

        # Stoch bearish behaviour
        if (
            (prev["%K"] > 80 and last["%K"] < last["%D"])  # cross down from high
            or (last["%K"] < prev["%K"] and last["%K"] > 60)  # rolling over
        ):
            sig["confirmations"] += 1

        # Bearish candle
        if last.close < last.open:
            sig["confirmations"] += 1

        if sig["confirmations"] >= 2:
            sig["direction"] = "SELL"

    # ------------- 2) BREAKOUT SETUPS -------------

    # Only look for breakout if no reversal already chosen
    if sig["direction"] is None:
        b_sig = {"mode": None, "direction": None, "confirmations": 0}

        atr = float(last.ATR) if not pd.isna(last.ATR) else 0.0
        body = abs(last.close - last.open)

        # BUY breakout:
        # - previous close inside bands
        # - current close above upper band
        # - body >= 0.5 * ATR
        if (
            prev.close <= prev.BB_UPPER
            and prev.close >= prev.BB_LOWER
            and last.close > last.BB_UPPER
        ):
            b_sig["mode"] = "Breakout"

            # price outside band
            b_sig["confirmations"] += 1

            # decent body size
            if atr > 0 and body >= 0.5 * atr:
                b_sig["confirmations"] += 1

            # RSI bullish momentum
            if last.RSI > 60:
                b_sig["confirmations"] += 1

            if b_sig["confirmations"] >= 2:
                b_sig["direction"] = "BUY"

        # SELL breakout:
        # - previous close inside bands
        # - current close below lower band
        # - body >= 0.5 * ATR
        elif (
            prev.close >= prev.BB_LOWER
            and prev.close <= prev.BB_UPPER
            and last.close < last.BB_LOWER
        ):
            b_sig["mode"] = "Breakout"

            b_sig["confirmations"] += 1

            if atr > 0 and body >= 0.5 * atr:
                b_sig["confirmations"] += 1

            if last.RSI < 40:
                b_sig["confirmations"] += 1

            if b_sig["confirmations"] >= 2:
                b_sig["direction"] = "SELL"

        if b_sig.get("direction"):
            sig = b_sig

    return sig
