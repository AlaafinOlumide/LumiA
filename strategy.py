from typing import Dict
import pandas as pd

def analyze(df: pd.DataFrame) -> Dict:
    """
    Return a signal dict:
    {
        "mode": "Reversal" | "Breakout" | None,
        "direction": "BUY" | "SELL" | None,
        "confirmations": int
    }

    This version is more "relaxed":
    - RSI thresholds widened (35 / 65 instead of 30 / 70)
    - Reversal needs only 2 confirmations (was 3)
    - Breakout does not require a Bollinger squeeze, only a solid close outside band.
    """
    if len(df) < 30:
        return {"mode": None, "direction": None, "confirmations": 0}

    last = df.iloc[-1]
    prev = df.iloc[-2]

    sig = {"mode": None, "direction": None, "confirmations": 0}

    # -----------------------
    # 1) REVERSAL SETUPS
    # -----------------------

    # BUY reversal at lower band
    if last.close <= last.BB_LOWER:
        sig["mode"] = "Reversal"

        # RSI oversold but relaxed
        if last.RSI < 35:
            sig["confirmations"] += 1

        # Stochastic bullish behaviour:
        # either a crossover OR K rising from low area
        if (
            (prev["%K"] < 25 and last["%K"] > last["%D"])  # classic crossover from low
            or (last["%K"] > prev["%K"] and last["%K"] < 40)  # K turning up from low region
        ):
            sig["confirmations"] += 1

        # Bullish candle body
        if last.close > last.open:
            sig["confirmations"] += 1

        # 🔥 Only 2 confirmations needed now (was 3)
        if sig["confirmations"] >= 2:
            sig["direction"] = "BUY"

    # SELL reversal at upper band
    elif last.close >= last.BB_UPPER:
        sig["mode"] = "Reversal"

        # RSI overbought but relaxed
        if last.RSI > 65:
            sig["confirmations"] += 1

        # Stochastic bearish behaviour:
        if (
            (prev["%K"] > 75 and last["%K"] < last["%D"])  # classic cross down from high
            or (last["%K"] < prev["%K"] and last["%K"] > 60)  # K turning down from high region
        ):
            sig["confirmations"] += 1

        # Bearish candle body
        if last.close < last.open:
            sig["confirmations"] += 1

        if sig["confirmations"] >= 2:
            sig["direction"] = "SELL"

    # -----------------------
    # 2) BREAKOUT SETUPS
    # -----------------------
    # Only check breakout if NO reversal already triggered
    if sig["direction"] is None:
        # Simple breakout:
        # - Close beyond band
        # - RSI in direction of move (momentum)
        # We count confirmations to keep a similar interface.
        b_sig = {"mode": None, "direction": None, "confirmations": 0}

        # BUY breakout above upper band
        if last.close > last.BB_UPPER:
            b_sig["mode"] = "Breakout"

            # Price clearly outside band
            b_sig["confirmations"] += 1

            # RSI showing bullish momentum
            if last.RSI > 55:
                b_sig["confirmations"] += 1

            # Optional: Stoch rising
            if last["%K"] > last["%D"]:
                b_sig["confirmations"] += 1

            if b_sig["confirmations"] >= 2:
                b_sig["direction"] = "BUY"

        # SELL breakout below lower band
        elif last.close < last.BB_LOWER:
            b_sig["mode"] = "Breakout"

            b_sig["confirmations"] += 1

            # RSI showing bearish momentum
            if last.RSI < 45:
                b_sig["confirmations"] += 1

            # Optional: Stoch falling
            if last["%K"] < last["%D"]:
                b_sig["confirmations"] += 1

            if b_sig["confirmations"] >= 2:
                b_sig["direction"] = "SELL"

        # If breakout signal exists, override sig
        if b_sig.get("direction"):
            sig = b_sig

    return sig