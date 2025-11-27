
from typing import Dict
import pandas as pd

def analyze(df: pd.DataFrame) -> Dict:
    """Return a signal dict: {mode, direction, confirmations}."""
    if len(df) < 30:
        return {"mode": None, "direction": None, "confirmations": 0}

    last = df.iloc[-1]
    prev = df.iloc[-2]

    sig = {"mode": None, "direction": None, "confirmations": 0}

    # --- Reversal Mode ---
    # BUY reversal at lower band
    if last.close <= last.BB_LOWER:
        sig["mode"] = "Reversal"

        # RSI oversold
        if last.RSI < 30:
            sig["confirmations"] += 1

        # Stochastic bullish crossover from oversold
        if prev["%K"] < 20 and last["%K"] > last["%D"]:
            sig["confirmations"] += 1

        # Bullish candle
        if last.close > last.open:
            sig["confirmations"] += 1

        if sig["confirmations"] >= 3:
            sig["direction"] = "BUY"

    # SELL reversal at upper band
    elif last.close >= last.BB_UPPER:
        sig["mode"] = "Reversal"

        if last.RSI > 70:
            sig["confirmations"] += 1
        if prev["%K"] > 80 and last["%K"] < last["%D"]:
            sig["confirmations"] += 1
        if last.close < last.open:
            sig["confirmations"] += 1

        if sig["confirmations"] >= 3:
            sig["direction"] = "SELL"

    # --- Breakout Mode (simple) ---
    # Only check breakout if no reversal chosen
    if sig["direction"] is None:
        bb_width = (last.BB_UPPER - last.BB_LOWER) / last.BB_MID if last.BB_MID != 0 else 0
        bb_width_series = (df["BB_UPPER"] - df["BB_LOWER"]) / df["BB_MID"]
        avg_width = bb_width_series.rolling(20).mean().iloc[-1]

        if bb_width < avg_width:
            if last.close > last.BB_UPPER:
                sig = {"mode": "Breakout", "direction": "BUY", "confirmations": 3}
            elif last.close < last.BB_LOWER:
                sig = {"mode": "Breakout", "direction": "SELL", "confirmations": 3}

    return sig
