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
    if last.close <= last.BB_LOWER:
        sig["mode"] = "Reversal"
        if last.RSI < 30:
            sig["confirmations"] += 1
        if prev["%K"] < 20 and last["%K"] > last["%D"]:
            sig["confirmations"] += 1
        if last.close > last.open:
            sig["confirmations"] += 1
        if sig["confirmations"] >= 3:
            sig["direction"] = "BUY"

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

    # --- Breakout Mode ---
    if sig["direction"] is None:
        bb_width_series = (df["BB_UPPER"] - df["BB_LOWER"]) / df["BB_MID"]
        bb_width = bb_width_series.iloc[-1]
        avg_width = bb_width_series.rolling(20).mean().iloc[-1]

        if bb_width < avg_width:
            if last.close > last.BB_UPPER:
                sig = {"mode": "Breakout", "direction": "BUY", "confirmations": 3}
            elif last.close < last.BB_LOWER:
                sig = {"mode": "Breakout", "direction": "SELL", "confirmations": 3}

    return sig
