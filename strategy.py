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

    This version is more conservative than the last one:
    - Reversal requires rejection (close back inside band)
    - Breakout requires proper break (prev inside, current strong close outside)
    """

    if len(df) < 30:
        return {"mode": None, "direction": None, "confirmations": 0}

    last = df.iloc[-1]
    prev = df.iloc[-2]

    sig = {"mode": None, "direction": None, "confirmations": 0}

    # ------------- 1) REVERSAL SETUPS (BOUNCE) -------------

    # BUY reversal at lower band:
    # - prev candle: at/under lower band
    # - last candle: closes back ABOVE lower band (rejection)
    if prev.close <= prev.BB_LOWER and last.close > last.BB_LOWER:
        sig["mode"] = "Reversal"

        # RSI oversold (stricter again)
        if last.RSI < 30:
            sig["confirmations"] += 1

        # Stoch bullish behaviour (crossover or clear upturn from low)
        if (
            (prev["%K"] < 20 and last["%K"] > last["%D"])  # crossover from oversold
            or (last["%K"] > prev["%K"] and last["%K"] < 40)  # rising from low area
        ):
            sig["confirmations"] += 1

        # Bullish candle body
        if last.close > last.open:
            sig["confirmations"] += 1

        # At least 2 confirmations
        if sig["confirmations"] >= 2:
            sig["direction"] = "BUY"

    # SELL reversal at upper band:
    # - prev candle: at/above upper band
    # - last candle: closes back BELOW upper band
    elif prev.close >= prev.BB_UPPER and last.close < last.BB_UPPER:
        sig["mode"] = "Reversal"

        # RSI overbought
        if last.RSI > 70:
            sig["confirmations"] += 1

        # Stoch bearish behaviour
        if (
            (prev["%K"] > 80 and last["%K"] < last["%D"])  # cross down from overbought
            or (last["%K"] < prev["%K"] and last["%K"] > 60)  # turning down from high
        ):
            sig["confirmations"] += 1

        # Bearish candle
        if last.close < last.open:
            sig["confirmations"] += 1

        if sig["confirmations"] >= 2:
            sig["direction"] = "SELL"

    # ------------- 2) BREAKOUT SETUPS -------------

    # Only consider breakouts if no reversal already triggered
    if sig["direction"] is None:
        b_sig = {"mode": None, "direction": None, "confirmations": 0}

        # Helpers
        atr = float(last.ATR) if not pd.isna(last.ATR) else 0.0
        body = abs(last.close - last.open)

        # BUY breakout:
        # - prev close inside band
        # - last close clearly above upper band
        # - body >= 0.5 * ATR (so it's not a tiny poke)
        if (
            prev.close <= prev.BB_UPPER
            and prev.close >= prev.BB_LOWER
            and last.close > last.BB_UPPER
        ):
            b_sig["mode"] = "Breakout"

            # 1) price outside band
            b_sig["confirmations"] += 1

            # 2) strong body relative to ATR
            if atr > 0 and body >= 0.5 * atr:
                b_sig["confirmations"] += 1

            # 3) RSI bullish momentum
            if last.RSI > 60:
                b_sig["confirmations"] += 1

            if b_sig["confirmations"] >= 2:
                b_sig["direction"] = "BUY"

        # SELL breakout:
        # - prev close inside band
        # - last close clearly below lower band
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

        # If we found a breakout signal, override reversals
        if b_sig.get("direction"):
            sig = b_sig

    return sig
