
from typing import Dict

def format_signal(pair: str, tf: str, last, sig: Dict) -> str | None:
    if sig.get("direction") is None:
        return None

    direction = sig["direction"]
    mode = sig["mode"]
    conf = sig["confirmations"]

    atr = round(float(last.ATR), 2)
    entry = float(last.close)

    if direction == "BUY":
        sl = entry - (1.5 * atr)
        tp = entry + (3 * atr)
    else:
        sl = entry + (1.5 * atr)
        tp = entry - (3 * atr)

    text = f"""{pair} – {direction} – {mode}
Timeframe: {tf} | Confidence: {conf}/4

Entry: {entry:.2f}
SL: {sl:.2f}
TP: {tp:.2f}

Indicators:
RSI: {float(last.RSI):.2f}
Stoch: {float(last['%K']):.2f} / {float(last['%D']):.2f}
ATR: {atr:.2f}
"""
    return text.strip()
