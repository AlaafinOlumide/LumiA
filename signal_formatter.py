# signal_formatter.py

from __future__ import annotations
from typing import Optional

from strategy import Signal


def _fmt(x: float) -> str:
    return f"{x:.2f}"


def build_signal_message(
    signal: Signal,
    symbol_label: str,
    session: str,
    confidence_text: str = "Moderate",
    note: Optional[str] = None,
) -> str:
    direction_word = "BUY" if signal.direction == "BUY" else "SELL"

    lines = []
    lines.append(f"🚨 {direction_word} SIGNAL ({session})")
    lines.append(f"{direction_word} {symbol_label} at {_fmt(signal.entry)}")
    lines.append(f"SL: {_fmt(signal.sl)}")
    lines.append(f"TP1: {_fmt(signal.tp1)}")
    lines.append(f"TP2: {_fmt(signal.tp2)}")
    lines.append(f"Confidence: {confidence_text}")

    if signal.extra:
        score = signal.extra.get("entry_score")
        req = signal.extra.get("required_score")
        if score is not None and req is not None:
            lines.append(f"Score: {score}/{req}")

    if note:
        lines.append("")
        lines.append(note)

    return "\n".join(lines)