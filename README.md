# XAUUSD Multi-timeframe Telegram Signal Bot  
(Render + Twelve Data + ForexFactory News)

**DISCLAIMER:** Educational project only. This is **not financial advice**. Use at your own risk.

---

## Overview

This bot is a **rule-based XAUUSD (Gold) signal generator** that runs as a **background worker on Render** and sends **closed-candle trade signals** to Telegram.

It is designed to prioritise **accuracy, structure, and risk control** over signal frequency.

---

## Data Sources

- **Market data:** Twelve Data (XAU/USD)
- **News filter:** ForexFactory (high-impact USD events)
- **Execution:** Telegram (signals only — no auto trading)

---

## Multi-Timeframe Logic

| Timeframe | Purpose |
|---------|--------|
| **H1** | Primary trend detection (EMA + ADX + DI) |
| **M15** | Structure & trend confirmation (EMA + RSI) |
| **M5** | Entry trigger (BB, RSI, Stochastic, ADX, candle patterns) |

Signals are evaluated **only on closed M5 candles**.

---

## Supported Setups

- **Pullback (Trend Continuation)**
- **Breakout Continuation**

Each setup passes through:
- Trend alignment (H1 → M15 → M5)
- Volatility checks (ATR, Bollinger Bands)
- Momentum filters (RSI, Stochastic)
- Score-based entry gate
- Cooldown protection

---

## Trading Window

- **Active hours:** `08:00 – 19:00 UTC`
- **Weekends:** Disabled by default (configurable)

---

## Risk Profile Tags

Each Telegram signal is automatically tagged as:

- **SCALP** → M5 ADX between `18–29`
- **SWING** → M5 ADX `>= 30`

(Thresholds are configurable in the code.)

---

## Signal Output

Signals include:
- Entry price
- Stop Loss (ATR-based)
- TP1 / TP2 (R-multiples)
- Setup type
- Confidence score & label
- Trend bias & source
- Market regime
- News proximity flag

Notes

This bot does not place trades

It is intended for analysis, learning, and signal evaluation

Performance depends on market conditions and configuration
