# XAUUSD Multi-timeframe Telegram Signal Bot (Render + Twelve Data + ForexFactory News)

**DISCLAIMER: Educational example only. Not financial advice. Use at your own risk.**

This bot:

- Runs as a background worker on Render
- Fetches XAU/USD data from Twelve Data
- Uses:
  - M5: signal trigger (Bollinger Bands, RSI, Stochastic, candlesticks, ADX)
  - M15: trend confirmation (EMA + RSI)
  - H1: trend detection (EMA + ADX + DI)
- Only looks for trades during:
  - 08:00–19:00 UTC
- Sends signals to a Telegram chat
- Logs each signal into `signals_log.csv`
- Flags high-impact ForexFactory news near the signal time

## Risk profile tags

Inside the Telegram message, each signal is tagged as:

- `SCALP` when M5 ADX is moderate (18 <= ADX < 30)
- `SWING` when M5 ADX is stronger (ADX >= 30)

You can tweak these levels in `main.py` in the `build_signal_message` function.
