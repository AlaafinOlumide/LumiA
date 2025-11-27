# XAUUSD Signal Bot (Render + TwelveData + Telegram)

This bot:

- Tracks **XAUUSD** on a **5-minute timeframe**
- Fetches OHLC data from **TwelveData**
- Computes Bollinger Bands, RSI, Stochastic, ATR
- Detects **Reversal** and **Breakout** setups
- Enforces:
  - Checks only on exact times x:00, x:05, x:10, ..., x:55 (UTC)
  - Cooldown between signals
  - No duplicate signal for the same setup until it resets
- Sends formatted signals to **Telegram**
- Runs on **Render** as a single web service (FastAPI healthcheck + background bot loop)

## Environment Variables (Render)

- `TWELVEDATA_API_KEY` – your TwelveData API key
- `TELEGRAM_BOT_TOKEN` – Telegram bot token from BotFather
- `TELEGRAM_CHAT_ID` – Chat ID (user or group) for signals

## Commands

- Build: `pip install -r requirements.txt`
- Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`
