
# XAUUSD Signal Bot (Render + TwelveData + TradingView + Telegram)

This bot:

- Tracks **XAUUSD** on a **5-minute timeframe**
- Fetches OHLC data from **TwelveData**
- Computes Bollinger Bands, RSI, Stochastic, ATR
- Detects **Reversal** and **Breakout** setups
- Confirms setups using a **TradingView webhook**
- Sends formatted signals to **Telegram**
- Runs on **Render** as a single web service

## Environment Variables (Render)

Set these in Render's Dashboard:

- `TWELVEDATA_API_KEY` – your TwelveData API key
- `TELEGRAM_BOT_TOKEN` – Telegram bot token from BotFather
- `TELEGRAM_CHAT_ID` – Chat ID (user or group) for signals

Render automatically provides `PORT` for the web process.

## Start Commands

- Build: `pip install -r requirements.txt`
- Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`

## TradingView Setup

1. Copy `tradingview_xauusd_signal_confirmer.pine` into TradingView Pine Editor.
2. Save and add it to the **XAUUSD 5m** chart.
3. Create an Alert:
   - Condition: **Any alert() function call**
   - Webhook URL: `https://YOUR-RENDER-URL.onrender.com/tv-webhook`
   - Message: leave empty (script provides JSON).

The JSON looks like:

```json
{"symbol":"XAUUSD","direction":"BUY","mode":"Reversal"}
```

The bot will only send a Telegram signal if:

- Its own logic finds a setup, AND
- A matching TradingView webhook (same symbol/direction/mode) was received in the last 10 minutes, AND
- Cooldown and non-duplicate conditions are satisfied.
