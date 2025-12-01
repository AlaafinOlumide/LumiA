from fastapi import FastAPI
import threading
import time
from datetime import datetime, timezone, timedelta
import os
import requests

from data_provider import get_xauusd_data
from indicators import bollinger_bands, rsi, stochastic, atr
from strategy import analyze
from format_signal import format_signal

# Uvicorn expects this at top-level
app = FastAPI()

PAIR = "XAUUSD"
TF_LABEL = "5M"
INTERVAL = "5min"          # TwelveData interval
COOLDOWN_MINUTES = 5       # still keep small cooldown to avoid spam

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if TELEGRAM_BOT_TOKEN is None or TELEGRAM_CHAT_ID is None:
    print("⚠️ TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set. Telegram sending will be skipped.")

# State
last_setup_id: str | None = None
last_signal_ts: datetime | None = None
last_candle_id: str | None = None   # track last processed candle (by datetime string)


def get_setup_id(sig: dict):
    """
    Turn a signal dict into a simple ID string, e.g. 'Reversal_BUY' or 'Breakout_SELL'.
    Returns None if there is no valid direction/mode.
    """
    if not sig.get("direction") or not sig.get("mode"):
        return None
    return f"{sig['mode']}_{sig['direction']}"


def is_in_cooldown() -> bool:
    """True if we are still inside the cooldown window after the last signal."""
    global last_signal_ts
    if last_signal_ts is None:
        return False
    return datetime.now(timezone.utc) < last_signal_ts + timedelta(minutes=COOLDOWN_MINUTES)


def send_telegram_message(text: str):
    """Send a plain text message to the configured Telegram chat."""
    if TELEGRAM_BOT_TOKEN is None or TELEGRAM_CHAT_ID is None:
        print("Telegram env vars not set, skipping send.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}

    try:
        r = requests.post(url, json=payload, timeout=10)
        if not r.ok:
            print("Telegram send error:", r.text)
    except Exception as e:
        print("Telegram exception:", e)


def bot_loop():
    global last_setup_id, last_signal_ts, last_candle_id

    while True:
        try:
            # 🔁 Poll every 30 seconds (NOT tied to 5-min clock); only act when a NEW candle is closed
            time.sleep(30)

            df = get_xauusd_data(interval=INTERVAL, outputsize=200)
            df = bollinger_bands(df)
            df = rsi(df)
            df = stochastic(df)
            df = atr(df)

            # TwelveData returns a 'datetime' column for each candle
            if "datetime" not in df.columns:
                # fallback: just process latest, but this should not normally happen
                candle_id = str(df.index[-1])
            else:
                candle_id = str(df["datetime"].iloc[-1])

            # ⚠️ If this is the same candle we already processed, skip
            if last_candle_id == candle_id:
                continue

            # New closed candle detected → process patterns
            last_candle_id = candle_id

            sig = analyze(df)
            current_setup = get_setup_id(sig)

            # No valid pattern (no bounce/breakout) → reset & skip
            if current_setup is None:
                last_setup_id = None
                continue

            # Same setup as last time → avoid duplicate for same move
            if current_setup == last_setup_id:
                continue

            # Cooldown guard
            if is_in_cooldown():
                print(f"{datetime.now(timezone.utc)} - In cooldown, setup ignored: {current_setup}")
                continue

            # All good → send signal
            last = df.iloc[-1]
            text = format_signal(PAIR, TF_LABEL, last, sig)

            if text:
                print(
                    f"{datetime.now(timezone.utc)} - Sending signal: {current_setup} | "
                    f"close={last.close:.2f} RSI={last.RSI:.2f} "
                    f"K={last['%K']:.2f} D={last['%D']:.2f} ATR={last.ATR:.2f}"
                )
                send_telegram_message(text)
                last_setup_id = current_setup
                last_signal_ts = datetime.now(timezone.utc)

        except Exception as e:
            print("Bot loop error:", e)
            time.sleep(5)


@app.get("/")
def root():
    return {"status": "ok", "message": "XAUUSD bot running (candle-driven bounce/breakout only)"}


@app.on_event("startup")
def start_bot_thread():
    t = threading.Thread(target=bot_loop, daemon=True)
    t.start()
    print("🚀 Bot loop started in background thread.")
