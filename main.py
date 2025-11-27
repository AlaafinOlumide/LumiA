
from fastapi import FastAPI, Request
import threading
import time
from datetime import datetime, timedelta, timezone
import os
import requests

from data_provider import get_xauusd_data
from indicators import bollinger_bands, rsi, stochastic, atr
from strategy import analyze
from format_signal import format_signal

app = FastAPI()

PAIR = "XAUUSD"
TF_LABEL = "5M"
INTERVAL = "5min"          # TwelveData interval
COOLDOWN_MINUTES = 15

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if TELEGRAM_BOT_TOKEN is None or TELEGRAM_CHAT_ID is None:
    print("⚠️ TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set. Telegram sending will be skipped.")

# Global state
last_setup_id = None
last_signal_ts = None

tv_last_symbol = None
tv_last_direction = None
tv_last_mode = None
tv_last_ts = None


def get_setup_id(sig: dict):
    if not sig.get("direction") or not sig.get("mode"):
        return None
    return f"{sig['mode']}_{sig['direction']}"


def wait_for_next_5min_slot():
    """Block until the clock hits HH:00, :05, :10, ..., :55 at second == 0 (UTC)."""
    while True:
        now = datetime.now(timezone.utc)
        if now.minute % 5 == 0 and now.second == 0:
            break

        minutes_to_add = (5 - (now.minute % 5)) % 5
        target = now.replace(second=0, microsecond=0) + timedelta(minutes=minutes_to_add)
        if target <= now:
            target += timedelta(minutes=5)
        sleep_seconds = (target - now).total_seconds()
        time.sleep(sleep_seconds)


def is_in_cooldown() -> bool:
    global last_signal_ts
    if last_signal_ts is None:
        return False
    return datetime.now(timezone.utc) < last_signal_ts + timedelta(minutes=COOLDOWN_MINUTES)


def tv_confirms_signal(sig: dict) -> bool:
    """Check if TradingView has recently sent a matching signal."""
    global tv_last_symbol, tv_last_direction, tv_last_mode, tv_last_ts

    if tv_last_ts is None:
        return False

    # Only accept confirmations not too old (<= 10 mins)
    if datetime.now(timezone.utc) - tv_last_ts > timedelta(minutes=10):
        return False

    if tv_last_symbol != PAIR:
        return False

    if tv_last_direction != sig.get("direction"):
        return False

    if tv_last_mode != sig.get("mode"):
        return False

    return True


def send_telegram_message(text: str):
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
    global last_setup_id, last_signal_ts

    while True:
        try:
            wait_for_next_5min_slot()

            df = get_xauusd_data(interval=INTERVAL, outputsize=200)

            df = bollinger_bands(df)
            df = rsi(df)
            df = stochastic(df)
            df = atr(df)

            sig = analyze(df)
            current_setup = get_setup_id(sig)

            # CASE 1: no valid setup -> reset, so future setups can fire
            if current_setup is None:
                last_setup_id = None
                continue

            # CASE 2: same setup as last time -> no spam
            if current_setup == last_setup_id:
                continue

            # CASE 3: new setup candidate
            if is_in_cooldown():
                print(f"{datetime.now(timezone.utc)} - In cooldown, new setup ignored.")
                continue

            if not tv_confirms_signal(sig):
                print(f"{datetime.now(timezone.utc)} - Setup found but no matching TV confirmation.")
                continue

            # All checks passed -> send signal
            last = df.iloc[-1]
            text = format_signal(PAIR, TF_LABEL, last, sig)
            if text:
                send_telegram_message(text)
                last_setup_id = current_setup
                last_signal_ts = datetime.now(timezone.utc)
                print(f"{datetime.now(timezone.utc)} - Signal sent: {current_setup}")

        except Exception as e:
            print("Bot loop error:", e)
            # short pause before retry
            time.sleep(5)


@app.post("/tv-webhook")
async def tv_webhook(req: Request):
    """Receive TradingView alerts as JSON and store the latest confirmation."""
    global tv_last_symbol, tv_last_direction, tv_last_mode, tv_last_ts

    data = await req.json()
    tv_last_symbol = data.get("symbol")
    tv_last_direction = data.get("direction")
    tv_last_mode = data.get("mode")
    tv_last_ts = datetime.now(timezone.utc)

    print("Received TV webhook:", data)
    return {"status": "ok"}


@app.get("/")
def root():
    return {"status": "ok", "message": "XAUUSD bot running"}


@app.on_event("startup")
def start_bot_thread():
    t = threading.Thread(target=bot_loop, daemon=True)
    t.start()
    print("🚀 Bot loop started in background thread.")
