from fastapi import FastAPI
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
INTERVAL = "5min"
COOLDOWN_MINUTES = 15

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if TELEGRAM_BOT_TOKEN is None or TELEGRAM_CHAT_ID is None:
    print("⚠️ TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set. Telegram sending will be skipped.")

last_setup_id = None
last_signal_ts = None


def get_setup_id(sig: dict):
    if not sig.get("direction") or not sig.get("mode"):
        return None
    return f"{sig['mode']}_{sig['direction']}"


def wait_for_next_5min_slot():
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

            if current_setup is None:
                last_setup_id = None
                continue

            if current_setup == last_setup_id:
                continue

            if is_in_cooldown():
                print(f"{datetime.now(timezone.utc)} - In cooldown, new setup ignored.")
                continue

            last = df.iloc[-1]
            text = format_signal(PAIR, TF_LABEL, last, sig)
            if text:
                send_telegram_message(text)
                last_setup_id = current_setup
                last_signal_ts = datetime.now(timezone.utc)
                print(f"{datetime.now(timezone.utc)} - Signal sent: {current_setup}")

        except Exception as e:
            print("Bot loop error:", e)
            time.sleep(5)


@app.get("/")
def root():
    return {"status": "ok", "message": "XAUUSD bot running (Python-only signals)"}


@app.on_event("startup")
def start_bot_thread():
    t = threading.Thread(target=bot_loop, daemon=True)
    t.start()
    print("🚀 Bot loop started in background thread.")
