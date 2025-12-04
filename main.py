from fastapi import FastAPI
import threading
import time
from datetime import datetime, timezone, timedelta
import os
import requests
import pandas as pd

from data_provider import get_xauusd_data
from indicators import bollinger_bands, rsi, stochastic, atr
from strategy import analyze
from format_signal import format_signal

app = FastAPI()

PAIR = "XAUUSD"
TF_LABEL = "5M"
INTERVAL_5M = "5min"
INTERVAL_1H = "1h"

COOLDOWN_MINUTES = 5  # minimum time between any two signals

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if TELEGRAM_BOT_TOKEN is None or TELEGRAM_CHAT_ID is None:
    print("⚠️ TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set. Telegram sending will be skipped.")

# State
last_setup_id: str | None = None
last_signal_ts: datetime | None = None
last_candle_id_5m: str | None = None
last_trend_1h: str | None = None  # "BULL" or "BEAR"


def get_setup_id(sig: dict):
    if not sig.get("direction") or not sig.get("mode"):
        return None
    return f"{sig['mode']}_{sig['direction']}"


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


def detect_trend_1h() -> str:
    """
    Determine 1H trend using a 50-EMA on 1H closes.
    close > EMA50 => 'BULL', else 'BEAR'.
    """
    global last_trend_1h

    df_1h = get_xauusd_data(interval=INTERVAL_1H, outputsize=200)

    # 50-EMA on close
    df_1h["EMA50"] = df_1h["close"].ewm(span=50, adjust=False).mean()
    last_row = df_1h.iloc[-1]

    if pd.isna(last_row["EMA50"]):
        # fallback: if EMA not ready, keep previous trend or default to BEAR
        trend = last_trend_1h or "BEAR"
    else:
        trend = "BULL" if last_row["close"] > last_row["EMA50"] else "BEAR"

    if trend != last_trend_1h:
        print(f"{datetime.now(timezone.utc)} - 1H trend changed to: {trend}")

    last_trend_1h = trend
    return trend


def bot_loop():
    global last_setup_id, last_signal_ts, last_candle_id_5m

    while True:
        try:
            # Poll every ~30s; only act on new closed 5m candle
            time.sleep(30)

            # 1️⃣ Get higher-timeframe trend (1H)
            trend_1h = detect_trend_1h()  # 'BULL' or 'BEAR'

            # 2️⃣ Get 5m data & indicators
            df_5m = get_xauusd_data(interval=INTERVAL_5M, outputsize=200)
            df_5m = bollinger_bands(df_5m)
            df_5m = rsi(df_5m)
            df_5m = stochastic(df_5m)
            df_5m = atr(df_5m)

            # Identify last closed 5m candle
            if "datetime" in df_5m.columns:
                candle_id = str(df_5m["datetime"].iloc[-1])
            else:
                candle_id = str(df_5m.index[-1])

            # Same candle as last time → skip
            if last_candle_id_5m == candle_id:
                continue

            last_candle_id_5m = candle_id

            # 3️⃣ Run 5m bounce/breakout strategy
            sig = analyze(df_5m)
            current_setup = get_setup_id(sig)
            direction = sig.get("direction")  # 'BUY' or 'SELL'

            # No valid bounce/breakout
            if current_setup is None:
                last_setup_id = None
                continue

            # 4️⃣ Filter by 1H trend
            # If 1H is BULL, ignore SELL signals; if 1H is BEAR, ignore BUY signals.
            if trend_1h == "BULL" and direction == "SELL":
                print(f"{datetime.now(timezone.utc)} - Ignored {current_setup}: counter to 1H BULL trend")
                continue
            if trend_1h == "BEAR" and direction == "BUY":
                print(f"{datetime.now(timezone.utc)} - Ignored {current_setup}: counter to 1H BEAR trend")
                continue

            # 5️⃣ Avoid duplicate same setup
            if current_setup == last_setup_id:
                continue

            # 6️⃣ Global cooldown
            if is_in_cooldown():
                print(f"{datetime.now(timezone.utc)} - In cooldown, setup ignored: {current_setup}")
                continue

            # 7️⃣ All good → send signal
            last = df_5m.iloc[-1]
            text = format_signal(PAIR, TF_LABEL, last, sig)
            if text:
                print(
                    f"{datetime.now(timezone.utc)} - Sending signal: {current_setup} "
                    f"(1H trend: {trend_1h}) | close={last.close:.2f} "
                    f"RSI={last.RSI:.2f} K={last['%K']:.2f} D={last['%D']:.2f} ATR={last.ATR:.2f}"
                )
                send_telegram_message(text)
                last_setup_id = current_setup
                last_signal_ts = datetime.now(timezone.utc)

        except Exception as e:
            print("Bot loop error:", e)
            time.sleep(5)


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "XAUUSD bot running (5M bounce/breakout, aligned with 1H trend EMA50)",
    }


@app.on_event("startup")
def start_bot_thread():
    t = threading.Thread(target=bot_loop, daemon=True)
    t.start()
    print("🚀 Bot loop started in background thread.")