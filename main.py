import os
import time
import threading
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI

from strategy import detect_5m_signal, detect_1h_regime

# -----------------------------
# ENV VARS
# -----------------------------
API_KEY = os.getenv("TWELVEDATA_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

SYMBOL = "XAU/USD"
INTERVAL_5M = "5min"

# -----------------------------
# GLOBALS
# -----------------------------
last_signal_time = None
last_setup = None
last_candle_time = None

cached_1h_regime = None
cached_1h_timestamp = None  # last time 1H data was refreshed

app = FastAPI()


# ------------------------------------------------------
# SEND TELEGRAM MESSAGE
# ------------------------------------------------------
def send_telegram(text: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ Telegram not configured")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text})
    except Exception as e:
        print("Telegram error:", e)


# ------------------------------------------------------
# FETCH CANDLES
# ------------------------------------------------------
def fetch_twelvedata(symbol: str, interval: str, outputsize: int = 50):
    if not API_KEY:
        print("❌ TWELVEDATA_API_KEY not set")
        return None

    url = (
        f"https://api.twelvedata.com/time_series?"
        f"symbol={symbol}&interval={interval}&outputsize={outputsize}&apikey={API_KEY}"
    )
    r = requests.get(url).json()

    if "status" in r and r["status"] == "error":
        print("Bot loop error:", r)
        return None

    if "values" not in r:
        return None

    df = pd.DataFrame(r["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")
    df.set_index("datetime", inplace=True)

    df = df.astype(
        {
            "open": float,
            "high": float,
            "low": float,
            "close": float,
        }
    )
    return df


# ------------------------------------------------------
# BOT LOOP
# ------------------------------------------------------
def bot_loop():
    global last_signal_time, last_setup
    global cached_1h_regime, cached_1h_timestamp
    global last_candle_time

    print("🚀 Bot loop started")

    while True:
        try:
            now = datetime.now(timezone.utc)

            # -----------------------------
            # FETCH 5M DATA
            # -----------------------------
            df5 = fetch_twelvedata(SYMBOL, INTERVAL_5M, outputsize=30)
            if df5 is None or len(df5) < 10:
                time.sleep(20)
                continue

            last5_ts = df5.index[-1]

            # -----------------------------
            # Prevent intra-candle spam
            # -----------------------------
            if last_candle_time == last5_ts:
                time.sleep(20)
                continue
            last_candle_time = last5_ts

            # -----------------------------
            # REFRESH 1H REGIME (HOURLY)
            # -----------------------------
            if (
                cached_1h_timestamp is None
                or now - cached_1h_timestamp > timedelta(minutes=55)
            ):
                # fetch 5-minute but resample to 1H
                df1h_tmp = fetch_twelvedata(SYMBOL, INTERVAL_5M, outputsize=200)
                if df1h_tmp is not None:
                    ohlc_1h = df1h_tmp.resample("1h").agg(
                        {
                            "open": "first",
                            "high": "max",
                            "low": "min",
                            "close": "last",
                        }
                    ).dropna()

                    cached_1h_regime = detect_1h_regime(ohlc_1h)
                    cached_1h_timestamp = now
                    print(
                        f"{now} - 1H regime updated: {cached_1h_regime}"
                    )

            # -----------------------------
            # DETECT SIGNAL ON 5M
            # -----------------------------
            signal = detect_5m_signal(df5, cached_1h_regime)
            if signal is None:
                time.sleep(20)
                continue

            direction, setup_name, strength, extra = signal

            # -----------------------------
            # COOL DOWN (5 MINUTES)
            # -----------------------------
            if last_signal_time:
                if now - last_signal_time < timedelta(minutes=5):
                    print(now, "-", "Cooldown active, ignoring:", setup_name)
                    time.sleep(20)
                    continue

            # -----------------------------
            # DUPLICATE SETUP FILTER
            # -----------------------------
            if last_setup == setup_name:
                print(now, "-", "Duplicate setup ignored:", setup_name)
                time.sleep(20)
                continue

            # -----------------------------
            # SEND SIGNAL
            # -----------------------------
            msg = (
                f"📈 XAUUSD SIGNAL\n\n"
                f"Type: {setup_name} ({direction})\n"
                f"Strength: {strength}\n\n"
                f"5M Close: {df5['close'].iloc[-1]}\n"
                f"1H Regime: {cached_1h_regime}\n\n"
                f"Details:\n{extra}\n"
            )

            send_telegram(msg)
            print(now, "- Signal sent:", setup_name)

            last_signal_time = now
            last_setup = setup_name

        except Exception as e:
            print("Bot loop error:", e)

        time.sleep(20)  # 20 second loop


# ------------------------------------------------------
# START BOT ON SERVER BOOT
# ------------------------------------------------------
@app.on_event("startup")
def start_bot():
    t = threading.Thread(target=bot_loop, daemon=True)
    t.start()


# ------------------------------------------------------
# ROOT ENDPOINT
# ------------------------------------------------------
@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "XAUUSD bot running with 1H regime filters, bounce/breakout logic, and API-efficient loop.",
        "1h_regime": cached_1h_regime,
    }
