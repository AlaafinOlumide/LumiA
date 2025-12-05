import os
import time
import threading
import requests
import pandas as pd
from fastapi import FastAPI
from datetime import datetime
import numpy as np

# ================================
# ENV VARIABLES
# ================================

TD_API_KEY = os.getenv("TD_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

PAIR = "XAU/USD"
INTERVAL = "5min"
LOOP_SLEEP_SECONDS = 35  # safe for TwelveData free tier (avoid 800-limit exhaustion)
COOLDOWN_MINUTES = 20    # signal lock after firing


# ================================
# FASTAPI APP
# ================================

app = FastAPI()


@app.get("/")
def home():
    return {"status": "running", "message": "XAUUSD strategy bot alive"}


# ================================
# TELEGRAM SEND FUNCTION
# ================================

def send_telegram(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message})
    except Exception as e:
        print("Telegram error:", e)


# ================================
# TWELVEDATA FETCH
# ================================

def fetch_5m():
    url = (
        f"https://api.twelvedata.com/time_series?"
        f"symbol={PAIR.replace('/', '')}&interval={INTERVAL}&apikey={TD_API_KEY}&outputsize=400"
    )
    r = requests.get(url).json()
    if "values" not in r:
        raise Exception(f"TwelveData error: {r}")

    df = pd.DataFrame(r["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").set_index("datetime")

    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
    return df


# ================================
# INDICATORS
# ================================

def calc_rsi(series, length=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(length).mean()
    avg_loss = loss.rolling(length).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calc_stoch(df, length=14):
    low_min = df["low"].rolling(length).min()
    high_max = df["high"].rolling(length).max()
    k = (df["close"] - low_min) / (high_max - low_min) * 100
    d = k.rolling(3).mean()
    return k, d


def calc_atr(df, length=14):
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(length).mean()


def calc_adx(df, length=14):
    # True range
    h = df["high"]
    l = df["low"]
    c = df["close"]

    plus_dm = (h.diff() > l.diff()) & (h.diff() > 0)
    minus_dm = (l.diff() > h.diff()) & (l.diff() > 0)

    tr_components = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1)
    tr = tr_components.max(axis=1)

    atr = tr.rolling(length).mean()

    pdi = 100 * (plus_dm.rolling(length).sum() / atr)
    mdi = 100 * (minus_dm.rolling(length).sum() / atr)

    dx = (abs(pdi - mdi) / (pdi + mdi)) * 100
    return dx.rolling(length).mean()


# ================================
# REGIME DETECTION (TREND / RANGE)
# ================================

def detect_regime(df_5m):
    df_1h = df_5m.resample("1h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()

    adx = calc_adx(df_1h)
    last_adx = float(adx.iloc[-1])

    if last_adx >= 30:
        # Trend mode
        prev = df_1h["close"].iloc[-3]
        last = df_1h["close"].iloc[-1]
        regime = "BULL" if last > prev else "BEAR"
    else:
        regime = "RANGE"

    return regime, last_adx


# ================================
# SIGNAL GENERATION FROM 5M
# ================================

def detect_5m_signal(df_5m, regime):
    rsi = calc_rsi(df_5m["close"])
    k, d = calc_stoch(df_5m)
    atr = calc_atr(df_5m)

    close = df_5m["close"].iloc[-1]
    prev_close = df_5m["close"].iloc[-2]

    # Bollinger bands
    mid = df_5m["close"].rolling(20).mean().iloc[-1]
    std = df_5m["close"].rolling(20).std().iloc[-1]
    upper = mid + 2 * std
    lower = mid - 2 * std

    last_rsi = float(rsi.iloc[-1])
    last_k = float(k.iloc[-1])
    last_d = float(d.iloc[-1])
    last_atr = float(atr.iloc[-1])

    # -------------------------------
    # TREND FOLLOWING LOGIC
    # -------------------------------
    if regime == "BULL":
        if close > mid and last_rsi > 45 and last_k > last_d and close > prev_close:
            return "Trend_Bounce_Buy", close, last_rsi, last_k, last_d, last_atr

    if regime == "BEAR":
        if close < mid and last_rsi < 55 and last_k < last_d and close < prev_close:
            return "Trend_Bounce_Sell", close, last_rsi, last_k, last_d, last_atr

    # -------------------------------
    # RANGE LOGIC
    # -------------------------------
    if regime == "RANGE":
        # Oversold bounce
        if close <= lower and last_rsi < 25 and last_k > last_d:
            return "Range_Bounce_Buy", close, last_rsi, last_k, last_d, last_atr

        # Overbought rejection
        if close >= upper and last_rsi > 75 and last_k < last_d:
            return "Range_Bounce_Sell", close, last_rsi, last_k, last_d, last_atr

    # -------------------------------
    # BREAKOUT (allowed only in TREND)
    # -------------------------------
    if regime == "BULL":
        if close > upper and last_rsi > 60:
            return "Breakout_Buy", close, last_rsi, last_k, last_d, last_atr

    if regime == "BEAR":
        if close < lower and last_rsi < 40:
            return "Breakout_Sell", close, last_rsi, last_k, last_d, last_atr

    return None, None, None, None, None, None


# ================================
# BACKGROUND BOT LOOP
# ================================

last_signal_time = None

def bot_loop():
    global last_signal_time

    print("🚀 Bot loop started")

    while True:
        try:
            df = fetch_5m()

            regime, adx = detect_regime(df)
            print(f"{datetime.utcnow()} - Regime: {regime} (ADX={adx:.2f})")

            # Cooldown
            if last_signal_time:
                diff = (datetime.utcnow() - last_signal_time).total_seconds() / 60
                if diff < COOLDOWN_MINUTES:
                    time.sleep(LOOP_SLEEP_SECONDS)
                    continue

            sig, close, rsi, k, d, atr = detect_5m_signal(df, regime)

            if sig:
                msg = (
                    f"🔔 SIGNAL: {sig}\n"
                    f"Regime: {regime}\n"
                    f"Price: {close}\n"
                    f"RSI: {rsi:.2f}\n"
                    f"K/D: {k:.2f}/{d:.2f}\n"
                    f"ATR: {atr:.2f}\n"
                    f"Time: {datetime.utcnow()}"
                )
                send_telegram(msg)
                last_signal_time = datetime.utcnow()
                print("Sent:", msg)

        except Exception as e:
            print("Bot error:", e)

        time.sleep(LOOP_SLEEP_SECONDS)


# ================================
# START THREAD
# ================================

threading.Thread(target=bot_loop, daemon=True).start()
