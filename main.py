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

# ---------------- Config ----------------

PAIR = "XAUUSD"
TF_LABEL = "5M"

INTERVAL_5M = "5min"

SLEEP_SECONDS = 120          # 🔹 1 API call every 2 minutes ≈ 720 calls/day
COOLDOWN_MINUTES = 5         # min time between any two signals
RANGE_ATR_MULT = 0.5         # how close to EMA50(1H) counts as "range"
ADX_TREND_THRESHOLD = 20.0   # ADX(1H) above this => trending

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if TELEGRAM_BOT_TOKEN is None or TELEGRAM_CHAT_ID is None:
    print("⚠️ TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set. Telegram sending will be skipped.")

# ---------------- State ----------------

last_setup_id: str | None = None
last_signal_ts: datetime | None = None
last_candle_id_5m: str | None = None
last_trend_1h: str | None = None   # "BULL", "BEAR", "RANGE"


# ---------------- Helpers ----------------

def get_setup_id(sig: dict) -> str | None:
    """Combine mode+direction into a simple ID."""
    if not sig.get("direction") or not sig.get("mode"):
        return None
    return f"{sig['mode']}_{sig['direction']}"


def is_in_cooldown() -> bool:
    """Check time-based cooldown between any two signals."""
    global last_signal_ts
    if last_signal_ts is None:
        return False
    return datetime.now(timezone.utc) < last_signal_ts + timedelta(minutes=COOLDOWN_MINUTES)


def send_telegram_message(text: str):
    """Send plain text to your Telegram chat."""
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


def compute_1h_regime_from_5m(df_5m: pd.DataFrame) -> dict:
    """
    Build 1H candles from 5M data and determine regime using EMA50, ATR14, ADX14.

    Returns:
      {
        "regime": "BULL" | "BEAR" | "RANGE",
        "ema": float,
        "atr": float,
        "adx": float,
        "close": float
      }
    """
    global last_trend_1h

    # Ensure datetime index
    tmp = df_5m.copy()
    if "datetime" in tmp.columns:
        tmp["datetime"] = pd.to_datetime(tmp["datetime"])
        tmp = tmp.set_index("datetime")
    else:
        tmp.index = pd.to_datetime(tmp.index)

    # Resample to 1H OHLC
    ohlc_1h = tmp.resample("1H").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        }
    ).dropna()

    # EMA50 on close
    ohlc_1h["EMA50"] = ohlc_1h["close"].ewm(span=50, adjust=False).mean()

    # ATR(14) on 1H
    high_low = ohlc_1h["high"] - ohlc_1h["low"]
    high_close = (ohlc_1h["high"] - ohlc_1h["close"].shift()).abs()
    low_close = (ohlc_1h["low"] - ohlc_1h["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    ohlc_1h["ATR"] = tr.rolling(14).mean()

    # ADX(14) approx on 1H
    high_diff = ohlc_1h["high"].diff()
    low_diff = ohlc_1h["low"].diff()

    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0.0)
    minus_dm = (-low_diff).where((low_diff > high_diff) & (low_diff < 0), 0.0)

    tr14 = tr.rolling(14).mean()
    plus_dm14 = plus_dm.rolling(14).mean()
    minus_dm14 = minus_dm.rolling(14).mean()

    plus_di = 100 * (plus_dm14 / tr14)
    minus_di = 100 * (minus_dm14 / tr14)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    ohlc_1h["ADX"] = dx.rolling(14).mean()

    last_row = ohlc_1h.iloc[-1]

    ema = float(last_row["EMA50"])
    atr = float(last_row["ATR"]) if not pd.isna(last_row["ATR"]) else 0.0
    adx = float(last_row["ADX"]) if not pd.isna(last_row["ADX"]) else 0.0
    close = float(last_row["close"])

    # Decide regime
    if atr > 0 and abs(close - ema) < RANGE_ATR_MULT * atr:
        regime = "RANGE"
    else:
        if adx >= ADX_TREND_THRESHOLD:
            regime = "BULL" if close > ema else "BEAR"
        else:
            regime = "RANGE"

    if regime != last_trend_1h:
        print(f"{datetime.now(timezone.utc)} - 1H regime changed to: {regime} (ADX={adx:.2f})")

    last_trend_1h = regime

    return {
        "regime": regime,
        "ema": ema,
        "atr": atr,
        "adx": adx,
        "close": close,
    }


# ---------------- Bot Loop ----------------

def bot_loop():
    global last_setup_id, last_signal_ts, last_candle_id_5m

    while True:
        try:
            # 🔁 One iteration every SLEEP_SECONDS
            time.sleep(SLEEP_SECONDS)
            now = datetime.now(timezone.utc)

            # 1️⃣ Get 5m data ONCE (used for both regime & signals)
            df_5m = get_xauusd_data(interval=INTERVAL_5M, outputsize=300)
            df_5m = df_5m.sort_values("datetime") if "datetime" in df_5m.columns else df_5m.sort_index()

            # Identify last closed 5m candle
            if "datetime" in df_5m.columns:
                candle_id = str(df_5m["datetime"].iloc[-1])
            else:
                candle_id = str(df_5m.index[-1])

            # Same candle as last check → skip
            if last_candle_id_5m == candle_id:
                continue

            last_candle_id_5m = candle_id

            # 2️⃣ Compute higher timeframe regime from 5m data
            ht = compute_1h_regime_from_5m(df_5m)
            regime = ht["regime"]   # "BULL", "BEAR", "RANGE"

            # 3️⃣ Compute 5m indicators & run strategy
            df_5m = bollinger_bands(df_5m)
            df_5m = rsi(df_5m)
            df_5m = stochastic(df_5m)
            df_5m = atr(df_5m)

            sig = analyze(df_5m)
            current_setup = get_setup_id(sig)
            direction = sig.get("direction")   # "BUY" or "SELL"
            mode = sig.get("mode")             # "Reversal" or "Breakout"

            # No valid setup on this candle
            if current_setup is None:
                last_setup_id = None
                continue

            # 4️⃣ Filter by 1H regime
            # BULL → only BUY (Reversal + Breakout)
            if regime == "BULL" and direction == "SELL":
                print(f"{now} - Ignored {current_setup}: counter to 1H BULL regime")
                continue

            # BEAR → only SELL (Reversal + Breakout)
            if regime == "BEAR" and direction == "BUY":
                print(f"{now} - Ignored {current_setup}: counter to 1H BEAR regime")
                continue

            # RANGE → allow only Reversal (bounces), block Breakouts
            if regime == "RANGE" and mode == "Breakout":
                print(f"{now} - Ignored {current_setup}: Breakout during RANGE regime")
                continue

            # 5️⃣ Avoid duplicate same setup
            if current_setup == last_setup_id:
                continue

            # 6️⃣ Global cooldown
            if is_in_cooldown():
                print(f"{now} - In cooldown, setup ignored: {current_setup}")
                continue

            # 7️⃣ All checks passed → send signal
            last = df_5m.iloc[-1]
            base_text = format_signal(PAIR, TF_LABEL, last, sig)

            if base_text:
                ht_line = (
                    f"Higher timeframe (1H from 5M): {regime}\n"
                    f"ADX: {ht['adx']:.2f} | EMA50: {ht['ema']:.2f} | Price: {ht['close']:.2f}"
                )
                full_text = f"{base_text}\n\n{ht_line}"

                print(
                    f"{now} - Sending signal: {current_setup} "
                    f"(Regime: {regime}) | close={last.close:.2f} "
                    f"RSI={last.RSI:.2f} K={last['%K']:.2f} D={last['%D']:.2f} ATR={last.ATR:.2f}"
                )

                send_telegram_message(full_text)
                last_setup_id = current_setup
                last_signal_ts = now

        except Exception as e:
            msg = str(e)
            print("Bot loop error:", msg)

            # 🔴 If we hit the TwelveData daily limit (429), back off heavily
            if "429" in msg or "run out of API credits" in msg:
                print("Hit TwelveData daily limit. Backing off for 1 hour to avoid spamming the API.")
                time.sleep(3600)  # sleep 1 hour
            else:
                time.sleep(10)    # small backoff for other errors


# ---------------- FastAPI ----------------

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "XAUUSD bot running (5M bounce/breakout aligned with 1H regime from 5M: BULL/BEAR/RANGE using EMA50 + ATR + ADX, API-safe)",
    }


@app.on_event("startup")
def start_bot_thread():
    t = threading.Thread(target=bot_loop, daemon=True)
    t.start()
    print("🚀 Bot loop started in background thread.")
