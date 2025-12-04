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
INTERVAL_1H = "1h"

COOLDOWN_MINUTES = 5        # min time between any two signals
RANGE_ATR_MULT = 0.5        # how close to EMA50(1H) counts as "range"
ADX_TREND_THRESHOLD = 20.0  # ADX(1H) above this => trending

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


def detect_trend_1h() -> dict:
    """
    Determine 1H regime using EMA50, ATR14, and ADX14.

    Returns dict:
      {
        "regime": "BULL" | "BEAR" | "RANGE",
        "ema": float,
        "atr": float,
        "adx": float,
        "close": float
      }
    """

    global last_trend_1h

    df_1h = get_xauusd_data(interval=INTERVAL_1H, outputsize=200)

    # EMA50 on close
    df_1h["EMA50"] = df_1h["close"].ewm(span=50, adjust=False).mean()

    # ATR(14) on 1H
    high_low = df_1h["high"] - df_1h["low"]
    high_close = (df_1h["high"] - df_1h["close"].shift()).abs()
    low_close = (df_1h["low"] - df_1h["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df_1h["ATR"] = tr.rolling(14).mean()

    # ADX(14) approx on 1H
    high_diff = df_1h["high"].diff()
    low_diff = df_1h["low"].diff()

    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0.0)
    minus_dm = (-low_diff).where((low_diff > high_diff) & (low_diff < 0), 0.0)

    tr14 = tr.rolling(14).mean()
    plus_dm14 = plus_dm.rolling(14).mean()
    minus_dm14 = minus_dm.rolling(14).mean()

    plus_di = 100 * (plus_dm14 / tr14)
    minus_di = 100 * (minus_dm14 / tr14)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    df_1h["ADX"] = dx.rolling(14).mean()

    last_row = df_1h.iloc[-1]

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
            # Poll every ~30 seconds; only act on new closed 5m candles
            time.sleep(30)

            now = datetime.now(timezone.utc)

            # 1️⃣ Higher timeframe regime (1H)
            ht = detect_trend_1h()
            regime = ht["regime"]   # "BULL", "BEAR", "RANGE"

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

            # Same candle as last check → skip
            if last_candle_id_5m == candle_id:
                continue

            last_candle_id_5m = candle_id

            # 3️⃣ Run 5m bounce/breakout strategy
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
                    f"Higher timeframe (1H): {regime}\n"
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
            print("Bot loop error:", e)
            time.sleep(5)


# ---------------- FastAPI ----------------

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "XAUUSD bot running (5M bounce/breakout aligned with 1H regime: BULL/BEAR/RANGE using EMA50 + ATR + ADX)",
    }


@app.on_event("startup")
def start_bot_thread():
    t = threading.Thread(target=bot_loop, daemon=True)
    t.start()
    print("🚀 Bot loop started in background thread.")