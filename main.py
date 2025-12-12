import time
import logging
import datetime as dt

from config import load_settings
from data_fetcher import fetch_m5_ohlcv_twelvedata
from strategy import (
    detect_trend_h1,
    detect_trend_m15_direction,
    confirm_trend_m15,
    trigger_signal_m5,
    is_within_sessions,
)
from telegram_client import TelegramClient
from cooldown import CooldownManager
from signal_formatter import format_signal_message
from indicators import atr, adx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("xauusd_bot")


def main():
    settings = load_settings()
    tg = TelegramClient(settings.telegram_bot_token, settings.telegram_chat_id)
    cooldown = CooldownManager()

    last_m5_candle = None

    logger.info("Starting XAUUSD bot (H1 primary, M15 fallback) + NEW M5 candle detection.")

    while True:
        try:
            df_m5 = fetch_m5_ohlcv_twelvedata(settings.xau_symbol_td, settings.twelvedata_api_key)

            current_candle_time = df_m5["datetime"].iloc[-1]
            if last_m5_candle == current_candle_time:
                time.sleep(60)
                continue

            last_m5_candle = current_candle_time
            now = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)

            if not is_within_sessions(
                now,
                settings.session_1_start,
                settings.session_1_end,
                settings.session_2_start,
                settings.session_2_end,
            ):
                time.sleep(60)
                continue

            # --- Resample
            h1 = df_m5.resample("1H", on="datetime").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
            ).dropna()

            m15 = df_m5.resample("15T", on="datetime").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
            ).dropna()

            # --- Trend logic
            trend = detect_trend_h1(h1)
            trend_source = "H1"

            if trend is None:
                trend = detect_trend_m15_direction(m15)
                trend_source = "M15"

            if trend is None:
                continue

            if trend_source == "H1" and not confirm_trend_m15(m15, trend):
                continue

            # --- Trigger
            signal = trigger_signal_m5(df_m5, trend)
            if not signal:
                continue

            setup_type = signal.extra["setup_type"]

            if not cooldown.can_fire(setup_type, signal.direction, now):
                continue

            # --- Risk model (ATR-based)
            atr_h1 = atr(h1["high"], h1["low"], h1["close"], 14).iloc[-1]
            sl = signal.price - atr_h1 if signal.direction == "LONG" else signal.price + atr_h1
            tp1 = signal.price + atr_h1 * 1.5 if signal.direction == "LONG" else signal.price - atr_h1 * 1.5
            tp2 = signal.price + atr_h1 * 2.5 if signal.direction == "LONG" else signal.price - atr_h1 * 2.5

            # --- Market diagnostics
            h1_adx = adx(h1["high"], h1["low"], h1["close"], 14)[0].iloc[-1]
            market_state = "TRENDING" if h1_adx >= 20 else "RANGING"
            trend_strength = "Strong" if h1_adx >= 25 else "Weak"
            market_regime = "High Volatility" if atr_h1 > h1["close"].pct_change().rolling(20).std().mean() else "Normal"

            confidence = 75 if setup_type.startswith("PULLBACK") else 70

            message = format_signal_message(
                signal=signal,
                sl=sl,
                tp1=tp1,
                tp2=tp2,
                confidence=confidence,
                trend_bias=trend,
                trend_source=trend_source,
                session="07:00-20:00",
                market_state=market_state,
                trend_strength=trend_strength,
                market_regime=market_regime,
                atr_h1=atr_h1,
                h1_adx=h1_adx,
                news_ok=True,
            )

            tg.send_message(message)
            cooldown.register(setup_type, signal.direction, now)

            time.sleep(60)

        except Exception as e:
            logger.exception("Runtime error: %s", e)
            time.sleep(60)


if __name__ == "__main__":
    main()
