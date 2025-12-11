import time
import logging
import datetime as dt
from typing import Optional

import pandas as pd

from config import load_settings, Settings
from telegram_client import TelegramClient
from data_fetcher import fetch_m5_ohlcv_hybrid
from strategy import (
    detect_trend_h1,
    confirm_trend_m15,
    trigger_signal_m5,
    is_within_sessions,
)
from data_logger import log_signal
from high_impact_news import has_high_impact_news_near

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("xauusd_bot")

# Fetch 5m data at most once every 120 seconds (2 minutes)
FETCH_INTERVAL_SECONDS = 120


def _risk_tag_from_adx(adx_m5: float) -> str:
    """
    Classify the trade idea as SCALP vs SWING based on M5 ADX strength.
    """
    if adx_m5 >= 30:
        return "SWING"
    return "SCALP"


def build_signal_message(
    symbol_label: str,
    signal,
    trend_h1,
    session_window: str,
    high_news: bool,
) -> str:
    arrow = "🟢 BUY" if signal.direction == "LONG" else "🔴 SELL"
    adx_m5 = signal.extra["adx_m5"]
    risk_tag = _risk_tag_from_adx(adx_m5)

    if high_news:
        news_tag = "⚠️ *HIGH-IMPACT NEWS NEARBY* — expect extra volatility."
    else:
        news_tag = "ℹ️ No high-impact news flag near this time."

    text = (
        f"*XAUUSD Signal*  `[{risk_tag}]`\n"
        f"{arrow}  `{symbol_label}` at *{signal.price:.2f}*\n"
        f"Time (UTC): `{signal.time.isoformat()}`\n"
        f"Trend (H1): *{trend_h1}*\n"
        f"Session: `{session_window}`\n"
        f"Reason: {signal.reason}\n"
        f"{news_tag}\n"
        f"RSI(M5): `{signal.extra['m5_rsi']:.2f}`  |  "
        f"StochK(M5): `{signal.extra['m5_stoch_k']:.2f}`\n"
        f"ADX(M5): `{signal.extra['adx_m5']:.2f}` "
        f"(+DI: `{signal.extra['plus_di_m5']:.2f}`, "
        f"-DI: `{signal.extra['minus_di_m5']:.2f}`)\n"
        f"BB(M5): upper `{signal.extra['bb_upper']:.2f}`, "
        f"mid `{signal.extra['bb_mid']:.2f}`, "
        f"lower `{signal.extra['bb_lower']:.2f}`"
    )
    return text


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Resample a 5m OHLCV DataFrame to a higher timeframe (15m, 1h, etc.).
    Assumes df has columns: datetime, open, high, low, close, volume.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    tmp = df.copy()
    tmp = tmp.set_index("datetime")
    agg = tmp.resample(rule).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    agg = agg.dropna()
    agg = agg.reset_index()
    return agg


def main_loop():
    settings: Settings = load_settings()
    tg = TelegramClient(settings.telegram_bot_token, settings.telegram_chat_id)

    # For display in Telegram, we just call it XAUUSD
    symbol_label = "XAUUSD"
    logger.info(
        "Starting XAUUSD bot (hybrid data: yfinance primary, Twelve Data fallback)."
    )

    last_signal_time: Optional[dt.datetime] = None

    # Cache for 5m data to reduce external calls
    cached_m5_df: Optional[pd.DataFrame] = None
    last_m5_fetch_ts: Optional[float] = None

    while True:
        now_utc = dt.datetime.utcnow()

        if not is_within_sessions(
            now_utc,
            settings.session_1_start,
            settings.session_1_end,
            settings.session_2_start,
            settings.session_2_end,
        ):
            logger.info("Outside trading sessions, sleeping 60s...")
            time.sleep(60)
            continue

        try:
            # ---------- DATA FETCH / CACHE LAYER ----------
            now_ts = time.time()
            should_fetch_m5 = (
                last_m5_fetch_ts is None
                or (now_ts - last_m5_fetch_ts) >= FETCH_INTERVAL_SECONDS
            )

            if should_fetch_m5:
                logger.info("Fetching fresh M5 OHLCV data (hybrid: yfinance -> Twelve Data)...")
                m5_df = fetch_m5_ohlcv_hybrid(settings)
                cached_m5_df = m5_df
                last_m5_fetch_ts = now_ts
            else:
                if cached_m5_df is None:
                    logger.info("No cached M5 data yet, fetching hybrid source...")
                    m5_df = fetch_m5_ohlcv_hybrid(settings)
                    cached_m5_df = m5_df
                    last_m5_fetch_ts = now_ts
                else:
                    logger.info("Using cached M5 OHLCV data.")
                    m5_df = cached_m5_df

            if m5_df is None or m5_df.empty:
                logger.info("Empty M5 data, sleeping 60s.")
                time.sleep(60)
                continue

            # ---------- RESAMPLING ----------
            h1_df = resample_ohlcv(m5_df, "1H")
            m15_df = resample_ohlcv(m5_df, "15T")

            if len(h1_df) < 30 or len(m15_df) < 30 or len(m5_df) < 30:
                logger.info("Not enough data after resampling, sleeping 60s.")
                time.sleep(60)
                continue

            # ---------- STRATEGY PIPELINE ----------
            trend_h1 = detect_trend_h1(h1_df)
            if trend_h1 is None:
                logger.info("No clear H1 trend, skipping.")
                time.sleep(60)
                continue

            if not confirm_trend_m15(m15_df, trend_h1):
                logger.info("M15 does not confirm H1 trend, skipping.")
                time.sleep(60)
                continue

            signal = trigger_signal_m5(m5_df, trend_h1)
            if not signal:
                logger.info("No M5 trigger signal, sleeping 60s.")
                time.sleep(60)
                continue

            # Throttle: avoid spamming repeated signals too quickly
            if last_signal_time and (now_utc - last_signal_time).total_seconds() < 300:
                logger.info(
                    "Signal occurred too soon after previous, skipping (cooldown)."
                )
                time.sleep(60)
                continue

            # Session label (single big window: 07:00–20:00 UTC)
            hhmm = now_utc.hour * 100 + now_utc.minute
            if settings.session_1_start <= hhmm <= settings.session_1_end:
                session_window = "07:00-20:00"
            else:
                session_window = "OUTSIDE"

            high_news = has_high_impact_news_near(symbol_label, now_utc)

            msg = build_signal_message(symbol_label, signal, trend_h1, session_window, high_news)
            tg.send_message(msg)
            last_signal_time = now_utc

            # Log with ~12+ data fields
            row = {
                "symbol": symbol_label,
                "direction": signal.direction,
                "price": signal.price,
                "reason": signal.reason,
                "trend_h1": trend_h1,
                "session_window": session_window,
                "m5_rsi": signal.extra["m5_rsi"],
                "m5_stoch_k": signal.extra["m5_stoch_k"],
                "m5_stoch_d": signal.extra["m5_stoch_d"],
                "bb_upper": signal.extra["bb_upper"],
                "bb_mid": signal.extra["bb_mid"],
                "bb_lower": signal.extra["bb_lower"],
                "adx_m5": signal.extra["adx_m5"],
                "plus_di_m5": signal.extra["plus_di_m5"],
                "minus_di_m5": signal.extra["minus_di_m5"],
                "high_impact_news": high_news,
            }
            log_signal(row)

            logger.info("Signal sent and logged.")
            time.sleep(60)

        except Exception as e:
            logger.exception("Error in main loop: %s", e)
            time.sleep(60)


if __name__ == "__main__":
    main_loop()
