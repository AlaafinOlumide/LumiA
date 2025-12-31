# main.py
from __future__ import annotations

import time
import logging
import datetime as dt
from typing import Optional

import pandas as pd

from config import load_settings
from data_fetcher import fetch_m5_ohlcv_twelvedata
from telegram_client import TelegramClient
from high_impact_news import has_high_impact_news_near

import indicators
from strategy import (
    active_session,
    detect_trend_h1,
    trigger_signal_m5,
)

from signal_formatter import build_signal_message

logger = logging.getLogger("xauusd_bot")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    tmp = df.copy()
    tmp["datetime"] = pd.to_datetime(tmp["datetime"], utc=True)
    tmp = tmp.set_index("datetime")

    agg = tmp.resample(rule).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna()

    return agg.reset_index()


def market_state_from_adx(adx_h1: float) -> str:
    return "TRENDING" if adx_h1 >= 20 else "RANGING"


def trend_strength_from_adx(adx_h1: float) -> str:
    if adx_h1 >= 35:
        return "Very Strong"
    if adx_h1 >= 25:
        return "Strong"
    if adx_h1 >= 20:
        return "Moderate"
    if adx_h1 >= 15:
        return "Weak"
    return "Very Weak"


def in_cooldown(now_utc: dt.datetime, last_signal_time: Optional[dt.datetime], cooldown_minutes: int) -> bool:
    if not last_signal_time:
        return False
    return (now_utc - last_signal_time).total_seconds() < (cooldown_minutes * 60)


def main():
    settings = load_settings()

    symbol_td = settings.xau_symbol_td
    symbol_label = "XAUUSD"

    telegram = TelegramClient(settings.telegram_bot_token, settings.telegram_chat_id)

    logger.info("Starting XAUUSD bot with ASIA(00-02) + LONDON_NY(07-20) session gating.")

    last_fetch_ts: float = 0.0
    cached_m5: Optional[pd.DataFrame] = None

    last_closed_m5_time: Optional[pd.Timestamp] = None
    last_signal_time: Optional[dt.datetime] = None
    last_signal_dir: Optional[str] = None

    while True:
        now_utc = dt.datetime.now(dt.timezone.utc)

        session = active_session(
            now_utc,
            enable_asia=settings.enable_asia_session,
            trade_weekends=settings.trade_weekends,
        )

        if not session:
            logger.info("Outside trading sessions (ASIA 00-02, LONDON_NY 07-20). Sleeping %ss...", settings.sleep_seconds)
            time.sleep(settings.sleep_seconds)
            continue

        # Cooldown protocol
        if in_cooldown(now_utc, last_signal_time, settings.cooldown_minutes):
            if settings.cooldown_same_direction_only and last_signal_dir:
                # allow opposite direction during cooldown
                pass
            else:
                logger.info("Cooldown active (%sm). Skipping this candle.", settings.cooldown_minutes)
                time.sleep(settings.sleep_seconds)
                continue

        need_fetch = (time.time() - last_fetch_ts) >= settings.fetch_interval_seconds or cached_m5 is None
        if need_fetch:
            logger.info("Fetching fresh M5 OHLCV data from Twelve Data...")
            cached_m5 = fetch_m5_ohlcv_twelvedata(symbol=symbol_td, api_key=settings.twelvedata_api_key)
            last_fetch_ts = time.time()
        else:
            logger.info("Using cached M5 OHLCV data.")

        if cached_m5 is None or len(cached_m5) < 300:
            logger.warning("Not enough M5 data yet. Sleeping %ss...", settings.sleep_seconds)
            time.sleep(settings.sleep_seconds)
            continue

        # Evaluate ONLY on new CLOSED candle
        current_last_time = cached_m5["datetime"].iloc[-1]
        if last_closed_m5_time is not None and current_last_time == last_closed_m5_time:
            logger.info("No new M5 candle closed yet (last=%s). Sleeping %ss...", last_closed_m5_time, settings.sleep_seconds)
            time.sleep(settings.sleep_seconds)
            continue

        last_closed_m5_time = current_last_time
        logger.info("New M5 candle detected: %s — evaluating signal... session=%s", last_closed_m5_time, session)

        # Resample M15 / H1
        m15_df = resample_ohlc(cached_m5, "15min")
        h1_df = resample_ohlc(cached_m5, "1h")

        if len(h1_df) < 120 or len(m15_df) < 120:
            logger.info("Not enough data after resampling, sleeping %ss...", settings.sleep_seconds)
            time.sleep(settings.sleep_seconds)
            continue

        # ADX(H1)
        adx_h1_series, _, _ = indicators.adx(h1_df["high"], h1_df["low"], h1_df["close"], 14)
        adx_h1 = float(adx_h1_series.iloc[-1]) if pd.notna(adx_h1_series.iloc[-1]) else 0.0

        # Trend detection
        trend_dir = detect_trend_h1(h1_df)
        trend_source = "H1"

        if trend_dir is None:
            logger.info("No clear H1 trend. Skipping (prevents noisy night scalps).")
            time.sleep(settings.sleep_seconds)
            continue

        # News flag
        try:
            high_news = has_high_impact_news_near(symbol_label, now_utc, window_minutes=settings.news_window_minutes)
        except Exception:
            high_news = False

        # Trigger
        signal = trigger_signal_m5(
            m5_df=cached_m5,
            m15_df=m15_df,
            h1_df=h1_df,
            trend_dir=trend_dir,
            high_news=high_news,
            min_score=settings.min_entry_score,
            session=session,
            asia_extra_buffer=settings.asia_extra_score_buffer,
            tp1_rr=settings.tp1_rr,
            tp2_rr=settings.tp2_rr,
            asia_tp1_rr=settings.asia_tp1_rr,
            sl_atr_mult=settings.sl_atr_mult,
        )

        if not signal:
            logger.info("No signal on candle close (session=%s). Sleeping %ss.", session, settings.sleep_seconds)
            time.sleep(settings.sleep_seconds)
            continue

        # Market state labels
        market_state = market_state_from_adx(adx_h1)
        trend_strength = trend_strength_from_adx(adx_h1)

        # Build Telegram message (your formatter decides exact look)
        msg = build_signal_message(
            symbol_label=symbol_label,
            signal=signal,
            trend_label=trend_dir,
            trend_source=trend_source,
            session_window=("00:00-02:00" if session == "ASIA" else "07:00-20:00"),
            high_news=high_news,
            market_state=market_state,
            market_regime_text=signal.extra.get("tp_mode", ""),
            adx_h1=adx_h1,
            trend_strength=trend_strength,
            confidence_score=int(50 + signal.extra.get("entry_score", 0) * 5),
            confidence_text=("High" if signal.extra.get("entry_score", 0) >= 8 else "Moderate"),
        )

        telegram.send_message(msg)

        last_signal_time = now_utc
        last_signal_dir = signal.direction

        logger.info(
            "Signal sent. session=%s dir=%s score=%s tp_mode=%s",
            session,
            signal.direction,
            signal.extra.get("entry_score"),
            signal.extra.get("tp_mode"),
        )

        time.sleep(settings.sleep_seconds)


if __name__ == "__main__":
    main()