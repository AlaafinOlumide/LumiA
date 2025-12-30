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
    is_within_sessions,
    detect_trend_h1,
    detect_trend_m15_direction,
    confirm_trend_m15,
    trigger_signal_m5,
    market_regime,
    dynamic_tp_sl,
    compute_confidence,
    confidence_label,
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


def cooldown_minutes_for_regime(adx_h1: float, default_minutes: int) -> int:
    # PATCH: regime-based cooldown
    if adx_h1 >= 35:
        return 20
    if adx_h1 >= 25:
        return 12
    if adx_h1 >= 20:
        return 10
    return default_minutes


def main():
    settings = load_settings()

    symbol_td = settings.xau_symbol_td
    symbol_label = "XAUUSD"

    telegram = TelegramClient(settings.telegram_bot_token, settings.telegram_chat_id)

    logger.info("Starting XAUUSD bot (H1 primary, M15 fallback) + score gate + impulse continuation + liquidity filter.")

    last_fetch_ts: float = 0.0
    cached_m5: Optional[pd.DataFrame] = None

    last_closed_m5_time: Optional[pd.Timestamp] = None
    last_signal_time: Optional[dt.datetime] = None
    last_signal_dir: Optional[str] = None

    while True:
        now_utc = dt.datetime.now(dt.timezone.utc)

        # ------------------------------------------------------------------
        # PATCH #1: Hard trading window enforcement (prevents night signals)
        # ------------------------------------------------------------------
        if not (settings.hard_session_start_hour_utc <= now_utc.hour < settings.hard_session_end_hour_utc):
            logger.info("Outside HARD window %02d:00-%02d:00 UTC. Sleeping %ss...",
                        settings.hard_session_start_hour_utc, settings.hard_session_end_hour_utc, settings.sleep_seconds)
            time.sleep(settings.sleep_seconds)
            continue

        # ------------------------------------------------------------------
        # PATCH #2: Session helper (kept) + weekend filter
        # ------------------------------------------------------------------
        if not is_within_sessions(
            now_utc=now_utc,
            session_1_start=settings.session_1_start,
            session_1_end=settings.session_1_end,
            session_2_start=None,
            session_2_end=None,
            trade_weekends=settings.trade_weekends,
        ):
            logger.info("Outside session/weekend filter. Sleeping %ss...", settings.sleep_seconds)
            time.sleep(settings.sleep_seconds)
            continue

        # fetch / cache
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

        # Only evaluate on new CLOSED candle
        current_last_time = cached_m5["datetime"].iloc[-1]
        if last_closed_m5_time is not None and current_last_time == last_closed_m5_time:
            logger.info("No new M5 candle closed yet (last=%s). Sleeping %ss...", last_closed_m5_time, settings.sleep_seconds)
            time.sleep(settings.sleep_seconds)
            continue

        last_closed_m5_time = current_last_time
        logger.info("New M5 candle detected: %s — evaluating signal...", last_closed_m5_time)

        # resample
        m15_df = resample_ohlc(cached_m5, "15min")
        h1_df = resample_ohlc(cached_m5, "1h")

        if len(h1_df) < 60 or len(m15_df) < 60:
            logger.info("Not enough data after resampling, sleeping %ss...", settings.sleep_seconds)
            time.sleep(settings.sleep_seconds)
            continue

        # ADX(H1)
        adx_h1_series, _, _ = indicators.adx(h1_df["high"], h1_df["low"], h1_df["close"], period=14)
        adx_h1 = float(adx_h1_series.iloc[-1]) if pd.notna(adx_h1_series.iloc[-1]) else 0.0

        # trend detection
        h1_trend = detect_trend_h1(h1_df)
        trend_source = "H1"

        if h1_trend is None:
            m15_dir = detect_trend_m15_direction(m15_df)
            if m15_dir is None:
                logger.info("No clear H1 trend and no clear M15 direction, skipping.")
                time.sleep(settings.sleep_seconds)
                continue
            trend_dir = m15_dir
            trend_source = "M15"
        else:
            trend_dir = h1_trend
            _ = confirm_trend_m15(m15_df, trend_dir)

        # news
        try:
            high_news = has_high_impact_news_near(symbol_label, now_utc, window_minutes=settings.news_window_minutes)
        except Exception:
            high_news = False

        # ------------------------------------------------------------------
        # PATCH #3: Regime-based cooldown (stops over-firing)
        # ------------------------------------------------------------------
        cooldown_minutes = (
            cooldown_minutes_for_regime(adx_h1, settings.cooldown_minutes_default)
            if settings.use_regime_based_cooldown
            else settings.cooldown_minutes_default
        )

        if in_cooldown(now_utc, last_signal_time, cooldown_minutes):
            if settings.cooldown_same_direction_only and last_signal_dir and last_signal_dir != trend_dir:
                pass
            else:
                logger.info("Cooldown active (%sm). Skipping this candle.", cooldown_minutes)
                time.sleep(settings.sleep_seconds)
                continue

        # trigger with score gate + liquidity filter + impulse continuation
        signal = trigger_signal_m5(
            m5_df=cached_m5,
            trend_dir=trend_dir,
            m15_df=m15_df,
            h1_df=h1_df,
            high_news=high_news,
            min_score=settings.min_entry_score,
            range_filter_lookback=settings.range_filter_lookback,
            range_filter_min_ratio=settings.range_filter_min_ratio,
        )

        if not signal:
            logger.info("No M5 trigger signal on candle close, sleeping %ss.", settings.sleep_seconds)
            time.sleep(settings.sleep_seconds)
            continue

        # regime + dynamic TP/SL
        regime = market_regime(h1_df)
        dynamic_tp_sl(signal, h1_df, regime)

        # market classifiers
        market_state = market_state_from_adx(adx_h1)
        trend_strength = trend_strength_from_adx(adx_h1)

        # confidence
        setup_type = signal.extra.get("setup_type", "GENERIC")
        adx_m5 = float(signal.extra.get("adx_m5", 0.0))
        entry_score = int(signal.extra.get("entry_score", 0))

        conf = compute_confidence(
            trend_source=trend_source,
            setup_type=setup_type,
            adx_h1=adx_h1,
            adx_m5=adx_m5,
            high_news=high_news,
            entry_score=entry_score,
        )
        conf_text = confidence_label(conf)

        msg = build_signal_message(
            symbol_label=symbol_label,
            signal=signal,
            trend_label=trend_dir,
            trend_source=trend_source,
            session_window=f"{settings.hard_session_start_hour_utc:02d}:00-{settings.hard_session_end_hour_utc:02d}:00",
            high_news=high_news,
            market_state=market_state,
            market_regime_text=regime,
            adx_h1=adx_h1,
            trend_strength=trend_strength,
            confidence_score=conf,
            confidence_text=conf_text,
        )

        telegram.send_message(msg)
        last_signal_time = now_utc
        last_signal_dir = signal.direction

        logger.info(
            "Signal sent. source=%s dir=%s setup=%s score=%s conf=%s (%s)",
            trend_source, trend_dir, setup_type, entry_score, conf, conf_text
        )

        time.sleep(settings.sleep_seconds)


if __name__ == "__main__":
    main()