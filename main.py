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


def _pick_eval_time(df: pd.DataFrame, ignore_latest: bool) -> pd.Timestamp:
    """
    If ignore_latest=True, we evaluate on the *last fully closed* candle by using -2.
    This avoids firing on a still-forming candle some providers return as the last row.
    """
    if len(df) < 3:
        return df["datetime"].iloc[-1]
    return df["datetime"].iloc[-2] if ignore_latest else df["datetime"].iloc[-1]


def main():
    settings = load_settings()

    symbol_td = settings.xau_symbol_td
    symbol_label = "XAUUSD"

    telegram = TelegramClient(settings.telegram_bot_token, settings.telegram_chat_id)

    logger.info("Starting XAUUSD bot on Render/local.")
    logger.info(
        "Sessions UTC: ASIA=%s (%02d-%02d) | LONDON_NY (%02d-%02d) | weekends=%s",
        settings.enable_asia_session,
        settings.asia_start_hour_utc,
        settings.asia_end_hour_utc,
        settings.london_start_hour_utc,
        settings.london_end_hour_utc,
        settings.trade_weekends,
    )
    logger.info(
        "Data: outputsize=%s fetch_interval=%ss sleep=%ss ignore_latest_candle=%s",
        settings.outputsize,
        settings.fetch_interval_seconds,
        settings.sleep_seconds,
        settings.ignore_latest_candle,
    )
    logger.info(
        "Gates: min_score=%s asia_buffer=%s momentum_override=%s (ADX_M5>=%.1f)",
        settings.min_entry_score,
        settings.asia_extra_score_buffer,
        settings.enable_momentum_override,
        settings.momentum_override_adx_m5,
    )

    last_fetch_ts: float = 0.0
    cached_m5: Optional[pd.DataFrame] = None

    last_eval_m5_time: Optional[pd.Timestamp] = None
    last_signal_time: Optional[dt.datetime] = None
    last_signal_dir: Optional[str] = None

    while True:
        now_utc = dt.datetime.now(dt.timezone.utc)

        session = active_session(
            now_utc,
            enable_asia=settings.enable_asia_session,
            trade_weekends=settings.trade_weekends,
            asia_start_hour_utc=settings.asia_start_hour_utc,
            asia_end_hour_utc=settings.asia_end_hour_utc,
            london_start_hour_utc=settings.london_start_hour_utc,
            london_end_hour_utc=settings.london_end_hour_utc,
        )

        if not session:
            logger.info("Outside trading sessions. Sleeping %ss...", settings.sleep_seconds)
            time.sleep(settings.sleep_seconds)
            continue

        # Cooldown protocol
        if in_cooldown(now_utc, last_signal_time, settings.cooldown_minutes):
            if settings.cooldown_same_direction_only and last_signal_dir:
                pass  # allow opposite direction during cooldown
            else:
                logger.info("Cooldown active (%sm). Skipping this cycle.", settings.cooldown_minutes)
                time.sleep(settings.sleep_seconds)
                continue

        need_fetch = (time.time() - last_fetch_ts) >= settings.fetch_interval_seconds or cached_m5 is None
        if need_fetch:
            logger.info("Fetching fresh M5 OHLCV data from Twelve Data...")
            cached_m5 = fetch_m5_ohlcv_twelvedata(
                symbol=symbol_td,
                api_key=settings.twelvedata_api_key,
                outputsize=settings.outputsize,
            )
            last_fetch_ts = time.time()
        else:
            logger.info("Using cached M5 OHLCV data.")

        if cached_m5 is None or len(cached_m5) < 400:
            logger.warning("Not enough M5 data yet (%s rows). Sleeping %ss...", 0 if cached_m5 is None else len(cached_m5), settings.sleep_seconds)
            time.sleep(settings.sleep_seconds)
            continue

        # Evaluate ONLY on new CLOSED candle time
        eval_time = _pick_eval_time(cached_m5, settings.ignore_latest_candle)
        if last_eval_m5_time is not None and eval_time == last_eval_m5_time:
            logger.info("No new CLOSED M5 candle yet (eval_time=%s). Sleeping %ss...", last_eval_m5_time, settings.sleep_seconds)
            time.sleep(settings.sleep_seconds)
            continue

        last_eval_m5_time = eval_time
        logger.info("New CLOSED M5 candle detected: %s — evaluating signal... session=%s", last_eval_m5_time, session)

        # Use only candles up to eval_time
        m5_for_eval = cached_m5[cached_m5["datetime"] <= eval_time].copy()
        if len(m5_for_eval) < 350:
            logger.info("Not enough M5 after eval cutoff. Sleeping %ss...", settings.sleep_seconds)
            time.sleep(settings.sleep_seconds)
            continue

        # Resample M15 / H1
        m15_df = resample_ohlc(m5_for_eval, "15min")
        h1_df = resample_ohlc(m5_for_eval, "1h")

        if len(h1_df) < 120 or len(m15_df) < 120:
            logger.info("Not enough data after resampling (H1=%s M15=%s). Sleeping %ss...", len(h1_df), len(m15_df), settings.sleep_seconds)
            time.sleep(settings.sleep_seconds)
            continue

        # ADX(H1)
        adx_h1_series, _, _ = indicators.adx(h1_df["high"], h1_df["low"], h1_df["close"], 14)
        adx_h1 = float(adx_h1_series.iloc[-1]) if pd.notna(adx_h1_series.iloc[-1]) else 0.0

        # ADX(M5) for momentum override
        adx_m5_series, _, _ = indicators.adx(m5_for_eval["high"], m5_for_eval["low"], m5_for_eval["close"], 14)
        adx_m5_val = float(adx_m5_series.iloc[-1]) if pd.notna(adx_m5_series.iloc[-1]) else 0.0

        momentum_override = settings.enable_momentum_override and (adx_m5_val >= settings.momentum_override_adx_m5)

        # Trend detection (H1)
        trend_dir = detect_trend_h1(h1_df)
        trend_source = "H1"

        # If H1 trend unclear, allow momentum override to avoid missing dumps/spikes
        if trend_dir is None:
            if not momentum_override:
                logger.info("No clear H1 trend and no momentum override (ADX_M5=%.2f). Skipping.", adx_m5_val)
                time.sleep(settings.sleep_seconds)
                continue

            # Simple direction inference on last closed candle (fallback only)
            last_close = float(m5_for_eval["close"].iloc[-1])
            last_open = float(m5_for_eval["open"].iloc[-1])
            trend_dir = "SHORT" if last_close < last_open else "LONG"
            trend_source = "M5_MOMENTUM"
            logger.info("Momentum override active (ADX_M5=%.2f). Using trend_dir=%s source=%s", adx_m5_val, trend_dir, trend_source)

        # News flag
        try:
            high_news = has_high_impact_news_near(symbol_label, now_utc, window_minutes=settings.news_window_minutes)
        except Exception as e:
            logger.warning("News check failed: %s", e)
            high_news = False

        # Trigger
        signal = trigger_signal_m5(
            m5_df=m5_for_eval,
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

        msg = build_signal_message(
            symbol_label=symbol_label,
            signal=signal,
            trend_label=trend_dir,
            trend_source=trend_source,
            session_window=("00:00-03:00" if session == "ASIA" else "06:00-21:00"),
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
            "Signal sent. session=%s dir=%s score=%s tp_mode=%s trend_source=%s adx_m5=%.2f adx_h1=%.2f",
            session,
            signal.direction,
            signal.extra.get("entry_score"),
            signal.extra.get("tp_mode"),
            trend_source,
            adx_m5_val,
            adx_h1,
        )

        time.sleep(settings.sleep_seconds)


if __name__ == "__main__":
    main()