# main.py
import time
import logging
import datetime as dt
from typing import Optional, Tuple

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
)

from cooldown import in_cooldown
from trade_manager import (
    ActiveTrade,
    new_trade_id,
    check_tp_sl_hit,
    should_invalidate,
)
from data_logger import TradeJournal
from signal_formatter import build_signal_message


# -------------------------
# Logging
# -------------------------
logger = logging.getLogger("xauusd_bot")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# -------------------------
# Resampling (M5 -> M15/H1)
# -------------------------
def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    tmp = df.copy()
    if "datetime" in tmp.columns:
        tmp["datetime"] = pd.to_datetime(tmp["datetime"], utc=True)
        tmp = tmp.set_index("datetime")
    else:
        tmp.index = pd.to_datetime(tmp.index, utc=True)

    agg = (
        tmp.resample(rule)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .dropna()
    )
    return agg.reset_index()


# -------------------------
# Market open filter
# -------------------------
def is_market_open(now_utc: dt.datetime, block_weekends: bool, sunday_open_hour_utc: int) -> bool:
    """
    Blocks weekend signals.
    - Saturday: closed
    - Sunday: closed until sunday_open_hour_utc
    """
    if not block_weekends:
        return True

    wd = now_utc.weekday()  # Mon=0 ... Sun=6
    if wd == 5:  # Saturday
        return False
    if wd == 6:  # Sunday
        return now_utc.hour >= sunday_open_hour_utc
    return True


# -------------------------
# Dynamic TP/SL (ATR-based)
# -------------------------
def tp_sl_multipliers(setup_type: str) -> Tuple[float, float, float]:
    if setup_type.startswith("PULLBACK"):
        return (0.90, 1.60, 2.60)
    if setup_type.startswith("BREAKOUT_CONT"):
        return (0.55, 0.90, 1.50)
    if setup_type.startswith("BREAKOUT"):
        return (0.70, 1.20, 2.00)
    return (0.75, 1.10, 1.80)


def apply_dynamic_tp_sl(signal, h1_df: pd.DataFrame) -> None:
    if h1_df is None or len(h1_df) < 60:
        return

    atr_series = indicators.atr(h1_df["high"], h1_df["low"], h1_df["close"], period=14)
    atr_h1 = float(atr_series.iloc[-1]) if pd.notna(atr_series.iloc[-1]) else None
    if atr_h1 is None or atr_h1 <= 0:
        return

    setup_type = signal.extra.get("setup_type", "GENERIC")
    sl_mult, tp1_mult, tp2_mult = tp_sl_multipliers(setup_type)

    entry = float(signal.price)

    if signal.direction == "LONG":
        sl = entry - (atr_h1 * sl_mult)
        tp1 = entry + (atr_h1 * tp1_mult)
        tp2 = entry + (atr_h1 * tp2_mult)
    else:
        sl = entry + (atr_h1 * sl_mult)
        tp1 = entry - (atr_h1 * tp1_mult)
        tp2 = entry - (atr_h1 * tp2_mult)

    signal.extra["atr_h1"] = atr_h1
    signal.extra["sl"] = sl
    signal.extra["tp1"] = tp1
    signal.extra["tp2"] = tp2


# -------------------------
# Base Confidence Score
# -------------------------
def compute_confidence(
    trend_source: str,
    setup_type: str,
    adx_h1: float,
    adx_m5: float,
    high_news: bool,
) -> int:
    score = 50

    # Trend source reliability
    score += 10 if trend_source == "H1" else 5

    # Setup quality
    if setup_type.startswith("PULLBACK"):
        score += 15
    elif setup_type.startswith("BREAKOUT"):
        score += 8
    elif setup_type.startswith("BREAKOUT_CONT"):
        score += 3

    # Trend strength (ADX H1)
    if adx_h1 >= 30:
        score += 12
    elif adx_h1 >= 20:
        score += 7
    else:
        score -= 6

    # Micro structure (M5 ADX)
    if adx_m5 >= 25:
        score += 6
    elif adx_m5 < 15:
        score -= 5

    # News penalty
    if high_news:
        score -= 12

    return max(0, min(100, score))


# -------------------------
# Main loop
# -------------------------
def main():
    settings = load_settings()

    symbol_td = settings.xau_symbol_td
    symbol_label = "XAUUSD"

    telegram = TelegramClient(settings.telegram_bot_token, settings.telegram_chat_id)

    journal = TradeJournal(
        csv_path=settings.journal_csv_path,
        rolling_n=settings.journal_rolling_n,
    )

    logger.info("Starting XAUUSD bot (H1 primary, M15 fallback) + candle detection + cooldown + invalidation + journaling.")

    last_fetch_ts: float = 0.0
    cached_m5: Optional[pd.DataFrame] = None

    last_closed_m5_time: Optional[pd.Timestamp] = None
    active_trade: Optional[ActiveTrade] = None
    last_signal_time: Optional[dt.datetime] = None

    while True:
        now_utc = dt.datetime.now(dt.timezone.utc)

        # 1) Market open filter (fixes weekend signals)
        if not is_market_open(now_utc, settings.block_weekends, settings.sunday_open_hour_utc):
            logger.info("Market closed (weekend filter). Sleeping %ss...", settings.poll_seconds)
            time.sleep(settings.poll_seconds)
            continue

        # 2) Session filter (07:00–20:00 UTC etc.)
        if not is_within_sessions(
            now_utc=now_utc,
            session_1_start=settings.session_1_start,
            session_1_end=settings.session_1_end,
            session_2_start=None,
            session_2_end=None,
        ):
            logger.info("Outside session window, sleeping %ss...", settings.poll_seconds)
            time.sleep(settings.poll_seconds)
            continue

        # 3) Fetch/cached M5
        need_fetch = (time.time() - last_fetch_ts) >= settings.fetch_interval_seconds or cached_m5 is None
        if need_fetch:
            logger.info("Fetching fresh M5 OHLCV data from Twelve Data...")
            cached_m5 = fetch_m5_ohlcv_twelvedata(symbol=symbol_td, api_key=settings.twelvedata_api_key)
            last_fetch_ts = time.time()
        else:
            logger.info("Using cached M5 OHLCV data.")

        if cached_m5 is None or len(cached_m5) < 200:
            logger.warning("Not enough M5 data yet. Sleeping %ss...", settings.poll_seconds)
            time.sleep(settings.poll_seconds)
            continue

        cached_m5["datetime"] = pd.to_datetime(cached_m5["datetime"], utc=True)
        current_closed_time = cached_m5["datetime"].iloc[-1]

        # 4) New candle detection: only evaluate once per CLOSED candle
        if last_closed_m5_time is not None and current_closed_time <= last_closed_m5_time:
            logger.info(
                "No new M5 candle closed yet (last=%s, current=%s). Sleeping %ss...",
                last_closed_m5_time,
                current_closed_time,
                settings.poll_seconds,
            )
            time.sleep(settings.poll_seconds)
            continue

        last_closed_m5_time = current_closed_time
        logger.info("New M5 candle detected: %s — evaluating...", last_closed_m5_time)

        last_closed_row = cached_m5.iloc[-1]

        # 5) Manage an active trade first (TP/SL + invalidation)
        if active_trade is not None and active_trade.status == "OPEN":
            hit = check_tp_sl_hit(active_trade, last_closed_row)
            if hit:
                active_trade.status = hit["status"]
                active_trade.exit_time = last_closed_row["datetime"].to_pydatetime()
                active_trade.exit_price = float(hit["exit_price"])

                # R result
                from trade_manager import compute_r_result
                active_trade.result_r = compute_r_result(
                    active_trade.direction, active_trade.entry, active_trade.sl, active_trade.exit_price
                )
                journal.update_trade(active_trade)

                logger.info("Trade %s closed by %s at %.2f (R=%.2f)",
                            active_trade.trade_id, active_trade.status, active_trade.exit_price, active_trade.result_r)

                # cooldown anchor
                last_signal_time = active_trade.exit_time
                active_trade = None

            else:
                # invalidation window
                if active_trade.invalidation_deadline is not None:
                    now_dt = last_closed_row["datetime"].to_pydatetime()
                    if now_dt <= active_trade.invalidation_deadline:
                        # compute BB/RSI for invalidation checks using last closed candle
                        close_series = cached_m5["close"]
                        bb_u, bb_m, bb_l = indicators.bollinger_bands(close_series, period=20, std_factor=2.0)
                        rsi_series = indicators.rsi(close_series, period=14)

                        inv_reason = should_invalidate(
                            active_trade,
                            last_closed_row,
                            bb_mid=float(bb_m.iloc[-1]),
                            bb_upper=float(bb_u.iloc[-1]),
                            bb_lower=float(bb_l.iloc[-1]),
                            rsi_m5=float(rsi_series.iloc[-1]) if pd.notna(rsi_series.iloc[-1]) else 50.0,
                        )
                        if inv_reason:
                            active_trade.status = "INVALIDATED"
                            active_trade.exit_time = now_dt
                            active_trade.exit_price = float(last_closed_row["close"])
                            from trade_manager import compute_r_result
                            active_trade.result_r = compute_r_result(
                                active_trade.direction, active_trade.entry, active_trade.sl, active_trade.exit_price
                            )
                            active_trade.invalidated_reason = inv_reason

                            journal.update_trade(active_trade)
                            logger.info("Trade %s INVALIDATED: %s (exit=%.2f, R=%.2f)",
                                        active_trade.trade_id, inv_reason, active_trade.exit_price, active_trade.result_r)

                            last_signal_time = active_trade.exit_time
                            active_trade = None

        # 6) Cooldown check (prevents rapid re-entry)
        if in_cooldown(now_utc, last_signal_time, minutes=settings.cooldown_minutes):
            logger.info("Cooldown active. Sleeping %ss...", settings.poll_seconds)
            time.sleep(settings.poll_seconds)
            continue

        # 7) Resample for trend detection
        m15_df = resample_ohlc(cached_m5, "15min")
        h1_df = resample_ohlc(cached_m5, "1h")

        if len(h1_df) < 60 or len(m15_df) < 60:
            logger.info("Not enough data after resampling. Sleeping %ss...", settings.poll_seconds)
            time.sleep(settings.poll_seconds)
            continue

        # ADX(H1)
        adx_h1_series, _, _ = indicators.adx(h1_df["high"], h1_df["low"], h1_df["close"], period=14)
        adx_h1 = float(adx_h1_series.iloc[-1]) if pd.notna(adx_h1_series.iloc[-1]) else 0.0

        # Trend detection
        h1_trend = detect_trend_h1(h1_df)
        trend_source = "H1"

        if h1_trend is None:
            m15_dir = detect_trend_m15_direction(m15_df)
            if m15_dir is None:
                logger.info("No clear H1 trend and no clear M15 direction. Skipping.")
                time.sleep(settings.poll_seconds)
                continue
            trend_dir = m15_dir
            trend_source = "M15"
        else:
            trend_dir = h1_trend
            _ = confirm_trend_m15(m15_df, trend_dir)

        # News flag
        try:
            high_news = has_high_impact_news_near(symbol_label, now_utc, window_minutes=60)
        except Exception:
            high_news = False

        # 8) Trigger M5 signal (only on candle close)
        signal = trigger_signal_m5(cached_m5, trend_dir, m15_df=m15_df)
        if not signal:
            logger.info("No M5 trigger signal on candle close. Sleeping %ss.", settings.poll_seconds)
            time.sleep(settings.poll_seconds)
            continue

        # 9) Apply TP/SL and compute confidence (base + adaptive)
        apply_dynamic_tp_sl(signal, h1_df)

        setup_type = signal.extra.get("setup_type", "GENERIC")
        adx_m5 = float(signal.extra.get("adx_m5", 0.0))

        base_conf = compute_confidence(
            trend_source=trend_source,
            setup_type=setup_type,
            adx_h1=adx_h1,
            adx_m5=adx_m5,
            high_news=high_news,
        )

        # adaptive adjustment from rolling journal stats
        adj = journal.adaptive_confidence_adjustment(setup_type, trend_source)
        conf = max(0, min(100, base_conf + adj))
        signal.extra["confidence"] = conf

        # 10) Open trade + journal it
        entry = float(signal.price)
        sl = float(signal.extra.get("sl"))
        tp1 = float(signal.extra.get("tp1"))
        tp2 = float(signal.extra.get("tp2"))

        opened_time = signal.time if signal.time.tzinfo else signal.time.replace(tzinfo=dt.timezone.utc)

        active_trade = ActiveTrade(
            trade_id=new_trade_id(),
            opened_time=opened_time,
            direction=signal.direction,
            setup_type=setup_type,
            entry=entry,
            sl=sl,
            tp1=tp1,
            tp2=tp2,
            confidence=conf,
            trend_source=trend_source,
        )

        # invalidation deadline
        active_trade.invalidation_deadline = opened_time + dt.timedelta(minutes=settings.invalidation_minutes)

        journal.append_open(active_trade)
        last_signal_time = opened_time

        # 11) Build message using your template + send
        msg = build_signal_message(
            symbol_label=symbol_label,
            signal=signal,
            trend_source=trend_source,
            session_window="07:00-20:00",
            high_news=high_news,
            h1_df=h1_df,
        )

        telegram.send_message(msg)
        logger.info("Signal sent. trade_id=%s setup=%s conf=%s (adj=%+d) trend_source=%s",
                    active_trade.trade_id, setup_type, conf, adj, trend_source)

        time.sleep(settings.poll_seconds)


if __name__ == "__main__":
    main()
