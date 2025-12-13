# main.py
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
)

# Your repo modules
from cooldown import in_cooldown  # must exist in cooldown.py
from signal_formatter import build_signal_message  # must exist in signal_formatter.py
from data_logger import TradeJournal  # must exist in data_logger.py (CSV logger)
from trade_manager import (
    ActiveTrade,
    new_trade_id,
    check_tp_sl_hit,
    should_invalidate,
    compute_r_result,
)

logger = logging.getLogger("xauusd_bot")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# -------------------------
# Market open filter (fix weekend signals)
# -------------------------
def is_market_open(now_utc: dt.datetime) -> bool:
    """
    FX-style hours:
      - Closed Sat
      - Closed Sun until 22:00 UTC
      - Closed Fri from 22:00 UTC
    """
    wd = now_utc.weekday()  # Mon=0 ... Sun=6
    hhmm = now_utc.hour * 100 + now_utc.minute

    if wd == 5:  # Saturday
        return False
    if wd == 6:  # Sunday
        return hhmm >= 2200
    if wd == 4:  # Friday
        return hhmm < 2200
    return True


# -------------------------
# Resampling (M5 -> M15/H1)
# -------------------------
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


# -------------------------
# TP/SL (ATR H1)
# -------------------------
def tp_sl_multipliers(setup_type: str):
    if setup_type.startswith("PULLBACK"):
        return (0.90, 1.60, 2.60)
    if setup_type.startswith("BREAKOUT_CONT"):
        return (0.55, 0.90, 1.50)
    if setup_type.startswith("BREAKOUT"):
        return (0.70, 1.20, 2.00)
    return (0.75, 1.10, 1.80)


def apply_dynamic_tp_sl(signal, h1_df: pd.DataFrame) -> None:
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
# Main loop
# -------------------------
def main():
    settings = load_settings()

    symbol_td = settings.xau_symbol_td
    symbol_label = "XAUUSD"

    s1_start = settings.session_1_start
    s1_end = settings.session_1_end

    telegram = TelegramClient(settings.telegram_bot_token, settings.telegram_chat_id)
    journal = TradeJournal("trades.csv")  # data_logger.py

    sleep_seconds = 60
    fetch_interval_seconds = 180  # API pacing

    last_fetch_ts = 0.0
    cached_m5: Optional[pd.DataFrame] = None
    last_closed_candle_time: Optional[pd.Timestamp] = None

    last_signal_time: Optional[dt.datetime] = None
    active_trade: Optional[ActiveTrade] = None

    logger.info("Starting XAUUSD bot (H1 primary, M15 fallback) + candle-close eval + cooldown + journaling + invalidation.")

    while True:
        now_utc = dt.datetime.now(dt.timezone.utc)

        # 1) Market open filter (prevents weekend signals)
        if not is_market_open(now_utc):
            logger.info("Market closed (weekend window). Sleeping %ss...", sleep_seconds)
            time.sleep(sleep_seconds)
            continue

        # 2) Session filter
        if not is_within_sessions(
            now_utc=now_utc,
            session_1_start=s1_start,
            session_1_end=s1_end,
            session_2_start=None,
            session_2_end=None,
        ):
            logger.info("Outside trading session, sleeping %ss...", sleep_seconds)
            time.sleep(sleep_seconds)
            continue

        # 3) Fetch / cache
        need_fetch = (time.time() - last_fetch_ts) >= fetch_interval_seconds or cached_m5 is None
        if need_fetch:
            logger.info("Fetching fresh M5 OHLCV data from Twelve Data...")
            cached_m5 = fetch_m5_ohlcv_twelvedata(symbol=symbol_td, api_key=settings.twelvedata_api_key)
            last_fetch_ts = time.time()
        else:
            logger.info("Using cached M5 OHLCV data.")

        if cached_m5 is None or len(cached_m5) < 300:
            logger.warning("Not enough M5 data yet. Sleeping %ss...", sleep_seconds)
            time.sleep(sleep_seconds)
            continue

        cached_m5["datetime"] = pd.to_datetime(cached_m5["datetime"], utc=True)
        current_last_time = cached_m5["datetime"].iloc[-1]

        # 4) NEW candle detection (only evaluate on closed candle)
        if last_closed_candle_time is not None and current_last_time <= last_closed_candle_time:
            logger.info(
                "No new M5 candle closed yet (last=%s, current=%s). Sleeping %ss...",
                last_closed_candle_time, current_last_time, sleep_seconds
            )
            time.sleep(sleep_seconds)
            continue

        last_closed_candle_time = current_last_time
        logger.info("New M5 candle detected: %s — evaluating...", current_last_time)

        # 5) Resample
        m15_df = resample_ohlc(cached_m5, "15min")
        h1_df = resample_ohlc(cached_m5, "1h")
        if len(m15_df) < 60 or len(h1_df) < 60:
            logger.info("Not enough data after resampling, sleeping %ss...", sleep_seconds)
            time.sleep(sleep_seconds)
            continue

        last_candle = cached_m5.iloc[-1]

        # 6) Manage open trade (TP/SL + invalidation)
        if active_trade is not None and active_trade.status == "OPEN":
            # TP/SL check
            hit = check_tp_sl_hit(active_trade, last_candle)
            if hit:
                active_trade.status = hit["status"]
                active_trade.exit_time = current_last_time.to_pydatetime()
                active_trade.exit_price = float(hit["exit_price"])
                active_trade.result_r = compute_r_result(
                    active_trade.direction, active_trade.entry, active_trade.sl, active_trade.exit_price
                )
                journal.update_trade(active_trade)

                telegram.send_message(
                    f"Trade Update [{active_trade.status}]\n"
                    f"TradeID: {active_trade.trade_id}\n"
                    f"Exit: {active_trade.exit_price:.2f}\n"
                    f"R: {active_trade.result_r:.2f}"
                )
                last_signal_time = active_trade.exit_time
                active_trade = None

            else:
                # Invalidation check (within deadline)
                close = cached_m5["close"]
                bb_u, bb_m, bb_l = indicators.bollinger_bands(close, period=20, std_factor=2.0)
                rsi_m5 = float(indicators.rsi(close, period=14).iloc[-1]) if len(close) >= 20 else 50.0

                reason = should_invalidate(
                    trade=active_trade,
                    last_closed_m5=last_candle,
                    bb_mid=float(bb_m.iloc[-1]),
                    bb_upper=float(bb_u.iloc[-1]),
                    bb_lower=float(bb_l.iloc[-1]),
                    rsi_m5=rsi_m5,
                )

                if (
                    reason
                    and active_trade.invalidation_deadline is not None
                    and now_utc <= active_trade.invalidation_deadline
                ):
                    active_trade.status = "INVALIDATED"
                    active_trade.invalidated_reason = reason
                    active_trade.exit_time = current_last_time.to_pydatetime()
                    active_trade.exit_price = float(last_candle["close"])
                    active_trade.result_r = compute_r_result(
                        active_trade.direction, active_trade.entry, active_trade.sl, active_trade.exit_price
                    )
                    journal.update_trade(active_trade)

                    telegram.send_message(
                        f"Signal Invalidated ❌\n"
                        f"TradeID: {active_trade.trade_id}\n"
                        f"{reason}\n"
                        f"Close: {active_trade.exit_price:.2f}\n"
                        f"R (if entered): {active_trade.result_r:.2f}"
                    )

                    last_signal_time = active_trade.exit_time
                    active_trade = None

        # If trade still open, do not open new entries
        if active_trade is not None:
            logger.info("Active trade open, skipping new entries.")
            time.sleep(sleep_seconds)
            continue

        # 7) Cooldown protocol
        if in_cooldown(now_utc, last_signal_time, minutes=20):
            logger.info("Cooldown active, skipping.")
            time.sleep(sleep_seconds)
            continue

        # 8) Trend detection
        h1_trend = detect_trend_h1(h1_df)
        trend_source = "H1"
        if h1_trend is None:
            m15_dir = detect_trend_m15_direction(m15_df)
            if m15_dir is None:
                logger.info("No clear H1 trend and no clear M15 direction, skipping.")
                time.sleep(sleep_seconds)
                continue
            trend_dir = m15_dir
            trend_source = "M15"
        else:
            trend_dir = h1_trend
            _ = confirm_trend_m15(m15_df, trend_dir)

        # 9) News flag
        try:
            high_news = has_high_impact_news_near(symbol_label, now_utc, window_minutes=60)
        except Exception:
            high_news = False

        # 10) Trigger signal
        signal = trigger_signal_m5(cached_m5, trend_dir)
        if not signal:
            logger.info("No M5 trigger signal on candle close.")
            time.sleep(sleep_seconds)
            continue

        apply_dynamic_tp_sl(signal, h1_df)

        # 11) Send formatted message (your template lives in signal_formatter.py)
        msg = build_signal_message(
            symbol_label=symbol_label,
            signal=signal,
            trend_source=trend_source,
            session_window="07:00-20:00",
            high_news=high_news,
            h1_df=h1_df,   # let formatter compute ADX/Regime if you coded it that way
        )
        telegram.send_message(msg)

        # 12) Create active trade + journal open
        trade = ActiveTrade(
            trade_id=new_trade_id(),
            opened_time=signal.time if signal.time.tzinfo else signal.time.replace(tzinfo=dt.timezone.utc),
            direction=signal.direction,
            setup_type=signal.extra.get("setup_type", "GENERIC"),
            entry=float(signal.price),
            sl=float(signal.extra["sl"]),
            tp1=float(signal.extra["tp1"]),
            tp2=float(signal.extra["tp2"]),
            confidence=int(signal.extra.get("confidence", 0)),
            trend_source=trend_source,
            invalidation_deadline=(now_utc + dt.timedelta(minutes=10)),
        )
        journal.append_open(trade)

        active_trade = trade
        last_signal_time = now_utc

        logger.info("Signal sent + trade opened. trade_id=%s setup=%s", trade.trade_id, trade.setup_type)
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()
