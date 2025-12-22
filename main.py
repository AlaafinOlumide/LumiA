# main.py
import time
import uuid
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
from data_logger import (
    append_signal_open,
    update_signal_close,
    get_open_signals,
    safe_float,
    safe_dt_iso_to_utc,
)

logger = logging.getLogger("xauusd_bot")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def _iso(t: dt.datetime) -> str:
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    return t.astimezone(dt.timezone.utc).isoformat()


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
    return (now_utc - last_signal_time).total_seconds() < cooldown_minutes * 60


def _simulate_hit_order_conservative(direction: str, bar_high: float, bar_low: float, sl: float, tp1: float, tp2: float) -> Optional[str]:
    """
    If TP and SL both touched inside same bar, we treat it as SL first (conservative).
    """
    direction = direction.upper()
    if direction == "LONG":
        sl_hit = bar_low <= sl
        tp1_hit = bar_high >= tp1
        tp2_hit = bar_high >= tp2
        if sl_hit:
            return "SL"
        if tp2_hit:
            return "TP2"
        if tp1_hit:
            return "TP1"
        return None

    if direction == "SHORT":
        sl_hit = bar_high >= sl
        tp1_hit = bar_low <= tp1
        tp2_hit = bar_low <= tp2
        if sl_hit:
            return "SL"
        if tp2_hit:
            return "TP2"
        if tp1_hit:
            return "TP1"
        return None

    return None


def _run_journal_evaluation(
    *,
    settings,
    cached_m5: pd.DataFrame,
    now_utc: dt.datetime,
    telegram: TelegramClient,
) -> None:
    """
    DROP 2:
    - Check open signals and mark TP1/TP2/SL
    - Expire after max_hold_minutes
    """
    open_signals = get_open_signals(settings.journal_path)
    if not open_signals:
        return

    # Ensure datetime UTC
    m5 = cached_m5.copy()
    m5["datetime"] = pd.to_datetime(m5["datetime"], utc=True)

    for row in open_signals:
        signal_id = row.get("signal_id", "")
        direction = str(row.get("direction", "")).upper()
        entry_time = safe_dt_iso_to_utc(row.get("entry_time_utc", ""))

        if not signal_id or not direction or entry_time is None:
            continue

        entry = safe_float(row.get("entry"))
        sl = safe_float(row.get("sl"))
        tp1 = safe_float(row.get("tp1"))
        tp2 = safe_float(row.get("tp2"))

        # Slice bars AFTER entry time (strictly greater)
        future = m5[m5["datetime"] > pd.Timestamp(entry_time)]
        if future.empty:
            # still no future bars to evaluate
            continue

        # Expiry check
        age_min = (now_utc - entry_time).total_seconds() / 60.0
        if age_min >= settings.max_hold_minutes:
            last_close = float(m5["close"].iloc[-1])
            risk = abs(entry - sl)
            if risk <= 1e-9:
                risk = 1.0
            pnl_r = ((last_close - entry) / risk) if direction == "LONG" else ((entry - last_close) / risk)

            update_signal_close(
                settings.journal_path,
                signal_id,
                result="EXPIRED",
                hit_time_utc=_iso(now_utc),
                exit_price=last_close,
                pnl_r=pnl_r,
                notes=f"Expired after {settings.max_hold_minutes}m",
            )
            if settings.journal_notify_telegram:
                telegram.send_message(f"📒 Journal: {signal_id} -> EXPIRED | pnl ≈ {pnl_r:.2f}R")
            continue

        # Scan bars chronologically
        for _, b in future.iterrows():
            bar_high = float(b["high"])
            bar_low = float(b["low"])

            hit = _simulate_hit_order_conservative(direction, bar_high, bar_low, sl, tp1, tp2)
            if not hit:
                continue

            # Determine exit + pnl_r
            if hit == "SL":
                exit_price = sl
            elif hit == "TP2":
                exit_price = tp2
            else:
                exit_price = tp1

            risk = abs(entry - sl)
            if risk <= 1e-9:
                risk = 1.0

            pnl_r = ((exit_price - entry) / risk) if direction == "LONG" else ((entry - exit_price) / risk)

            update_signal_close(
                settings.journal_path,
                signal_id,
                result=hit,
                hit_time_utc=_iso(now_utc),
                exit_price=exit_price,
                pnl_r=pnl_r,
                notes=f"Hit {hit}",
            )
            if settings.journal_notify_telegram:
                telegram.send_message(f"📒 Journal: {signal_id} -> {hit} | pnl ≈ {pnl_r:.2f}R")
            break


def main():
    settings = load_settings()

    symbol_td = settings.xau_symbol_td
    symbol_label = "XAUUSD"

    telegram = TelegramClient(settings.telegram_bot_token, settings.telegram_chat_id)

    logger.info("Starting XAUUSD bot (H1 primary, M15 fallback) + score gate + journaling + invalidation.")

    last_fetch_ts: float = 0.0
    cached_m5: Optional[pd.DataFrame] = None

    last_closed_m5_time: Optional[pd.Timestamp] = None
    last_signal_time: Optional[dt.datetime] = None
    last_signal_dir: Optional[str] = None

    while True:
        now_utc = dt.datetime.now(dt.timezone.utc)

        # Session + weekend filter (your strategy.is_within_sessions should support trade_weekends)
        if not is_within_sessions(
            now_utc=now_utc,
            session_1_start=settings.session_1_start,
            session_1_end=settings.session_1_end,
            session_2_start=None,
            session_2_end=None,
            trade_weekends=settings.trade_weekends,
        ):
            logger.info("Outside trading session/weekend filter, sleeping %ss...", settings.sleep_seconds)
            time.sleep(settings.sleep_seconds)
            continue

        # Fetch/cache
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

        # Run journaling evaluation every loop once we have data
        _run_journal_evaluation(
            settings=settings,
            cached_m5=cached_m5,
            now_utc=now_utc,
            telegram=telegram,
        )

        # New candle detection (evaluate only on a new CLOSED M5 candle)
        current_last_time = pd.to_datetime(cached_m5["datetime"].iloc[-1], utc=True)
        if last_closed_m5_time is not None and current_last_time == last_closed_m5_time:
            logger.info("No new M5 candle closed yet (last=%s). Sleeping %ss...", last_closed_m5_time, settings.sleep_seconds)
            time.sleep(settings.sleep_seconds)
            continue

        last_closed_m5_time = current_last_time
        logger.info("New M5 candle detected: %s — evaluating signal...", last_closed_m5_time)

        # Resample for H1 + M15
        m15_df = resample_ohlc(cached_m5, "15min")
        h1_df = resample_ohlc(cached_m5, "1h")

        if h1_df is None or len(h1_df) < 60 or m15_df is None or len(m15_df) < 60:
            logger.info("Not enough data after resampling, sleeping %ss...", settings.sleep_seconds)
            time.sleep(settings.sleep_seconds)
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
                logger.info("No clear H1 trend and no clear M15 direction, skipping.")
                time.sleep(settings.sleep_seconds)
                continue
            trend_dir = m15_dir
            trend_source = "M15"
        else:
            trend_dir = h1_trend
            _ = confirm_trend_m15(m15_df, trend_dir)

        # News flag
        try:
            high_news = has_high_impact_news_near(symbol_label, now_utc, window_minutes=settings.news_window_minutes)
        except Exception:
            high_news = False

        # Cooldown protocol
        if in_cooldown(now_utc, last_signal_time, settings.cooldown_minutes):
            if settings.cooldown_same_direction_only and last_signal_dir and last_signal_dir != trend_dir:
                pass
            else:
                logger.info("Cooldown active (%sm). Skipping this candle.", settings.cooldown_minutes)
                time.sleep(settings.sleep_seconds)
                continue

        # Trigger (must include score gate inside strategy.trigger_signal_m5)
        signal = trigger_signal_m5(
            m5_df=cached_m5,
            trend_dir=trend_dir,
            m15_df=m15_df,
            h1_df=h1_df,
            high_news=high_news,
            min_score=settings.min_entry_score,
        )

        if not signal:
            logger.info("No M5 trigger signal on candle close, sleeping %ss.", settings.sleep_seconds)
            time.sleep(settings.sleep_seconds)
            continue

        # DROP 2: invalidate existing open signals if opposite direction signal appears
        if settings.invalidate_on_opposite_signal:
            open_signals = get_open_signals(settings.journal_path)
            for row in open_signals:
                if str(row.get("direction", "")).upper() != signal.direction.upper():
                    # invalidate opposite signal
                    signal_id = row.get("signal_id", "")
                    entry = safe_float(row.get("entry"))
                    sl = safe_float(row.get("sl"))
                    last_close = float(cached_m5["close"].iloc[-1])
                    risk = abs(entry - sl)
                    if risk <= 1e-9:
                        risk = 1.0
                    direction = str(row.get("direction", "")).upper()
                    pnl_r = ((last_close - entry) / risk) if direction == "LONG" else ((entry - last_close) / risk)

                    update_signal_close(
                        settings.journal_path,
                        signal_id,
                        result="INVALIDATED",
                        hit_time_utc=_iso(now_utc),
                        exit_price=last_close,
                        pnl_r=pnl_r,
                        notes="Invalidated by opposite signal",
                    )
                    if settings.journal_notify_telegram:
                        telegram.send_message(f"📒 Journal: {signal_id} -> INVALIDATED (opposite signal) | pnl ≈ {pnl_r:.2f}R")

        # Regime + adaptive TP/SL
        regime = market_regime(h1_df)
        dynamic_tp_sl(signal, h1_df, regime)

        # Market classifiers
        market_state = market_state_from_adx(adx_h1)
        trend_strength = trend_strength_from_adx(adx_h1)

        # Confidence
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
            session_window="07:00-20:00",
            high_news=high_news,
            market_state=market_state,
            market_regime_text=regime,
            adx_h1=adx_h1,
            trend_strength=trend_strength,
            confidence_score=conf,
            confidence_text=conf_text,
        )

        telegram.send_message(msg)

        # Journal OPEN
        signal_id = str(uuid.uuid4())[:8]
        append_signal_open(settings.journal_path, {
            "signal_id": signal_id,
            "symbol": symbol_label,
            "direction": signal.direction,
            "setup_type": setup_type,
            "trend_source": trend_source,
            "trend_bias": trend_dir,
            "entry_time_utc": _iso(now_utc),
            "entry": f"{signal.price:.5f}",
            "sl": f"{float(signal.extra.get('sl', 0.0)):.5f}",
            "tp1": f"{float(signal.extra.get('tp1', 0.0)):.5f}",
            "tp2": f"{float(signal.extra.get('tp2', 0.0)):.5f}",
            "confidence": str(conf),
            "entry_score": str(entry_score),
            "status": "OPEN",
            "closed_time_utc": "",
            "exit_price": "",
            "pnl_r": "",
            "notes": "",
        })

        last_signal_time = now_utc
        last_signal_dir = signal.direction

        logger.info("Signal sent. source=%s dir=%s setup=%s score=%s conf=%s (%s) id=%s",
                    trend_source, signal.direction, setup_type, entry_score, conf, conf_text, signal_id)

        time.sleep(settings.sleep_seconds)


if __name__ == "__main__":
    main()