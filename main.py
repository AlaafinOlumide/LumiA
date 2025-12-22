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
    market_regime,
    dynamic_tp_sl,
    compute_confidence,
    confidence_label,
)

from signal_formatter import build_signal_message
from journal import (
    new_signal_id,
    append_open_signal,
    load_open_signals,
    update_signal_close,
    evaluate_outcome_on_m5,
)

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


def _iso(dtobj: dt.datetime) -> str:
    if dtobj.tzinfo is None:
        dtobj = dtobj.replace(tzinfo=dt.timezone.utc)
    return dtobj.astimezone(dt.timezone.utc).isoformat()


def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _run_journal_evaluation(settings, cached_m5: pd.DataFrame, telegram: TelegramClient) -> None:
    """
    For OPEN signals:
      - wait eval_delay_minutes
      - then check if SL/TP1/TP2 got hit on M5
      - close if hit OR if max_hold exceeded
    """
    open_df = load_open_signals(settings.journal_path)
    if open_df is None or open_df.empty:
        return

    now_utc = dt.datetime.now(dt.timezone.utc)

    for _, row in open_df.iterrows():
        try:
            signal_id = str(row["signal_id"])
            entry_time_iso = str(row["time_utc"])
            entry_time = dt.datetime.fromisoformat(entry_time_iso)
            if entry_time.tzinfo is None:
                entry_time = entry_time.replace(tzinfo=dt.timezone.utc)
            entry_time = entry_time.astimezone(dt.timezone.utc)

            age_min = (now_utc - entry_time).total_seconds() / 60.0

            # not ready yet
            if age_min < settings.eval_delay_minutes:
                continue

            # expired
            if age_min >= settings.max_hold_minutes:
                # exit at latest close
                last_close = float(cached_m5["close"].iloc[-1])
                entry = _safe_float(row["entry"])
                sl = _safe_float(row["sl"])
                risk = abs(entry - sl) if abs(entry - sl) > 1e-9 else 1.0
                direction = str(row["direction"])
                pnl_r = ((last_close - entry) / risk) if direction == "LONG" else ((entry - last_close) / risk

                update_signal_close(
                    settings.journal_path,
                    signal_id,
                    result="EXPIRED",
                    hit_time_utc=_iso(now_utc),
                    exit_price=last_close,
                    pnl_r=pnl_r,
                    notes=f"Expired after {settings.max_hold_minutes}m",
                )
                telegram.send_message(f"📒 Journal update: {signal_id} -> EXPIRED | pnl ≈ {pnl_r:.2f}R")
                continue

            # evaluate hits
            direction = str(row["direction"])
            entry = _safe_float(row["entry"])
            sl = _safe_float(row["sl"])
            tp1 = _safe_float(row["tp1"])
            tp2 = _safe_float(row["tp2"])

            result, hit_time, exit_price, pnl_r, notes = evaluate_outcome_on_m5(
                cached_m5,
                direction=direction,
                entry_time_utc_iso=entry_time_iso,
                entry=entry,
                sl=sl,
                tp1=tp1,
                tp2=tp2,
            )

            # only close if SL/TP1/TP2 hit (NOT NONE)
            if result in {"SL", "TP1", "TP2"}:
                update_signal_close(
                    settings.journal_path,
                    signal_id,
                    result=result,
                    hit_time_utc=_iso(hit_time) if hit_time else "",
                    exit_price=exit_price,
                    pnl_r=pnl_r,
                    notes=notes,
                )
                telegram.send_message(f"📒 Journal update: {signal_id} -> {result} | pnl ≈ {pnl_r:.2f}R")
        except Exception as e:
            logger.warning("Journal evaluation error: %s", e)


def main():
    settings = load_settings()
    symbol_td = settings.xau_symbol_td
    symbol_label = "XAUUSD"

    telegram = TelegramClient(settings.telegram_bot_token, settings.telegram_chat_id)

    logger.info("Starting XAUUSD bot (Drop1+Drop2+Journaling integrated).")

    last_fetch_ts: float = 0.0
    cached_m5: Optional[pd.DataFrame] = None
    last_closed_m5_time: Optional[pd.Timestamp] = None

    last_signal_time: Optional[dt.datetime] = None
    last_signal_dir: Optional[str] = None

    score_weights = {
        "score_m15_structure": settings.score_m15_structure,
        "score_pullback_zone": settings.score_pullback_zone,
        "score_rsi_reset": settings.score_rsi_reset,
        "score_stoch_reset": settings.score_stoch_reset,
        "score_rejection": settings.score_rejection,
        "score_adx_ok": settings.score_adx_ok,
        "score_no_news": settings.score_no_news,
    }

    while True:
        now_utc = dt.datetime.now(dt.timezone.utc)

        if not is_within_sessions(
            now_utc=now_utc,
            session_1_start=settings.session_1_start,
            session_1_end=settings.session_1_end,
            session_2_start=settings.session_2_start,
            session_2_end=settings.session_2_end,
            trade_weekends=settings.trade_weekends,
        ):
            logger.info("Outside trading session/weekend filter, sleeping %ss...", settings.sleep_seconds)
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

        # Run journaling evaluation every loop (cheap)
        _run_journal_evaluation(settings, cached_m5, telegram)

        # New candle detection
        current_last_time = cached_m5["datetime"].iloc[-1]
        if last_closed_m5_time is not None and current_last_time == last_closed_m5_time:
            logger.info("No new M5 candle closed yet (last=%s). Sleeping %ss...", last_closed_m5_time, settings.sleep_seconds)
            time.sleep(settings.sleep_seconds)
            continue

        last_closed_m5_time = current_last_time
        logger.info("New M5 candle detected: %s — evaluating signal...", last_closed_m5_time)

        # Resample
        m15_df = resample_ohlc(cached_m5, "15min")
        h1_df = resample_ohlc(cached_m5, "1h")

        if h1_df is None or len(h1_df) < 60 or m15_df is None or len(m15_df) < 60:
            logger.info("Not enough data after resampling, sleeping %ss...", settings.sleep_seconds)
            time.sleep(settings.sleep_seconds)
            continue

        # ADX(H1)
        adx_h1_series, _, _ = indicators.adx(h1_df["high"], h1_df["low"], h1_df["close"], period=14)
        adx_h1 = float(adx_h1_series.iloc[-1]) if pd.notna(adx_h1_series.iloc[-1]) else 0.0

        # Trend
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

        # News
        try:
            high_news = has_high_impact_news_near(symbol_label, now_utc, window_minutes=settings.news_window_minutes)
        except Exception:
            high_news = False

        # Cooldown
        if in_cooldown(now_utc, last_signal_time, settings.cooldown_minutes):
            if settings.cooldown_same_direction_only and last_signal_dir and last_signal_dir != trend_dir:
                pass
            else:
                logger.info("Cooldown active (%sm). Skipping this candle.", settings.cooldown_minutes)
                time.sleep(settings.sleep_seconds)
                continue

        # Trigger
        signal = trigger_signal_m5(
            m5_df=cached_m5,
            trend_dir=trend_dir,
            m15_df=m15_df,
            h1_df=h1_df,
            high_news=high_news,
            min_score=settings.min_entry_score,
            score_weights=score_weights,
        )
        if not signal:
            logger.info("No M5 trigger signal on candle close, sleeping %ss.", settings.sleep_seconds)
            time.sleep(settings.sleep_seconds)
            continue

        # Regime + adaptive TP/SL
        regime = market_regime(h1_df)
        dynamic_tp_sl(signal, h1_df, regime)

        market_state = market_state_from_adx(adx_h1)
        trend_strength = trend_strength_from_adx(adx_h1)

        setup_type = str(signal.extra.get("setup_type", "GENERIC"))
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

        # Message
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

        # Journal OPEN entry
        sid = new_signal_id()
        time_iso = _iso(signal.time if isinstance(signal.time, dt.datetime) else now_utc)

        row = {
            "signal_id": sid,
            "symbol": symbol_label,
            "direction": signal.direction,
            "setup_type": setup_type,
            "trend_source": trend_source,
            "trend_dir": trend_dir,
            "time_utc": time_iso,
            "entry": float(signal.price),
            "sl": float(signal.extra.get("sl")),
            "tp1": float(signal.extra.get("tp1")),
            "tp2": float(signal.extra.get("tp2")),
            "entry_score": entry_score,
            "confidence": conf,
            "adx_h1": float(adx_h1),
            "adx_m5": float(adx_m5),
            "regime": regime,
            "status": "OPEN",
            "result": "",
            "hit_time_utc": "",
            "exit_price": "",
            "pnl_r": "",
            "notes": "",
        }
        append_open_signal(settings.journal_path, row)

        telegram.send_message(f"📌 Signal ID: {sid} (journaled)")

        last_signal_time = now_utc
        last_signal_dir = signal.direction

        logger.info("Signal sent. id=%s source=%s dir=%s setup=%s score=%s conf=%s (%s)",
                    sid, trend_source, trend_dir, setup_type, entry_score, conf, conf_text)

        time.sleep(settings.sleep_seconds)


if __name__ == "__main__":
    main()