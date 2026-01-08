# main.py
from __future__ import annotations

import os
import time
import datetime as dt
import logging
from typing import Optional, Tuple

import pandas as pd

from strategy import active_session, detect_trend_h1, trigger_signal_m5
from signal_formatter import build_signal_message

from data_fetcher import get_ohlcv_df
from notifier import send_telegram_message
from high_impact_news import is_high_impact_now


logger = logging.getLogger("xauusd_bot")


# -------------------------
# Env helpers
# -------------------------
def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    try:
        return int(v)
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    try:
        return float(v)
    except Exception:
        return default


def _setup_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


# -------------------------
# DataFrame time handling
# -------------------------
def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    if "datetime" in df.columns:
        df = df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        df = df.dropna(subset=["datetime"])
        df = df.sort_values("datetime")
        df = df.set_index("datetime", drop=False)

    # Ensure tz-aware UTC index
    try:
        if getattr(df.index, "tz", None) is None:
            df = df.copy()
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    except Exception:
        pass

    return df


def _last_closed_candle_time(df: pd.DataFrame, ignore_latest_candle: bool) -> Optional[pd.Timestamp]:
    if df is None or len(df) < 3:
        return None
    idx = -2 if ignore_latest_candle else -1
    try:
        return pd.Timestamp(df.index[idx]).tz_convert("UTC")
    except Exception:
        try:
            return pd.Timestamp(df.iloc[idx]["datetime"]).tz_convert("UTC")
        except Exception:
            return None


# -------------------------
# ADX + momentum override
# -------------------------
def _calc_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    high = pd.to_numeric(high, errors="coerce")
    low = pd.to_numeric(low, errors="coerce")
    close = pd.to_numeric(close, errors="coerce")

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()

    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, pd.NA))
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, pd.NA))

    denom = (plus_di + minus_di).replace(0, pd.NA)
    dx = 100 * (plus_di - minus_di).abs() / denom

    # ✅ force numeric dtype BEFORE fillna to avoid FutureWarning
    dx = pd.to_numeric(dx, errors="coerce").fillna(0.0)

    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx


def _momentum_override_trend(m5_df: pd.DataFrame) -> Tuple[Optional[str], float]:
    if m5_df is None or len(m5_df) < 60:
        return None, 0.0

    df = m5_df.copy()
    for c in ("high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["high", "low", "close"])
    if len(df) < 60:
        return None, 0.0

    adx = _calc_adx(df["high"], df["low"], df["close"], period=14)
    last_adx = float(adx.iloc[-1]) if pd.notna(adx.iloc[-1]) else 0.0

    # Simple DI direction inference
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [(df["high"] - df["low"]), (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()

    plus_di = 100 * (plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr.replace(0, pd.NA))
    minus_di = 100 * (minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr.replace(0, pd.NA))

    last_plus = float(plus_di.iloc[-1]) if pd.notna(plus_di.iloc[-1]) else 0.0
    last_minus = float(minus_di.iloc[-1]) if pd.notna(minus_di.iloc[-1]) else 0.0

    if last_plus > last_minus:
        return "BULL", last_adx
    if last_minus > last_plus:
        return "BEAR", last_adx
    return None, last_adx


# -------------------------
# News wrapper (fix your error)
# -------------------------
def _safe_news_check(now_utc: dt.datetime, block_on_unknown_news: bool) -> bool:
    """
    Your repo's is_high_impact_now() currently takes NO args.
    This wrapper supports both:
      - is_high_impact_now()
      - is_high_impact_now(now_utc)
      - is_high_impact_now(now_utc=...)
    """
    try:
        try:
            return bool(is_high_impact_now(now_utc=now_utc))  # if supported
        except TypeError:
            try:
                return bool(is_high_impact_now(now_utc))      # if supported positionally
            except TypeError:
                return bool(is_high_impact_now())             # ✅ your current case
    except Exception as e:
        logger.warning("News check failed (%s).", e)
        return True if block_on_unknown_news else False


# -------------------------
# Main loop
# -------------------------
def main() -> None:
    _setup_logging()

    symbol = os.getenv("SYMBOL", "XAU/USD")
    symbol_label = os.getenv("SYMBOL_LABEL", "XAUUSD")

    outputsize = _env_int("OUTPUTSIZE", 2000)
    fetch_interval = _env_int("FETCH_INTERVAL_SEC", 180)
    sleep_seconds = _env_int("SLEEP_SECONDS", 60)
    ignore_latest_candle = _env_bool("IGNORE_LATEST_CANDLE", True)

    # Sessions (match your logs)
    enable_asia = _env_bool("ENABLE_ASIA", True)
    trade_weekends = _env_bool("TRADE_WEEKENDS", False)
    asia_start = _env_int("ASIA_START_UTC", 0)
    asia_end = _env_int("ASIA_END_UTC", 3)
    london_start = _env_int("LONDON_START_UTC", 6)
    london_end = _env_int("LONDON_END_UTC", 21)

    # Gates
    min_entry_score = _env_int("MIN_ENTRY_SCORE", 65)
    asia_extra_score_buffer = _env_int("ASIA_EXTRA_BUFFER", 2)

    enable_momentum_override = _env_bool("MOMENTUM_OVERRIDE", True)
    adx_override_threshold = _env_float("ADX_OVERRIDE_THRESHOLD", 35.0)

    # Risk / RR
    tp1_rr = _env_float("TP1_RR", 1.2)
    tp2_rr = _env_float("TP2_RR", 2.0)
    asia_tp1_rr = _env_float("ASIA_TP1_RR", 0.9)
    sl_atr_mult = _env_float("SL_ATR_MULT", 1.5)

    block_on_unknown_news = _env_bool("BLOCK_ON_UNKNOWN_NEWS", False)

    logger.info("Starting XAUUSD bot on Render/local.")
    logger.info(
        "Sessions UTC: ASIA=%s (%02d-%02d) | LONDON_NY (%02d-%02d) | weekends=%s",
        enable_asia, asia_start, asia_end, london_start, london_end, trade_weekends
    )
    logger.info(
        "Data: outputsize=%s fetch_interval=%ss sleep=%ss ignore_latest_candle=%s",
        outputsize, fetch_interval, sleep_seconds, ignore_latest_candle
    )
    logger.info(
        "Gates: min_score=%s asia_buffer=%s momentum_override=%s (ADX_M5>=%.1f)",
        min_entry_score, asia_extra_score_buffer, enable_momentum_override, adx_override_threshold
    )

    cached_m5: Optional[pd.DataFrame] = None
    cached_m15: Optional[pd.DataFrame] = None
    cached_h1: Optional[pd.DataFrame] = None
    last_fetch_ts = 0.0
    last_eval_candle_time: Optional[pd.Timestamp] = None

    while True:
        now_utc = _utc_now()

        session = active_session(
            now_utc=now_utc,
            enable_asia=enable_asia,
            trade_weekends=trade_weekends,
            asia_start_hour_utc=asia_start,
            asia_end_hour_utc=asia_end,
            london_start_hour_utc=london_start,
            london_end_hour_utc=london_end,
        )

        if session is None:
            logger.info("Outside trading sessions. Sleeping %ss...", sleep_seconds)
            time.sleep(sleep_seconds)
            continue

        do_refresh = (time.time() - last_fetch_ts) >= fetch_interval

        try:
            if do_refresh or cached_m5 is None:
                logger.info("Fetching fresh M5 OHLCV data from Twelve Data...")
                cached_m5 = get_ohlcv_df(symbol=symbol, interval="5min", outputsize=outputsize)
                last_fetch_ts = time.time()
            else:
                logger.info("Using cached M5 OHLCV data.")

            if do_refresh or cached_m15 is None:
                cached_m15 = get_ohlcv_df(symbol=symbol, interval="15min", outputsize=outputsize)
            if do_refresh or cached_h1 is None:
                cached_h1 = get_ohlcv_df(symbol=symbol, interval="1h", outputsize=outputsize)

            m5_df = _ensure_utc_index(cached_m5)
            m15_df = _ensure_utc_index(cached_m15)
            h1_df = _ensure_utc_index(cached_h1)

        except Exception as e:
            logger.exception("Data fetch failed: %s. Sleeping %ss...", e, sleep_seconds)
            time.sleep(sleep_seconds)
            continue

        eval_candle_time = _last_closed_candle_time(m5_df, ignore_latest_candle=ignore_latest_candle)
        if eval_candle_time is None:
            logger.info("Not enough M5 data to evaluate. Sleeping %ss...", sleep_seconds)
            time.sleep(sleep_seconds)
            continue

        if last_eval_candle_time is not None and eval_candle_time <= last_eval_candle_time:
            logger.info("No new CLOSED M5 candle yet (eval_time=%s). Sleeping %ss...", str(eval_candle_time), sleep_seconds)
            time.sleep(sleep_seconds)
            continue

        last_eval_candle_time = eval_candle_time
        logger.info("New CLOSED M5 candle detected: %s — evaluating signal... session=%s", str(eval_candle_time), session)

        # ✅ FIXED: news check call
        high_news = _safe_news_check(now_utc=now_utc, block_on_unknown_news=block_on_unknown_news)

        # Trend (H1)
        trend = detect_trend_h1(h1_df)
        trend_source = "H1_TREND"
        adx_m5_last = 0.0

        # Momentum override if no H1 trend
        if trend is None and enable_momentum_override:
            override_trend, adx_m5_last = _momentum_override_trend(m5_df)
            if adx_m5_last >= adx_override_threshold and override_trend in ("BULL", "BEAR"):
                trend = override_trend
                trend_source = "M5_MOMENTUM"
                logger.info(
                    "Momentum override active (ADX_M5=%.2f). Using trend_dir=%s source=%s",
                    adx_m5_last, ("LONG" if trend == "BULL" else "SHORT"), trend_source
                )
            else:
                logger.info("No clear H1 trend and no momentum override (ADX_M5=%.2f). Skipping.", adx_m5_last)
                time.sleep(sleep_seconds)
                continue

        if trend is None:
            logger.info("No clear trend. Skipping.")
            time.sleep(sleep_seconds)
            continue

        signal = trigger_signal_m5(
            m5_df=m5_df,
            m15_df=m15_df,
            h1_df=h1_df,
            trend_dir=trend,  # "BULL" or "BEAR"
            high_news=high_news,
            min_score=min_entry_score,
            session=session,
            asia_extra_buffer=asia_extra_score_buffer,
            tp1_rr=tp1_rr,
            tp2_rr=tp2_rr,
            asia_tp1_rr=asia_tp1_rr,
            sl_atr_mult=sl_atr_mult,
        )

        if signal is None:
            logger.info("No signal on candle close (session=%s). Sleeping %ss.", session, sleep_seconds)
            time.sleep(sleep_seconds)
            continue

        entry_score = int(signal.extra.get("entry_score", 0))
        confidence_text = "High" if entry_score >= 80 else "Moderate"

        note = f"Trend source: {trend_source}"
        if trend_source == "M5_MOMENTUM":
            note += f" | ADX_M5={adx_m5_last:.2f}"

        msg = build_signal_message(
            signal=signal,
            symbol_label=symbol_label,
            session=session,
            confidence_text=confidence_text,
            note=note,
        )

        try:
            send_telegram_message(msg)
            logger.info("Signal sent to Telegram: %s %s @ %.2f (score=%s)",
                        signal.direction, symbol_label, signal.entry, entry_score)
        except Exception as e:
            logger.exception("Failed to send Telegram message: %s", e)

        time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()