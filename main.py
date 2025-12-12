import time
import logging
import datetime as dt
from typing import Optional, Tuple

import pandas as pd

from config import load_settings
from telegram_client import TelegramClient
import indicators

# Strategy (your updated one)
from strategy import (
    is_within_sessions,
    detect_trend_h1,
    detect_trend_m15_direction,
    confirm_trend_m15,
    trigger_signal_m5,
)

# Data fetcher (matches your earlier data_fetcher.py that uses Settings)
from data_fetcher import fetch_m5_ohlcv_hybrid

# Optional: news flag (do not crash deploy if names change)
try:
    from high_impact_news import has_high_impact_news_nearby as _news_flag
except Exception:
    try:
        from high_impact_news import has_high_impact_news_near as _news_flag
    except Exception:
        _news_flag = None


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
    """
    rule: "15min" or "1h"
    Expects df with a datetime column (UTC-ish) or datetime index.
    """
    tmp = df.copy()

    if "datetime" in tmp.columns:
        # Force UTC-aware timestamps (prevents weird future timestamps + resample issues)
        tmp["datetime"] = pd.to_datetime(tmp["datetime"], utc=True, errors="coerce")
        tmp = tmp.dropna(subset=["datetime"]).set_index("datetime")
    else:
        tmp.index = pd.to_datetime(tmp.index, utc=True, errors="coerce")

    agg = tmp.resample(rule).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna()

    return agg.reset_index()


# -------------------------
# Labels / Classifiers
# -------------------------
def setup_label(setup_type: str) -> str:
    if not setup_type:
        return "Generic"
    if setup_type.startswith("PULLBACK"):
        return "Pullback"
    if setup_type.startswith("BREAKOUT_CONT"):
        return "Breakout Continuation"
    if setup_type.startswith("BREAKOUT"):
        return "Breakout"
    return "Generic"


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


def confidence_label(score: int) -> str:
    if score >= 75:
        return "High"
    if score >= 55:
        return "Medium"
    return "Low"


def risk_tag_from_adx_m5(adx_m5: float) -> str:
    return "SCALP"


def market_regime(h1_df: pd.DataFrame) -> str:
    if h1_df is None or len(h1_df) < 20:
        return "Unknown"

    close = h1_df["close"]
    high = h1_df["high"]
    low = h1_df["low"]

    upper, mid, lower = indicators.bollinger_bands(close, period=20, std_factor=2.0)
    bw = (upper - lower) / mid.replace(0, pd.NA)

    a = indicators.atr(high, low, close, period=14)

    bw_last = float(bw.iloc[-1]) if pd.notna(bw.iloc[-1]) else 0.0
    atr_last = float(a.iloc[-1]) if pd.notna(a.iloc[-1]) else 0.0

    if bw_last >= 0.02 or atr_last >= 20:
        return "High Volatility"
    if bw_last <= 0.008 and atr_last <= 10:
        return "Low Volatility / Compression"
    return "Normal Volatility"


# -------------------------
# Dynamic TP/SL (ATR-based)
# -------------------------
def tp_sl_multipliers(setup_type: str) -> Tuple[float, float, float]:
    """
    (sl_mult, tp1_mult, tp2_mult) on ATR(H1,14)
    """
    if setup_type.startswith("PULLBACK"):
        return (0.90, 1.60, 2.60)
    if setup_type.startswith("BREAKOUT_CONT"):
        return (0.55, 0.90, 1.50)
    if setup_type.startswith("BREAKOUT"):
        return (0.70, 1.20, 2.00)
    return (0.75, 1.10, 1.80)


def apply_dynamic_tp_sl(signal, h1_df: pd.DataFrame) -> None:
    # We only need ~20 H1 candles to compute ATR(14) safely
    if h1_df is None or len(h1_df) < 20:
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
# Confidence Score
# -------------------------
def compute_confidence(
    trend_source: str,
    setup_type: str,
    adx_h1: float,
    adx_m5: float,
    high_news: bool,
) -> int:
    score = 50

    score += 10 if trend_source == "H1" else 5

    if setup_type.startswith("PULLBACK"):
        score += 15
    elif setup_type.startswith("BREAKOUT"):
        score += 8
    elif setup_type.startswith("BREAKOUT_CONT"):
        score += 3

    if adx_h1 >= 30:
        score += 12
    elif adx_h1 >= 20:
        score += 7
    else:
        score -= 6

    if adx_m5 >= 25:
        score += 6
    elif adx_m5 < 15:
        score -= 5

    if high_news:
        score -= 12

    return max(0, min(100, score))


# -------------------------
# Telegram message (PLAIN TEXT)
# -------------------------
def build_signal_message(
    symbol_label: str,
    signal,
    trend_label: str,
    trend_source: str,
    session_window: str,
    high_news: bool,
    market_state: str,
    market_regime_text: str,
    adx_h1: float,
    trend_strength: str,
    confidence_score: int,
    confidence_text: str,
) -> str:
    entry = float(signal.price)
    setup_type = signal.extra.get("setup_type", "GENERIC")
    setup_text = setup_label(setup_type)

    adx_m5 = float(signal.extra.get("adx_m5", 0.0))
    risk_tag = risk_tag_from_adx_m5(adx_m5)

    sl = signal.extra.get("sl")
    tp1 = signal.extra.get("tp1")
    tp2 = signal.extra.get("tp2")
    atr_h1 = signal.extra.get("atr_h1")

    arrow = "BUY" if signal.direction == "LONG" else "SELL"

    lines = []
    lines.append(f"XAUUSD Signal [{risk_tag}]")
    lines.append(f"{arrow} {symbol_label} at {entry:.2f}")
    lines.append(f"Setup: {setup_text} ({setup_type})")
    lines.append(f"Confidence: {confidence_score} ({confidence_text})")

    if sl is not None and tp1 is not None and tp2 is not None:
        lines.append(f"SL: {float(sl):.2f}")
        lines.append(f"TP1: {float(tp1):.2f}")
        lines.append(f"TP2: {float(tp2):.2f}")

        if signal.direction == "LONG":
            risk = entry - float(sl)
            rr1 = (float(tp1) - entry) / risk if risk > 0 else 0.0
            rr2 = (float(tp2) - entry) / risk if risk > 0 else 0.0
        else:
            risk = float(sl) - entry
            rr1 = (entry - float(tp1)) / risk if risk > 0 else 0.0
            rr2 = (entry - float(tp2)) / risk if risk > 0 else 0.0

        if risk > 0:
            lines.append(f"RR to TP1: {rr1:.2f}R | RR to TP2: {rr2:.2f}R")

    lines.append("")
    lines.append(f"Time (UTC): {signal.time.isoformat()}")
    lines.append(f"Trend Bias: {trend_label} (source: {trend_source})")
    lines.append(f"Session: {session_window}")
    lines.append(f"Reason: {signal.reason}")
    lines.append("")
    lines.append(f"Market State (H1): {market_state} (ADX {adx_h1:.2f})")
    lines.append(f"Trend Strength (H1): {trend_strength}")
    lines.append(f"Market Regime: {market_regime_text}")
    lines.append("")
    if atr_h1 is not None:
        lines.append(f"ATR(H1,14): {float(atr_h1):.2f}")
    lines.append("")
    lines.append("HIGH-IMPACT NEWS NEARBY: expect extra volatility." if high_news else "No high-impact news flag near this time.")
    lines.append(f"RSI(M5): {float(signal.extra.get('m5_rsi', 0.0)):.2f} | StochK(M5): {float(signal.extra.get('m5_stoch_k', 0.0)):.2f}")
    lines.append(f"ADX(M5): {float(signal.extra.get('adx_m5', 0.0)):.2f} (+DI: {float(signal.extra.get('plus_di_m5', 0.0)):.2f}, -DI: {float(signal.extra.get('minus_di_m5', 0.0)):.2f})")
    lines.append("BB(M5): upper {0:.2f}, mid {1:.2f}, lower {2:.2f}".format(
        float(signal.extra.get("bb_upper", 0.0)),
        float(signal.extra.get("bb_mid", 0.0)),
        float(signal.extra.get("bb_lower", 0.0)),
    ))

    return "\n".join(lines)


# -------------------------
# Main loop
# -------------------------
def main():
    settings = load_settings()

    symbol_label = "XAUUSD"
    s1_start = settings.session_1_start
    s1_end = settings.session_1_end

    # Polling (how often the bot wakes up)
    sleep_seconds = 30  # feels responsive; still respects candle-close logic

    # Fetch throttling (how often we call TwelveData)
    fetch_interval_seconds = 180  # prevents API spam

    telegram = TelegramClient(settings.telegram_bot_token, settings.telegram_chat_id)

    logger.info("Starting XAUUSD bot (H1 primary, M15 fallback) + NEW M5 candle detection.")

    last_fetch_ts: float = 0.0
    cached_m5: Optional[pd.DataFrame] = None
    last_closed_candle_time: Optional[pd.Timestamp] = None

    while True:
        now_utc = dt.datetime.now(dt.timezone.utc)

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

        need_fetch = (time.time() - last_fetch_ts) >= fetch_interval_seconds or cached_m5 is None
        if need_fetch:
            logger.info("Fetching fresh M5 OHLCV data from Twelve Data...")
            # Your fetcher currently uses Settings and returns M5 candles.
            # IMPORTANT: ensure outputsize is large enough in data_fetcher (1000+),
            # otherwise you won't have enough H1 history.
            cached_m5 = fetch_m5_ohlcv_hybrid(settings)
            last_fetch_ts = time.time()
        else:
            logger.info("Using cached M5 OHLCV data.")

        if cached_m5 is None or len(cached_m5) < 120:
            logger.warning("Not enough M5 data yet. Sleeping %ss...", sleep_seconds)
            time.sleep(sleep_seconds)
            continue

        # --- NEW M5 candle detection ---
        cached_m5["datetime"] = pd.to_datetime(cached_m5["datetime"], utc=True, errors="coerce")
        cached_m5 = cached_m5.dropna(subset=["datetime"])

        current_last = cached_m5["datetime"].iloc[-1]
        if last_closed_candle_time is not None and current_last <= last_closed_candle_time:
            logger.info(
                "No new M5 candle closed yet (last=%s, current=%s). Sleeping %ss...",
                last_closed_candle_time,
                current_last,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)
            continue

        last_closed_candle_time = current_last
        logger.info("New M5 candle detected: %s — evaluating signal...", current_last)

        # Resample
        m15_df = resample_ohlc(cached_m5, "15min")
        h1_df = resample_ohlc(cached_m5, "1h")

        # Realistic minimums (do NOT require 60 H1 candles unless you fetch 60 hours of M5)
        if h1_df is None or len(h1_df) < 20 or m15_df is None or len(m15_df) < 40:
            logger.info("Not enough data after resampling, sleeping %ss...", sleep_seconds)
            time.sleep(sleep_seconds)
            continue

        # ADX(H1)
        adx_h1_series, _, _ = indicators.adx(h1_df["high"], h1_df["low"], h1_df["close"], period=14)
        adx_h1 = float(adx_h1_series.iloc[-1]) if pd.notna(adx_h1_series.iloc[-1]) else 0.0

        # Trend detection
        h1_trend = detect_trend_h1(h1_df)
        trend_source = "H1"

        if h1_trend is None:
            # H1 ranging -> bias with M15 direction
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

        trend_label = "LONG" if trend_dir == "LONG" else "SHORT"

        # Market classifiers
        market_state = market_state_from_adx(adx_h1)
        trend_strength = trend_strength_from_adx(adx_h1)
        regime = market_regime(h1_df)

        # News flag
        high_news = False
        if _news_flag is not None:
            try:
                high_news = bool(_news_flag(now_utc))
            except Exception:
                high_news = False

        # M5 Trigger (your merged pullback/breakout/continuation)
        signal = trigger_signal_m5(cached_m5, trend_dir)
        if not signal:
            logger.info("No M5 trigger signal on candle close, sleeping %ss.", sleep_seconds)
            time.sleep(sleep_seconds)
            continue

        # Dynamic TP/SL
        apply_dynamic_tp_sl(signal, h1_df)

        # Confidence
        setup_type = signal.extra.get("setup_type", "GENERIC")
        adx_m5 = float(signal.extra.get("adx_m5", 0.0))

        conf = compute_confidence(
            trend_source=trend_source,
            setup_type=setup_type,
            adx_h1=adx_h1,
            adx_m5=adx_m5,
            high_news=high_news,
        )
        conf_text = confidence_label(conf)

        # Message
        msg = build_signal_message(
            symbol_label=symbol_label,
            signal=signal,
            trend_label=trend_label,
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
        logger.info(
            "Signal processed. Trend source=%s direction=%s setup=%s confidence=%s (%s)",
            trend_source,
            trend_label,
            setup_type,
            conf,
            conf_text,
        )

        time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()
