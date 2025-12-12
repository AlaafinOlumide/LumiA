import time
import logging
import datetime as dt
from typing import Optional, Tuple, Dict, Any

import pandas as pd

import config
from data_fetcher import fetch_m5_ohlcv_twelvedata
from telegram_client import TelegramClient
from high_impact_news import has_high_impact_news_nearby

from strategy import (
    is_within_sessions,
    detect_trend_h1,
    detect_trend_m15_direction,
    confirm_trend_m15,
    trigger_signal_m5,
)

# -------------------------
# Logging
# -------------------------
logger = logging.getLogger("xauusd_bot")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# -------------------------
# Helpers: Indicators (ATR, BB width)
# -------------------------
def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range (Wilder-style smoothing via EMA approximation)."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def bollinger_width(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.Series:
    ma = close.rolling(period).mean()
    sd = close.rolling(period).std(ddof=0)
    upper = ma + std_dev * sd
    lower = ma - std_dev * sd
    width = (upper - lower) / ma.replace(0, pd.NA)
    return width


def _risk_tag_from_adx(adx_m5: float) -> str:
    # Keep it simple and useful in Telegram
    if adx_m5 >= 25:
        return "SCALP"
    return "SCALP"


def _confidence_label(score: int) -> str:
    if score >= 75:
        return "High"
    if score >= 55:
        return "Medium"
    return "Low"


def _trend_strength_label(adx_h1: float) -> str:
    if adx_h1 >= 30:
        return "Strong"
    if adx_h1 >= 20:
        return "Moderate"
    return "Weak"


def _market_state_from_adx(adx_h1: float) -> str:
    # Simple, robust
    return "TRENDING" if adx_h1 >= 20 else "RANGING"


def _market_regime(h1_df: pd.DataFrame) -> str:
    """
    Regime = directionless description of market behavior.
    Uses BB width + ATR to classify volatility + structure.
    """
    if h1_df is None or len(h1_df) < 60:
        return "Unknown"

    close = h1_df["close"]
    high = h1_df["high"]
    low = h1_df["low"]

    bw = bollinger_width(close, period=20, std_dev=2.0)
    a = atr(high, low, close, period=14)

    bw_last = float(bw.iloc[-1]) if pd.notna(bw.iloc[-1]) else 0.0
    atr_last = float(a.iloc[-1]) if pd.notna(a.iloc[-1]) else 0.0

    # crude but stable thresholds
    if bw_last >= 0.02 or atr_last >= 20:
        return "High Volatility"
    if bw_last <= 0.008 and atr_last <= 10:
        return "Low Volatility / Compression"
    return "Normal Volatility"


def _setup_label(setup_type: str) -> str:
    if setup_type.startswith("PULLBACK"):
        return "Pullback"
    if setup_type.startswith("BREAKOUT_CONT"):
        return "Breakout Continuation"
    if setup_type.startswith("BREAKOUT"):
        return "Breakout"
    return "Generic"


def _tp_sl_multipliers(setup_type: str) -> Tuple[float, float, float]:
    """
    (sl_mult, tp1_mult, tp2_mult) based on setup.
    - Pullback: wider SL, larger targets
    - Breakout: medium
    - Continuation: tighter SL, smaller targets
    """
    if setup_type.startswith("PULLBACK"):
        return (0.90, 1.60, 2.60)
    if setup_type.startswith("BREAKOUT_CONT"):
        return (0.55, 0.90, 1.50)
    if setup_type.startswith("BREAKOUT"):
        return (0.70, 1.20, 2.00)
    return (0.75, 1.10, 1.80)


def apply_dynamic_tp_sl(signal, h1_df: pd.DataFrame) -> None:
    """
    Adds:
      - atr_h1
      - sl, tp1, tp2
    into signal.extra using ATR(H1,14) and setup-specific multipliers.
    """
    if h1_df is None or len(h1_df) < 60:
        return

    high = h1_df["high"]
    low = h1_df["low"]
    close = h1_df["close"]

    a = atr(high, low, close, period=14)
    atr_h1 = float(a.iloc[-1]) if pd.notna(a.iloc[-1]) else None
    if atr_h1 is None or atr_h1 <= 0:
        return

    setup_type = signal.extra.get("setup_type", "GENERIC")
    sl_mult, tp1_mult, tp2_mult = _tp_sl_multipliers(setup_type)

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


def compute_confidence_score(
    trend_source: str,
    trend_dir: str,
    setup_type: str,
    adx_h1: float,
    adx_m5: float,
    high_news: bool,
) -> int:
    """
    Simple, explainable confidence score (0-100).
    """
    score = 50

    # Trend source reliability
    if trend_source == "H1":
        score += 10
    elif trend_source == "M15":
        score += 5

    # Setup quality bonus
    if setup_type.startswith("PULLBACK"):
        score += 15
    elif setup_type.startswith("BREAKOUT"):
        score += 8
    elif setup_type.startswith("BREAKOUT_CONT"):
        score += 3

    # Trend strength bonus
    if adx_h1 >= 30:
        score += 12
    elif adx_h1 >= 20:
        score += 7
    else:
        score -= 5

    # M5 structure bonus
    if adx_m5 >= 25:
        score += 6
    elif adx_m5 < 15:
        score -= 5

    # News penalty
    if high_news:
        score -= 12

    # clamp
    score = max(0, min(100, score))
    return score


# -------------------------
# Resampling (M5 -> M15/H1)
# -------------------------
def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    rule: "15min" or "1h"
    Expects df with datetime column OR datetime index.
    """
    tmp = df.copy()
    if "datetime" in tmp.columns:
        tmp["datetime"] = pd.to_datetime(tmp["datetime"], utc=True)
        tmp = tmp.set_index("datetime")
    else:
        tmp.index = pd.to_datetime(tmp.index, utc=True)

    agg = tmp.resample(rule).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna()

    agg = agg.reset_index()
    return agg


# -------------------------
# Telegram message builder
# -------------------------
def build_signal_message(
    symbol_label: str,
    signal,
    trend_label: str,
    trend_source: str,
    session_window: str,
    high_news: bool,
    market_state: str,
    market_regime: str,
    adx_h1_value: float,
    trend_strength_label: str,
    confidence_score: int,
    confidence_label: str,
) -> str:
    adx_m5 = float(signal.extra.get("adx_m5", 0.0))
    risk_tag = _risk_tag_from_adx(adx_m5)

    entry = float(signal.price)
    setup_type = signal.extra.get("setup_type", "GENERIC")
    setup_label = _setup_label(setup_type)

    sl = signal.extra.get("sl")
    tp1 = signal.extra.get("tp1")
    tp2 = signal.extra.get("tp2")
    atr_h1 = signal.extra.get("atr_h1")

    arrow = "🟢 BUY" if signal.direction == "LONG" else "🔴 SELL"

    lines = []
    lines.append(f"XAUUSD Signal [{risk_tag}]")
    lines.append(f"{arrow} {symbol_label} at {entry:.2f}")
    lines.append(f"Setup: {setup_label} ({setup_type}) | Confidence: {confidence_score} ({confidence_label})")

    if sl is not None and tp1 is not None and tp2 is not None:
        lines.append(f"– SL: {sl:.2f}")
        lines.append(f"– TP1: {tp1:.2f}")
        lines.append(f"– TP2: {tp2:.2f}")

        # RR
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

    lines.append(f"Market State (H1): {market_state} (ADX {adx_h1_value:.2f})")
    lines.append(f"Market Regime: {market_regime}")
    lines.append(f"Trend Strength (H1): {trend_strength_label}")
    lines.append("")

    lines.append("Suggested TP/SL (ATR-based)")
    if atr_h1 is not None:
        lines.append(f"– ATR(H1, 14): {float(atr_h1):.2f}")
    lines.append("")

    if high_news:
        lines.append("⚠️ HIGH-IMPACT NEWS NEARBY — expect extra volatility.")
    else:
        lines.append("ℹ️ No high-impact news flag near this time.")

    lines.append(
        f"RSI(M5): {float(signal.extra.get('m5_rsi', 0.0)):.2f} | "
        f"StochK(M5): {float(signal.extra.get('m5_stoch_k', 0.0)):.2f}"
    )
    lines.append(
        f"ADX(M5): {float(signal.extra.get('adx_m5', 0.0)):.2f} "
        f"(+DI: {float(signal.extra.get('plus_di_m5', 0.0)):.2f}, "
        f"-DI: {float(signal.extra.get('minus_di_m5', 0.0)):.2f})"
    )
    lines.append(
        "BB(M5): upper {0:.2f}, mid {1:.2f}, lower {2:.2f}".format(
            float(signal.extra.get("bb_upper", 0.0)),
            float(signal.extra.get("bb_mid", 0.0)),
            float(signal.extra.get("bb_lower", 0.0)),
        )
    )

    return "\n".join(lines)


# -------------------------
# Main loop
# -------------------------
def main():
    symbol = getattr(config, "SYMBOL_TWELVE", "XAU/USD")
    symbol_label = getattr(config, "SYMBOL_LABEL", "XAUUSD")

    # Trading window: 07:00–20:00 UTC (single session)
    s1_start = getattr(config, "TRADING_SESSION_1_START", 700)
    s1_end = getattr(config, "TRADING_SESSION_1_END", 2000)
    s2_start = getattr(config, "TRADING_SESSION_2_START", None)
    s2_end = getattr(config, "TRADING_SESSION_2_END", None)

    sleep_seconds = getattr(config, "SLEEP_SECONDS", 60)
    fetch_interval_seconds = getattr(config, "FETCH_INTERVAL_SECONDS", 180)  # reduces API usage

    td_key = getattr(config, "TWELVE_DATA_API_KEY", "")
    tg_token = getattr(config, "TELEGRAM_BOT_TOKEN", "")
    tg_chat = getattr(config, "TELEGRAM_CHAT_ID", "")

    telegram = TelegramClient(tg_token, tg_chat)

    logger.info("Starting XAUUSD bot (Twelve Data only, H1 primary, M15 fallback).")

    last_fetch_ts: float = 0.0
    cached_m5: Optional[pd.DataFrame] = None

    while True:
        now_utc = dt.datetime.now(dt.timezone.utc)

        in_session = is_within_sessions(
            now_utc=now_utc,
            session_1_start=s1_start,
            session_1_end=s1_end,
            session_2_start=s2_start,
            session_2_end=s2_end,
        )

        if not in_session:
            logger.info("Outside trading sessions, sleeping %ss...", sleep_seconds)
            time.sleep(sleep_seconds)
            continue

        # Fetch / cache M5 OHLCV
        need_fetch = (time.time() - last_fetch_ts) >= fetch_interval_seconds or cached_m5 is None
        if need_fetch:
            logger.info("Fetching fresh M5 OHLCV data from Twelve Data...")
            cached_m5 = fetch_m5_ohlcv_twelvedata(symbol=symbol, api_key=td_key)
            last_fetch_ts = time.time()
        else:
            logger.info("Using cached M5 OHLCV data.")

        if cached_m5 is None or len(cached_m5) < 100:
            logger.warning("Not enough M5 data, sleeping %ss...", sleep_seconds)
            time.sleep(sleep_seconds)
            continue

        # Build M15/H1 from M5 (same source)
        m15_df = resample_ohlc(cached_m5, "15min")
        h1_df = resample_ohlc(cached_m5, "1h")

        if h1_df is None or len(h1_df) < 60 or m15_df is None or len(m15_df) < 60:
            logger.info("Not enough data after resampling, sleeping %ss.", sleep_seconds)
            time.sleep(sleep_seconds)
            continue

        # Trend detection
        h1_trend = detect_trend_h1(h1_df)
        trend_source = "H1"

        if h1_trend is None:
            # If H1 ranging/unclear → use M15 direction as primary for signals
            m15_dir = detect_trend_m15_direction(m15_df)
            if m15_dir is None:
                logger.info("No clear H1 trend and no clear M15 direction, skipping.")
                time.sleep(sleep_seconds)
                continue
            trend_dir = m15_dir
            trend_source = "M15"
        else:
            # Confirm H1 with M15 if possible; if fails, allow anyway but it may lower confidence later.
            trend_dir = h1_trend
            _ = confirm_trend_m15(m15_df, trend_dir)

        # Market classifications (H1)
        # Compute ADX(H1) via ATR/BB doesn't give ADX; so we approximate market state from trend presence + BB/ATR regime.
        # We'll use trend existence as directional and treat ADX as "proxy" by trend detection strength:
        # But we *can* still infer an "ADX-like" from trend_dir presence only — better: classify from h1_trend presence.
        # Here we do simple state/strength using trend presence:
        # If you already compute ADX(H1) elsewhere, you can swap in.
        # We'll estimate adx_h1 as 25 when trend exists, else 15.
        adx_h1_value = 25.0 if h1_trend is not None else 15.0
        market_state = _market_state_from_adx(adx_h1_value)
        trend_strength_label = _trend_strength_label(adx_h1_value)
        market_regime = _market_regime(h1_df)

        # News filter
        try:
            high_news = has_high_impact_news_nearby(now_utc)
        except Exception:
            high_news = False

        # M5 trigger (merged logic)
        signal = trigger_signal_m5(cached_m5, trend_dir)
        if not signal:
            logger.info("No M5 trigger signal, sleeping %ss.", sleep_seconds)
            time.sleep(sleep_seconds)
            continue

        # Dynamic TP/SL from ATR(H1)
        apply_dynamic_tp_sl(signal, h1_df)

        # Confidence
        setup_type = signal.extra.get("setup_type", "GENERIC")
        adx_m5 = float(signal.extra.get("adx_m5", 0.0))

        confidence_score = compute_confidence_score(
            trend_source=trend_source,
            trend_dir=trend_dir,
            setup_type=setup_type,
            adx_h1=adx_h1_value,
            adx_m5=adx_m5,
            high_news=high_news,
        )
        confidence_label = _confidence_label(confidence_score)

        # Message
        session_window = "07:00-20:00"
        trend_label = "LONG" if trend_dir == "LONG" else "SHORT"

        msg = build_signal_message(
            symbol_label=symbol_label,
            signal=signal,
            trend_label=trend_label,
            trend_source=trend_source,
            session_window=session_window,
            high_news=high_news,
            market_state=market_state,
            market_regime=market_regime,
            adx_h1_value=adx_h1_value,
            trend_strength_label=trend_strength_label,
            confidence_score=confidence_score,
            confidence_label=confidence_label,
        )

        # Send Telegram
        telegram.send_message(msg)
        logger.info(
            "Signal sent and logged. Trend source: %s, direction: %s, setup: %s, confidence: %s (%s)",
            trend_source,
            trend_label,
            setup_type,
            confidence_score,
            confidence_label,
        )

        time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()
