import time
import logging
import datetime as dt
from typing import Optional, Tuple

import pandas as pd

from config import load_settings
from data_fetcher import fetch_m5_ohlcv_twelvedata
from telegram_client import TelegramClient
import indicators
from strategy import (
    is_within_sessions,
    detect_trend_h1,
    detect_trend_m15_direction,
    confirm_trend_m15,
    trigger_signal_m5,
)

# -------------------------
# Safe import for news
# -------------------------
try:
    from high_impact_news import has_high_impact_news_nearby as has_high_impact_news
except ImportError:
    from high_impact_news import has_high_impact_news_near as has_high_impact_news

# -------------------------
# Logging
# -------------------------
logger = logging.getLogger("xauusd_bot")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# -------------------------
# Resampling
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
# Market Classifiers
# -------------------------
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

def market_regime(h1_df: pd.DataFrame) -> str:
    close = h1_df["close"]
    high = h1_df["high"]
    low = h1_df["low"]

    upper, mid, lower = indicators.bollinger_bands(close)
    bw = (upper - lower) / mid.replace(0, pd.NA)
    atr_val = indicators.atr(high, low, close)

    bw_last = float(bw.iloc[-1])
    atr_last = float(atr_val.iloc[-1])

    if bw_last >= 0.02 or atr_last >= 20:
        return "High Volatility"
    if bw_last <= 0.008 and atr_last <= 10:
        return "Low Volatility / Compression"
    return "Normal Volatility"

# -------------------------
# TP / SL (ATR-based)
# -------------------------
def tp_sl_multipliers(setup_type: str) -> Tuple[float, float, float]:
    if setup_type.startswith("PULLBACK"):
        return (0.9, 1.6, 2.6)
    if setup_type.startswith("BREAKOUT_CONT"):
        return (0.55, 0.9, 1.5)
    if setup_type.startswith("BREAKOUT"):
        return (0.7, 1.2, 2.0)
    return (0.75, 1.1, 1.8)

def apply_dynamic_tp_sl(signal, h1_df: pd.DataFrame) -> None:
    atr_series = indicators.atr(
        h1_df["high"], h1_df["low"], h1_df["close"]
    )
    atr_h1 = float(atr_series.iloc[-1])
    sl_mult, tp1_mult, tp2_mult = tp_sl_multipliers(
        signal.extra.get("setup_type", "GENERIC")
    )

    entry = signal.price
    if signal.direction == "LONG":
        sl = entry - atr_h1 * sl_mult
        tp1 = entry + atr_h1 * tp1_mult
        tp2 = entry + atr_h1 * tp2_mult
    else:
        sl = entry + atr_h1 * sl_mult
        tp1 = entry - atr_h1 * tp1_mult
        tp2 = entry - atr_h1 * tp2_mult

    signal.extra.update(
        {"sl": sl, "tp1": tp1, "tp2": tp2, "atr_h1": atr_h1}
    )

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
# Telegram Message
# -------------------------
def build_message(signal, trend_dir, trend_source, conf, conf_text, market_state, regime):
    arrow = "BUY" if signal.direction == "LONG" else "SELL"
    e = signal.price

    return "\n".join(
        [
            "XAUUSD Signal [SCALP]",
            f"{arrow} XAUUSD at {e:.2f}",
            f"Setup: {signal.extra.get('setup_type')}",
            f"Confidence: {conf} ({conf_text})",
            f"SL: {signal.extra['sl']:.2f}",
            f"TP1: {signal.extra['tp1']:.2f}",
            f"TP2: {signal.extra['tp2']:.2f}",
            "",
            f"Time (UTC): {signal.time.isoformat()}",
            f"Trend Bias: {trend_dir} (source: {trend_source})",
            f"Market State (H1): {market_state}",
            f"Market Regime: {regime}",
        ]
    )

# -------------------------
# MAIN LOOP
# -------------------------
def main():
    settings = load_settings()
    telegram = TelegramClient(
        settings.telegram_bot_token, settings.telegram_chat_id
    )

    last_fetch = 0
    cached_m5: Optional[pd.DataFrame] = None

    logger.info("Starting XAUUSD bot (H1 primary, M15 fallback).")

    while True:
        now = dt.datetime.now(dt.timezone.utc)

        if not is_within_sessions(now, 700, 2000, None, None):
            time.sleep(60)
            continue

        if time.time() - last_fetch > 180 or cached_m5 is None:
            cached_m5 = fetch_m5_ohlcv_twelvedata(
                settings.xau_symbol_td, settings.twelvedata_api_key
            )
            last_fetch = time.time()

        m15 = resample_ohlc(cached_m5, "15min")
        h1 = resample_ohlc(cached_m5, "1h")

        adx_h1 = float(
            indicators.adx(h1["high"], h1["low"], h1["close"])[0].iloc[-1]
        )

        trend = detect_trend_h1(h1)
        trend_source = "H1"

        if trend is None:
            trend = detect_trend_m15_direction(m15)
            trend_source = "M15"

        if trend is None:
            time.sleep(60)
            continue

        signal = trigger_signal_m5(cached_m5, trend)
        if not signal:
            time.sleep(60)
            continue

        apply_dynamic_tp_sl(signal, h1)

        high_news = has_high_impact_news(now)
        conf = compute_confidence(
            trend_source,
            signal.extra["setup_type"],
            adx_h1,
            signal.extra["adx_m5"],
            high_news,
        )

        msg = build_message(
            signal,
            trend,
            trend_source,
            conf,
            confidence_label(conf),
            market_state_from_adx(adx_h1),
            market_regime(h1),
        )

        telegram.send_message(msg)
        logger.info("Signal sent successfully.")
        time.sleep(60)

if __name__ == "__main__":
    main()
