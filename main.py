import time
import logging
import datetime as dt
from typing import Optional

import pandas as pd

from config import load_settings, Settings
from telegram_client import TelegramClient
from data_fetcher import fetch_m5_ohlcv_hybrid  import time
import logging
import datetime as dt
from typing import Optional

import pandas as pd

from config import load_settings, Settings
from telegram_client import TelegramClient
from data_fetcher import fetch_m5_ohlcv_hybrid  # Twelve Data only in your setup
from strategy import (
    detect_trend_h1,
    detect_trend_m15_direction,
    confirm_trend_m15,
    trigger_signal_m5,
    is_within_sessions,
)
from data_logger import log_signal
from high_impact_news import has_high_impact_news_near
from indicators import atr, adx  # ATR + ADX on H1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("xauusd_bot")

# Fetch 5m data at most once every 120 seconds (2 minutes)
FETCH_INTERVAL_SECONDS = 120


def _risk_tag_from_adx(adx_m5: float) -> str:
    """
    Classify the trade idea as SCALP vs SWING based on M5 ADX strength.
    """
    if adx_m5 >= 30:
        return "SWING"
    return "SCALP"


def _trend_strength_label(adx_h1_value: float) -> str:
    """
    Give a human label for H1 trend strength based on ADX.
    """
    if adx_h1_value < 18:
        return "Weak"
    elif adx_h1_value < 25:
        return "Moderate"
    elif adx_h1_value < 35:
        return "Strong"
    else:
        return "Very Strong"


def _market_state_and_regime(
    adx_h1_value: float,
    atr_h1: float,
    last_h1_close: float,
) -> tuple[str, str]:
    """
    Coarse market state (TRENDING / RANGING) + finer regime label
    using ADX(H1) and ATR(H1) as % of price.
    """
    atr_ratio = atr_h1 / last_h1_close if last_h1_close > 0 else 0.0

    # Coarse state
    if adx_h1_value < 18:
        state = "RANGING"
    else:
        state = "TRENDING"

    # Volatility label from ATR ratio
    if atr_ratio < 0.004:  # < 0.4% of price
        vol_label = "Low vol"
    elif atr_ratio < 0.008:  # 0.4–0.8%
        vol_label = "Normal vol"
    else:  # > 0.8%
        vol_label = "High vol"

    # Regime
    if state == "RANGING":
        if vol_label == "Low vol":
            regime = "Quiet Range"
        else:
            regime = "Choppy Range"
    else:  # TRENDING
        if vol_label == "Low vol":
            regime = "Slow Trend"
        elif vol_label == "Normal vol":
            regime = "Steady Trend"
        else:
            regime = "High-Vol Trend"

    regime_full = f"{regime} ({vol_label})"
    return state, regime_full


def _trading_confidence_score(
    adx_h1_value: float,
    adx_m5_value: float,
    market_state: str,
    regime: str,
    risk_tag: str,
    high_news: bool,
) -> tuple[int, str]:
    """
    Build a 0–100 confidence score and label (Low/Medium/High)
    based on trend strength, regime, risk tag, and news.
    """
    score = 50.0

    # Add from H1 trend strength, capped
    score += min(adx_h1_value, 40.0) * 0.8  # max +32

    # Slight boost if M5 ADX is strong
    score += min(adx_m5_value, 40.0) * 0.2  # max +8

    # SWING trades get a small bump
    if risk_tag == "SWING":
        score += 5.0

    # Ranging markets and choppy ranges reduce confidence
    if market_state == "RANGING":
        score -= 10.0
    if "Range" in regime:
        score -= 5.0

    # High-impact news reduces confidence
    if high_news:
        score -= 10.0

    # Clamp 0–100
    score = max(0.0, min(100.0, score))
    score_int = int(round(score))

    if score_int <= 40:
        label = "Low"
    elif score_int <= 70:
        label = "Medium"
    else:
        label = "High"

    return score_int, label


def build_signal_message(
    symbol_label: str,
    signal,
    trend_label: str,
    session_window: str,
    high_news: bool,
    market_state: str,
    market_regime: str,
    adx_h1_value: float,
    trend_strength_label: str,
    confidence_score: int,
    confidence_label: str,
) -> str:
    """
    Build Telegram message with:

    - Header, direction, SL, TP1, TP2
    - RR to TP1/TP2
    - Time, trend bias, session, reason
    - Market state, regime, trend strength, confidence
    - ATR info
    - News + key indicators
    """
    adx_m5 = signal.extra["adx_m5"]
    risk_tag = _risk_tag_from_adx(adx_m5)

    atr_h1 = signal.extra.get("atr_h1")
    sl = signal.extra.get("sl")
    tp1 = signal.extra.get("tp1")
    tp2 = signal.extra.get("tp2")

    arrow = "🟢 BUY" if signal.direction == "LONG" else "🔴 SELL"

    if high_news:
        news_line = "⚠️ HIGH-IMPACT NEWS NEARBY — expect extra volatility."
    else:
        news_line = "ℹ️ No high-impact news flag near this time."

    entry = signal.price
    setup_type = signal.extra.get("setup_type", "GENERIC")

    lines: list[str] = []

    # ----- Header + entry + TP/SL + RR -----
    lines.append(f"XAUUSD Signal [{risk_tag}]")
    lines.append(f"{arrow} {symbol_label} at {entry:.2f}")
    if setup_type != "GENERIC":
        lines.append(f"Setup: {setup_type}")

    if sl is not None and tp1 is not None and tp2 is not None:
        lines.append(f"– SL: {sl:.2f}")
        lines.append(f"– TP1: {tp1:.2f}")
        lines.append(f"– TP2: {tp2:.2f}")

        # Risk / Reward computation
        if signal.direction == "LONG":
            risk = entry - sl
            rr1 = (tp1 - entry) / risk if risk > 0 else 0.0
            rr2 = (tp2 - entry) / risk if risk > 0 else 0.0
        else:
            risk = sl - entry
            rr1 = (entry - tp1) / risk if risk > 0 else 0.0
            rr2 = (entry - tp2) / risk if risk > 0 else 0.0

        if risk > 0:
            lines.append(
                f"RR to TP1: {rr1:.2f}R | RR to TP2: {rr2:.2f}R"
            )

    lines.append("")  # blank

    # ----- Context -----
    lines.append(f"Time (UTC): {signal.time.isoformat()}")
    lines.append(f"Trend Bias: {trend_label}")
    lines.append(f"Session: {session_window}")
    lines.append(f"Reason: {signal.reason}")
    lines.append("")  # blank

    # ----- Market state / regime / trend strength / confidence -----
    lines.append(
        f"Market State (H1): {market_state} (ADX {adx_h1_value:.2f})"
    )
    lines.append(f"Market Regime: {market_regime}")
    lines.append(
        f"Trend Strength (H1): {trend_strength_label}"
    )
    lines.append(
        f"Trading Confidence: {confidence_score} ({confidence_label})"
    )
    lines.append("")  # blank

    # ----- ATR info -----
    lines.append("Suggested TP/SL (ATR-based)")
    if atr_h1 is not None:
        lines.append(f"– ATR(H1, 14): {atr_h1:.2f}")
    lines.append("")  # blank

    # ----- News & indicators -----
    lines.append(news_line)
    lines.append(
        f"RSI(M5): {signal.extra['m5_rsi']:.2f} | "
        f"StochK(M5): {signal.extra['m5_stoch_k']:.2f}"
    )
    lines.append(
        f"ADX(M5): {signal.extra['adx_m5']:.2f} "
        f"(+DI: {signal.extra['plus_di_m5']:.2f}, "
        f"-DI: {signal.extra['minus_di_m5']:.2f})"
    )
    lines.append(
        "BB(M5): upper {0:.2f}, mid {1:.2f}, lower {2:.2f}".format(
            signal.extra["bb_upper"],
            signal.extra["bb_mid"],
            signal.extra["bb_lower"],
        )
    )

    return "\n".join(lines)


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Resample a 5m OHLCV DataFrame to a higher timeframe (15m, 1h, etc.).
    Assumes df has columns: datetime, open, high, low, close, volume.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    tmp = df.copy()
    tmp = tmp.set_index("datetime")
    agg = tmp.resample(rule).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    agg = agg.dropna()
    agg = agg.reset_index()
    return agg


def main_loop():
    settings: Settings = load_settings()
    tg = TelegramClient(settings.telegram_bot_token, settings.telegram_chat_id)

    symbol_label = "XAUUSD"
    logger.info("Starting XAUUSD bot (Twelve Data only, H1 primary, M15 fallback).")

    last_signal_time: Optional[dt.datetime] = None

    cached_m5_df: Optional[pd.DataFrame] = None
    last_m5_fetch_ts: Optional[float] = None

    while True:
        now_utc = dt.datetime.utcnow()

        if not is_within_sessions(
            now_utc,
            settings.session_1_start,
            settings.session_1_end,
            settings.session_2_start,
            settings.session_2_end,
        ):
            logger.info("Outside trading sessions, sleeping 60s...")
            time.sleep(60)
            continue

        try:
            # ---------- DATA FETCH / CACHE ----------
            now_ts = time.time()
            should_fetch_m5 = (
                last_m5_fetch_ts is None
                or (now_ts - last_m5_fetch_ts) >= FETCH_INTERVAL_SECONDS
            )

            if should_fetch_m5:
                logger.info("Fetching fresh M5 OHLCV data from Twelve Data...")
                m5_df = fetch_m5_ohlcv_hybrid(settings)
                cached_m5_df = m5_df
                last_m5_fetch_ts = now_ts
            else:
                if cached_m5_df is None:
                    logger.info("No cached M5 data yet, fetching from Twelve Data...")
                    m5_df = fetch_m5_ohlcv_hybrid(settings)
                    cached_m5_df = m5_df
                    last_m5_fetch_ts = now_ts
                else:
                    logger.info("Using cached M5 OHLCV data.")
                    m5_df = cached_m5_df

            if m5_df is None or m5_df.empty:
                logger.info("Empty M5 data, sleeping 60s.")
                time.sleep(60)
                continue

            # ---------- RESAMPLING ----------
            h1_df = resample_ohlcv(m5_df, "1h")
            m15_df = resample_ohlcv(m5_df, "15min")

            if len(h1_df) < 20 or len(m15_df) < 20 or len(m5_df) < 50:
                logger.info("Not enough data after resampling, sleeping 60s.")
                time.sleep(60)
                continue

            # ---------- TREND SELECTION: H1 PRIMARY, M15 FALLBACK ----------
            trend_h1 = detect_trend_h1(h1_df)
            trend_source = "H1"
            trend_for_signal: Optional[str] = trend_h1

            if trend_for_signal is not None:
                # When H1 has a direction, we still require M15 confirmation
                if not confirm_trend_m15(m15_df, trend_h1):
                    logger.info("M15 does not confirm H1 trend, skipping.")
                    time.sleep(60)
                    continue
            else:
                # H1 is ranging / unclear -> try M15 as fallback
                trend_m15_bias = detect_trend_m15_direction(m15_df)
                if trend_m15_bias is None:
                    logger.info("No clear H1 or M15 trend, skipping.")
                    time.sleep(60)
                    continue
                trend_source = "M15"
                trend_for_signal = trend_m15_bias

            # Safety check
            if trend_for_signal is None:
                logger.info("No usable trend_for_signal, skipping.")
                time.sleep(60)
                continue

            # ---------- M5 TRIGGER ----------
            signal = trigger_signal_m5(m5_df, trend_for_signal)
            if not signal:
                logger.info("No M5 trigger signal, sleeping 60s.")
                time.sleep(60)
                continue

            # Cooldown: avoid signal spam
            if last_signal_time and (now_utc - last_signal_time).total_seconds() < 300:
                logger.info(
                    "Signal occurred too soon after previous, skipping (cooldown)."
                )
                time.sleep(60)
                continue

            # ---------- H1 VOL / ADX / STATE / REGIME ----------
            atr_series = atr(h1_df["high"], h1_df["low"], h1_df["close"], period=14)
            atr_h1 = float(atr_series.iloc[-1])

            adx_h1_series, _, _ = adx(
                h1_df["high"], h1_df["low"], h1_df["close"], period=14
            )
            adx_h1_value = float(adx_h1_series.iloc[-1])

            last_h1_close = float(h1_df["close"].iloc[-1])

            market_state, market_regime = _market_state_and_regime(
                adx_h1_value, atr_h1, last_h1_close
            )
            trend_strength_label = _trend_strength_label(adx_h1_value)

            # ---------- RISK TAG & SETUP TYPE ----------
            adx_m5 = signal.extra["adx_m5"]
            risk_tag = _risk_tag_from_adx(adx_m5)
            setup_type = signal.extra.get("setup_type", "GENERIC")

            # ---------- DYNAMIC TP/SL (ATR + SCALP/SWING + SETUP TYPE) ----------
            # Idea:
            # - Breakouts: tighter SL, nearer TP (fast continuation)
            # - Pullbacks: slightly wider SL, further TP (deeper swings)
            if risk_tag == "SCALP":
                if "BREAKOUT" in setup_type:
                    # Fast scalp breakout
                    sl_mult = 0.5
                    tp1_mult = 0.8
                    tp2_mult = 1.2
                elif "PULLBACK" in setup_type:
                    # Pullback scalp, can target a bit more
                    sl_mult = 0.6
                    tp1_mult = 1.0
                    tp2_mult = 1.6
                else:
                    # Fallback defaults
                    sl_mult = 0.6
                    tp1_mult = 0.9
                    tp2_mult = 1.3
            else:  # SWING
                if "BREAKOUT" in setup_type:
                    # Swing breakout: still tighter than pullback swing
                    sl_mult = 0.7
                    tp1_mult = 1.2
                    tp2_mult = 1.8
                elif "PULLBACK" in setup_type:
                    # Classic swing pullback, largest RR
                    sl_mult = 0.9
                    tp1_mult = 1.8
                    tp2_mult = 2.7
                else:
                    # Fallback defaults
                    sl_mult = 0.8
                    tp1_mult = 1.3
                    tp2_mult = 2.0

            entry_price = signal.price
            if signal.direction == "LONG":
                sl = entry_price - sl_mult * atr_h1
                tp1 = entry_price + tp1_mult * atr_h1
                tp2 = entry_price + tp2_mult * atr_h1
            else:
                sl = entry_price + sl_mult * atr_h1
                tp1 = entry_price - tp1_mult * atr_h1
                tp2 = entry_price - tp2_mult * atr_h1

            # ---------- SESSION LABEL ----------
            hhmm = now_utc.hour * 100 + now_utc.minute
            if settings.session_1_start <= hhmm <= settings.session_1_end:
                session_window = "07:00-20:00"
            else:
                session_window = "OUTSIDE"

            # ---------- NEWS ----------
            high_news = has_high_impact_news_near(symbol_label, now_utc)

            # ---------- CONFIDENCE SCORE ----------
            confidence_score, confidence_label = _trading_confidence_score(
                adx_h1_value=adx_h1_value,
                adx_m5_value=adx_m5,
                market_state=market_state,
                regime=market_regime,
                risk_tag=risk_tag,
                high_news=high_news,
            )

            # Trend label for message
            if trend_source == "H1":
                trend_label = f"{trend_for_signal} (H1)"
            else:
                trend_label = f"{trend_for_signal} (M15, H1 ranging)"

            # Attach everything to signal.extra
            signal.extra["atr_h1"] = atr_h1
            signal.extra["sl"] = sl
            signal.extra["tp1"] = tp1
            signal.extra["tp2"] = tp2
            signal.extra["adx_h1"] = adx_h1_value
            signal.extra["market_state"] = market_state
            signal.extra["market_regime"] = market_regime
            signal.extra["trend_strength_label"] = trend_strength_label
            signal.extra["confidence_score"] = confidence_score
            signal.extra["confidence_label"] = confidence_label

            msg = build_signal_message(
                symbol_label,
                signal,
                trend_label,
                session_window,
                high_news,
                market_state,
                market_regime,
                adx_h1_value,
                trend_strength_label,
                confidence_score,
                confidence_label,
            )
            tg.send_message(msg)
            last_signal_time = now_utc

            # ---------- LOG ----------
            row = {
                "symbol": symbol_label,
                "direction": signal.direction,
                "price": signal.price,
                "reason": signal.reason,
                "trend_h1": trend_h1,  # may be None if using M15 fallback
                "session_window": session_window,
                "m5_rsi": signal.extra["m5_rsi"],
                "m5_stoch_k": signal.extra["m5_stoch_k"],
                "m5_stoch_d": signal.extra["m5_stoch_d"],
                "bb_upper": signal.extra["bb_upper"],
                "bb_mid": signal.extra["bb_mid"],
                "bb_lower": signal.extra["bb_lower"],
                "adx_m5": signal.extra["adx_m5"],
                "plus_di_m5": signal.extra["plus_di_m5"],
                "minus_di_m5": signal.extra["minus_di_m5"],
                "high_impact_news": high_news,
                "atr_h1": atr_h1,
                "sl": sl,
                "tp1": tp1,
                "tp2": tp2,
                "adx_h1": adx_h1_value,
                "market_state": market_state,
                "market_regime": market_regime,
                "trend_strength_label": trend_strength_label,
                "confidence_score": confidence_score,
                "confidence_label": confidence_label,
                "setup_type": setup_type,
                "risk_tag": risk_tag,
            }
            log_signal(row)

            logger.info(
                "Signal sent and logged. Trend source: %s, direction: %s, setup: %s, risk_tag: %s",
                trend_source,
                trend_for_signal,
                setup_type,
                risk_tag,
            )
            time.sleep(60)

        except Exception as e:
            logger.exception("Error in main loop: %s", e)
            time.sleep(60)


if __name__ == "__main__":
    main_loop()
# implemented to call Twelve Data
from strategy import (
    detect_trend_h1,
    detect_trend_m15_direction,
    confirm_trend_m15,
    trigger_signal_m5,
    is_within_sessions,
)
from data_logger import log_signal
from high_impact_news import has_high_impact_news_near
from indicators import atr, adx  # ATR + ADX on H1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("xauusd_bot")

# Fetch 5m data at most once every 120 seconds (2 minutes)
FETCH_INTERVAL_SECONDS = 120


def _risk_tag_from_adx(adx_m5: float) -> str:
    """
    Classify the trade idea as SCALP vs SWING based on M5 ADX strength.
    """
    if adx_m5 >= 30:
        return "SWING"
    return "SCALP"


def _trend_strength_label(adx_h1_value: float) -> str:
    """
    Give a human label for H1 trend strength based on ADX.
    """
    if adx_h1_value < 18:
        return "Weak"
    elif adx_h1_value < 25:
        return "Moderate"
    elif adx_h1_value < 35:
        return "Strong"
    else:
        return "Very Strong"


def _market_state_and_regime(
    adx_h1_value: float,
    atr_h1: float,
    last_h1_close: float,
) -> tuple[str, str]:
    """
    Coarse market state (TRENDING / RANGING) + finer regime label
    using ADX(H1) and ATR(H1) as % of price.
    """
    atr_ratio = atr_h1 / last_h1_close if last_h1_close > 0 else 0.0

    # Coarse state
    if adx_h1_value < 18:
        state = "RANGING"
    else:
        state = "TRENDING"

    # Volatility label from ATR ratio
    if atr_ratio < 0.004:  # < 0.4% of price
        vol_label = "Low vol"
    elif atr_ratio < 0.008:  # 0.4–0.8%
        vol_label = "Normal vol"
    else:  # > 0.8%
        vol_label = "High vol"

    # Regime
    if state == "RANGING":
        if vol_label == "Low vol":
            regime = "Quiet Range"
        else:
            regime = "Choppy Range"
    else:  # TRENDING
        if vol_label == "Low vol":
            regime = "Slow Trend"
        elif vol_label == "Normal vol":
            regime = "Steady Trend"
        else:
            regime = "High-Vol Trend"

    regime_full = f"{regime} ({vol_label})"
    return state, regime_full


def _trading_confidence_score(
    adx_h1_value: float,
    adx_m5_value: float,
    market_state: str,
    regime: str,
    risk_tag: str,
    high_news: bool,
) -> tuple[int, str]:
    """
    Build a 0–100 confidence score and label (Low/Medium/High)
    based on trend strength, regime, risk tag, and news.
    """
    score = 50.0

    # Add from H1 trend strength, capped
    score += min(adx_h1_value, 40.0) * 0.8  # max +32

    # Slight boost if M5 ADX is strong
    score += min(adx_m5_value, 40.0) * 0.2  # max +8

    # SWING trades get a small bump
    if risk_tag == "SWING":
        score += 5.0

    # Ranging markets and choppy ranges reduce confidence
    if market_state == "RANGING":
        score -= 10.0
    if "Range" in regime:
        score -= 5.0

    # High-impact news reduces confidence
    if high_news:
        score -= 10.0

    # Clamp 0–100
    score = max(0.0, min(100.0, score))
    score_int = int(round(score))

    if score_int <= 40:
        label = "Low"
    elif score_int <= 70:
        label = "Medium"
    else:
        label = "High"

    return score_int, label


def build_signal_message(
    symbol_label: str,
    signal,
    trend_label: str,
    session_window: str,
    high_news: bool,
    market_state: str,
    market_regime: str,
    adx_h1_value: float,
    trend_strength_label: str,
    confidence_score: int,
    confidence_label: str,
) -> str:
    """
    Build Telegram message with:

    - Header, direction, SL, TP1, TP2
    - RR to TP1/TP2
    - Time, trend bias, session, reason
    - Market state, regime, trend strength, confidence
    - ATR info
    - News + key indicators
    """
    adx_m5 = signal.extra["adx_m5"]
    risk_tag = _risk_tag_from_adx(adx_m5)

    atr_h1 = signal.extra.get("atr_h1")
    sl = signal.extra.get("sl")
    tp1 = signal.extra.get("tp1")
    tp2 = signal.extra.get("tp2")

    arrow = "🟢 BUY" if signal.direction == "LONG" else "🔴 SELL"

    if high_news:
        news_line = "⚠️ HIGH-IMPACT NEWS NEARBY — expect extra volatility."
    else:
        news_line = "ℹ️ No high-impact news flag near this time."

    entry = signal.price
    setup_type = signal.extra.get("setup_type", "GENERIC")

    lines: list[str] = []

    # ----- Header + entry + TP/SL + RR -----
    lines.append(f"XAUUSD Signal [{risk_tag}]")
    lines.append(f"{arrow} {symbol_label} at {entry:.2f}")
    if setup_type != "GENERIC":
        lines.append(f"Setup: {setup_type}")

    if sl is not None and tp1 is not None and tp2 is not None:
        lines.append(f"– SL: {sl:.2f}")
        lines.append(f"– TP1: {tp1:.2f}")
        lines.append(f"– TP2: {tp2:.2f}")

        # Risk / Reward computation
        if signal.direction == "LONG":
            risk = entry - sl
            rr1 = (tp1 - entry) / risk if risk > 0 else 0.0
            rr2 = (tp2 - entry) / risk if risk > 0 else 0.0
        else:
            risk = sl - entry
            rr1 = (entry - tp1) / risk if risk > 0 else 0.0
            rr2 = (entry - tp2) / risk if risk > 0 else 0.0

        if risk > 0:
            lines.append(
                f"RR to TP1: {rr1:.2f}R | RR to TP2: {rr2:.2f}R"
            )

    lines.append("")  # blank

    # ----- Context -----
    lines.append(f"Time (UTC): {signal.time.isoformat()}")
    lines.append(f"Trend Bias: {trend_label}")
    lines.append(f"Session: {session_window}")
    lines.append(f"Reason: {signal.reason}")
    lines.append("")  # blank

    # ----- Market state / regime / trend strength / confidence -----
    lines.append(
        f"Market State (H1): {market_state} (ADX {adx_h1_value:.2f})"
    )
    lines.append(f"Market Regime: {market_regime}")
    lines.append(
        f"Trend Strength (H1): {trend_strength_label}"
    )
    lines.append(
        f"Trading Confidence: {confidence_score} ({confidence_label})"
    )
    lines.append("")  # blank

    # ----- ATR info -----
    lines.append("Suggested TP/SL (ATR-based)")
    if atr_h1 is not None:
        lines.append(f"– ATR(H1, 14): {atr_h1:.2f}")
    lines.append("")  # blank

    # ----- News & indicators -----
    lines.append(news_line)
    lines.append(
        f"RSI(M5): {signal.extra['m5_rsi']:.2f} | "
        f"StochK(M5): {signal.extra['m5_stoch_k']:.2f}"
    )
    lines.append(
        f"ADX(M5): {signal.extra['adx_m5']:.2f} "
        f"(+DI: {signal.extra['plus_di_m5']:.2f}, "
        f"-DI: {signal.extra['minus_di_m5']:.2f})"
    )
    lines.append(
        "BB(M5): upper {0:.2f}, mid {1:.2f}, lower {2:.2f}".format(
            signal.extra["bb_upper"],
            signal.extra["bb_mid"],
            signal.extra["bb_lower"],
        )
    )

    return "\n".join(lines)


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Resample a 5m OHLCV DataFrame to a higher timeframe (15m, 1h, etc.).
    Assumes df has columns: datetime, open, high, low, close, volume.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    tmp = df.copy()
    tmp = tmp.set_index("datetime")
    agg = tmp.resample(rule).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    agg = agg.dropna()
    agg = agg.reset_index()
    return agg


def main_loop():
    settings: Settings = load_settings()
    tg = TelegramClient(settings.telegram_bot_token, settings.telegram_chat_id)

    symbol_label = "XAUUSD"
    logger.info("Starting XAUUSD bot (Twelve Data only, H1 primary, M15 fallback).")

    last_signal_time: Optional[dt.datetime] = None

    cached_m5_df: Optional[pd.DataFrame] = None
    last_m5_fetch_ts: Optional[float] = None

    while True:
        now_utc = dt.datetime.utcnow()

        if not is_within_sessions(
            now_utc,
            settings.session_1_start,
            settings.session_1_end,
            settings.session_2_start,
            settings.session_2_end,
        ):
            logger.info("Outside trading sessions, sleeping 60s...")
            time.sleep(60)
            continue

        try:
            # ---------- DATA FETCH / CACHE ----------
            now_ts = time.time()
            should_fetch_m5 = (
                last_m5_fetch_ts is None
                or (now_ts - last_m5_fetch_ts) >= FETCH_INTERVAL_SECONDS
            )

            if should_fetch_m5:
                logger.info("Fetching fresh M5 OHLCV data from Twelve Data...")
                m5_df = fetch_m5_ohlcv_hybrid(settings)
                cached_m5_df = m5_df
                last_m5_fetch_ts = now_ts
            else:
                if cached_m5_df is None:
                    logger.info("No cached M5 data yet, fetching from Twelve Data...")
                    m5_df = fetch_m5_ohlcv_hybrid(settings)
                    cached_m5_df = m5_df
                    last_m5_fetch_ts = now_ts
                else:
                    logger.info("Using cached M5 OHLCV data.")
                    m5_df = cached_m5_df

            if m5_df is None or m5_df.empty:
                logger.info("Empty M5 data, sleeping 60s.")
                time.sleep(60)
                continue

            # ---------- RESAMPLING ----------
            h1_df = resample_ohlcv(m5_df, "1h")
            m15_df = resample_ohlcv(m5_df, "15min")

            if len(h1_df) < 20 or len(m15_df) < 20 or len(m5_df) < 50:
                logger.info("Not enough data after resampling, sleeping 60s.")
                time.sleep(60)
                continue

            # ---------- TREND SELECTION: H1 PRIMARY, M15 FALLBACK ----------
            trend_h1 = detect_trend_h1(h1_df)
            trend_source = "H1"
            trend_for_signal: Optional[str] = trend_h1

            if trend_for_signal is not None:
                # When H1 has a direction, we still require M15 confirmation
                if not confirm_trend_m15(m15_df, trend_h1):
                    logger.info("M15 does not confirm H1 trend, skipping.")
                    time.sleep(60)
                    continue
            else:
                # H1 is ranging / unclear -> try M15 as fallback
                trend_m15_bias = detect_trend_m15_direction(m15_df)
                if trend_m15_bias is None:
                    logger.info("No clear H1 or M15 trend, skipping.")
                    time.sleep(60)
                    continue
                trend_source = "M15"
                trend_for_signal = trend_m15_bias

            # Safety check
            if trend_for_signal is None:
                logger.info("No usable trend_for_signal, skipping.")
                time.sleep(60)
                continue

            # ---------- M5 TRIGGER ----------
            signal = trigger_signal_m5(m5_df, trend_for_signal)
            if not signal:
                logger.info("No M5 trigger signal, sleeping 60s.")
                time.sleep(60)
                continue

            # Cooldown
            if last_signal_time and (now_utc - last_signal_time).total_seconds() < 300:
                logger.info(
                    "Signal occurred too soon after previous, skipping (cooldown)."
                )
                time.sleep(60)
                continue

            # ---------- H1 VOL / ADX / STATE / REGIME ----------
            atr_series = atr(h1_df["high"], h1_df["low"], h1_df["close"], period=14)
            atr_h1 = float(atr_series.iloc[-1])

            adx_h1_series, _, _ = adx(
                h1_df["high"], h1_df["low"], h1_df["close"], period=14
            )
            adx_h1_value = float(adx_h1_series.iloc[-1])

            last_h1_close = float(h1_df["close"].iloc[-1])

            market_state, market_regime = _market_state_and_regime(
                adx_h1_value, atr_h1, last_h1_close
            )
            trend_strength_label = _trend_strength_label(adx_h1_value)

            # Risk tag from M5 ADX
            adx_m5 = signal.extra["adx_m5"]
            risk_tag = _risk_tag_from_adx(adx_m5)

            # ---------- DYNAMIC TP/SL (ATR + SCALP/SWING) ----------
            if risk_tag == "SCALP":
                sl_mult = 0.6
                tp1_mult = 0.9
                tp2_mult = 1.3
            else:  # SWING
                sl_mult = 0.8
                tp1_mult = 1.3
                tp2_mult = 2.0

            entry_price = signal.price
            if signal.direction == "LONG":
                sl = entry_price - sl_mult * atr_h1
                tp1 = entry_price + tp1_mult * atr_h1
                tp2 = entry_price + tp2_mult * atr_h1
            else:
                sl = entry_price + sl_mult * atr_h1
                tp1 = entry_price - tp1_mult * atr_h1
                tp2 = entry_price - tp2_mult * atr_h1

            # ---------- SESSION LABEL ----------
            hhmm = now_utc.hour * 100 + now_utc.minute
            if settings.session_1_start <= hhmm <= settings.session_1_end:
                session_window = "07:00-20:00"
            else:
                session_window = "OUTSIDE"

            # ---------- NEWS ----------
            high_news = has_high_impact_news_near(symbol_label, now_utc)

            # ---------- CONFIDENCE SCORE ----------
            confidence_score, confidence_label = _trading_confidence_score(
                adx_h1_value=adx_h1_value,
                adx_m5_value=adx_m5,
                market_state=market_state,
                regime=market_regime,
                risk_tag=risk_tag,
                high_news=high_news,
            )

            # Trend label for message
            if trend_source == "H1":
                trend_label = f"{trend_for_signal} (H1)"
            else:
                trend_label = f"{trend_for_signal} (M15, H1 ranging)"

            # Attach everything to signal.extra
            signal.extra["atr_h1"] = atr_h1
            signal.extra["sl"] = sl
            signal.extra["tp1"] = tp1
            signal.extra["tp2"] = tp2
            signal.extra["adx_h1"] = adx_h1_value
            signal.extra["market_state"] = market_state
            signal.extra["market_regime"] = market_regime
            signal.extra["trend_strength_label"] = trend_strength_label
            signal.extra["confidence_score"] = confidence_score
            signal.extra["confidence_label"] = confidence_label

            msg = build_signal_message(
                symbol_label,
                signal,
                trend_label,
                session_window,
                high_news,
                market_state,
                market_regime,
                adx_h1_value,
                trend_strength_label,
                confidence_score,
                confidence_label,
            )
            tg.send_message(msg)
            last_signal_time = now_utc

            # ---------- LOG ----------
            row = {
                "symbol": symbol_label,
                "direction": signal.direction,
                "price": signal.price,
                "reason": signal.reason,
                "trend_h1": trend_h1,  # may be None if using M15 fallback
                "session_window": session_window,
                "m5_rsi": signal.extra["m5_rsi"],
                "m5_stoch_k": signal.extra["m5_stoch_k"],
                "m5_stoch_d": signal.extra["m5_stoch_d"],
                "bb_upper": signal.extra["bb_upper"],
                "bb_mid": signal.extra["bb_mid"],
                "bb_lower": signal.extra["bb_lower"],
                "adx_m5": signal.extra["adx_m5"],
                "plus_di_m5": signal.extra["plus_di_m5"],
                "minus_di_m5": signal.extra["minus_di_m5"],
                "high_impact_news": high_news,
                "atr_h1": atr_h1,
                "sl": sl,
                "tp1": tp1,
                "tp2": tp2,
                "adx_h1": adx_h1_value,
                "market_state": market_state,
                "market_regime": market_regime,
                "trend_strength_label": trend_strength_label,
                "confidence_score": confidence_score,
                "confidence_label": confidence_label,
            }
            log_signal(row)

            logger.info(
                "Signal sent and logged. Trend source: %s, direction: %s",
                trend_source,
                trend_for_signal,
            )
            time.sleep(60)

        except Exception as e:
            logger.exception("Error in main loop: %s", e)
            time.sleep(60)


if __name__ == "__main__":
    main_loop()
