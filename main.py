# main.py
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

from high_impact_news import has_high_impact_news_near

from trade_manager import (
    ActiveTrade,
    TradeJournal,
    new_trade_id,
    should_invalidate,
    check_tp_sl_hit,
    compute_r_result,
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
# Market open (fix weekend signals)
# -------------------------
def is_market_open(now_utc: dt.datetime) -> bool:
    """
    FX-style market hours:
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
    if h1_df is None or len(h1_df) < 60:
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
# Adaptive Confidence Score
# -------------------------
def compute_confidence_adaptive(
    trend_source: str,
    setup_type: str,
    adx_h1: float,
    adx_m5: float,
    high_news: bool,
    m5_rsi: float,
    plus_di_m5: float,
    minus_di_m5: float,
) -> int:
    """
    Adaptive: adds penalties for fragile contexts and DI disagreement.
    """
    score = 50

    # Trend source reliability
    score += 10 if trend_source == "H1" else 5

    # Setup quality
    if setup_type.startswith("PULLBACK"):
        score += 15
    elif setup_type.startswith("BREAKOUT") and not setup_type.startswith("BREAKOUT_CONT"):
        score += 8
    elif setup_type.startswith("BREAKOUT_CONT"):
        score += 3

    # H1 trend strength
    if adx_h1 >= 30:
        score += 12
    elif adx_h1 >= 20:
        score += 7
    elif adx_h1 < 18:
        score -= 10

    # M5 structure
    if adx_m5 >= 25:
        score += 6
    elif adx_m5 < 15:
        score -= 5

    # DI disagreement penalty
    # (if ADX high but DI suggests opposite pressure, confidence should drop)
    if adx_m5 >= 30:
        if plus_di_m5 < minus_di_m5 and "LONG" in setup_type:
            score -= 10
        if minus_di_m5 < plus_di_m5 and "SHORT" in setup_type:
            score -= 10

    # RSI context penalty
    if setup_type.endswith("_LONG") and m5_rsi < 50:
        score -= 6
    if setup_type.endswith("_SHORT") and m5_rsi > 50:
        score -= 6

    # News penalty
    if high_news:
        score -= 12

    return max(0, min(100, score))


# -------------------------
# Telegram message template (exact format you requested)
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

    # RR
    rr_line = ""
    if sl is not None and tp1 is not None and tp2 is not None:
        if signal.direction == "LONG":
            risk = entry - float(sl)
            rr1 = (float(tp1) - entry) / risk if risk > 0 else 0.0
            rr2 = (float(tp2) - entry) / risk if risk > 0 else 0.0
        else:
            risk = float(sl) - entry
            rr1 = (entry - float(tp1)) / risk if risk > 0 else 0.0
            rr2 = (entry - float(tp2)) / risk if risk > 0 else 0.0
        rr_line = f"RR to TP1: {rr1:.2f}R | RR to TP2: {rr2:.2f}R"

    lines = []
    lines.append(f"XAUUSD Signal [{risk_tag}]")
    lines.append(f"{arrow} {symbol_label} at {entry:.2f}")
    lines.append("")
    if sl is not None and tp1 is not None and tp2 is not None:
        lines.append(f"SL: {float(sl):.2f}")
        lines.append(f"TP1: {float(tp1):.2f}")
        lines.append(f"TP2: {float(tp2):.2f}")
    lines.append("")
    lines.append(f"Setup: {setup_text} ({setup_type})")
    lines.append(f"Confidence: {confidence_score} ({confidence_text})")
    lines.append("")
    if rr_line:
        lines.append(rr_line)
    lines.append("")
    lines.append(f"Time (UTC): {signal.time.isoformat()}")
    lines.append(f"Trend Bias: {trend_label} (source: {trend_source})")
    lines.append(f"Session: {session_window}")
    lines.append(f"Reason: {signal.reason}")
    lines.append("")
    lines.append(f"Market State (H1): {market_state} (ADX {adx_h1:.2f})")
    lines.append(f"Trend Strength (H1): {trend_strength}")
    lines.append(f"Market Regime: {market_regime_text}")
    if atr_h1 is not None:
        lines.append(f"ATR(H1,14): {float(atr_h1):.2f}")
    lines.append("")
    lines.append("No high-impact news flag near this time." if not high_news else "HIGH-IMPACT NEWS NEARBY: expect extra volatility.")
    lines.append(f"RSI(M5): {float(signal.extra.get('m5_rsi', 0.0)):.2f} | StochK(M5): {float(signal.extra.get('m5_stoch_k', 0.0)):.2f}")
    lines.append(f"ADX(M5): {float(signal.extra.get('adx_m5', 0.0)):.2f} (+DI: {float(signal.extra.get('plus_di_m5', 0.0)):.2f}, -DI: {float(signal.extra.get('minus_di_m5', 0.0)):.2f})")
    lines.append("BB(M5): upper {0:.2f}, mid {1:.2f}, lower {2:.2f}".format(
        float(signal.extra.get("bb_upper", 0.0)),
        float(signal.extra.get("bb_mid", 0.0)),
        float(signal.extra.get("bb_lower", 0.0)),
    ))

    return "\n".join(lines)


# -------------------------
# Cooldown protocol
# -------------------------
def in_cooldown(now_utc: dt.datetime, last_signal_time: Optional[dt.datetime], minutes: int = 20) -> bool:
    if last_signal_time is None:
        return False
    return (now_utc - last_signal_time).total_seconds() < (minutes * 60)


# -------------------------
# Main loop
# -------------------------
def main():
    settings = load_settings()

    symbol_td = settings.xau_symbol_td
    symbol_label = "XAUUSD"

    s1_start = settings.session_1_start
    s1_end = settings.session_1_end

    sleep_seconds = 60
    fetch_interval_seconds = 180  # keep API calls sane

    telegram = TelegramClient(settings.telegram_bot_token, settings.telegram_chat_id)
    journal = TradeJournal("trades.csv")

    logger.info("Starting XAUUSD bot (H1 primary, M15 fallback) + NEW M5 candle detection + journaling + invalidation.")

    last_fetch_ts: float = 0.0
    cached_m5: Optional[pd.DataFrame] = None

    last_closed_candle_time: Optional[pd.Timestamp] = None
    last_signal_time: Optional[dt.datetime] = None
    active_trade: Optional[ActiveTrade] = None

    while True:
        now_utc = dt.datetime.now(dt.timezone.utc)

        # Market open filter (fix weekend signals)
        if not is_market_open(now_utc):
            logger.info("Market closed (weekend window). Sleeping %ss...", sleep_seconds)
            time.sleep(sleep_seconds)
            continue

        # Session window
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

        # Fetch / cache
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

        # Only evaluate on NEW closed candle
        if last_closed_candle_time is not None and current_last_time <= last_closed_candle_time:
            logger.info("No new M5 candle closed yet (last=%s, current=%s). Sleeping %ss...",
                        last_closed_candle_time, current_last_time, sleep_seconds)
            time.sleep(sleep_seconds)
            continue

        last_closed_candle_time = current_last_time
        logger.info("New M5 candle detected: %s — evaluating...", current_last_time)

        # Build higher TF
        m15_df = resample_ohlc(cached_m5, "15min")
        h1_df = resample_ohlc(cached_m5, "1h")
        if h1_df is None or len(h1_df) < 60 or m15_df is None or len(m15_df) < 60:
            logger.info("Not enough data after resampling, sleeping %ss...", sleep_seconds)
            time.sleep(sleep_seconds)
            continue

        # Update active trade outcomes (journaling)
        last_candle = cached_m5.iloc[-1]

        if active_trade is not None and active_trade.status == "OPEN":
            hit = check_tp_sl_hit(active_trade, last_candle)
            if hit:
                active_trade.status = hit["status"]
                active_trade.exit_time = last_closed_candle_time.to_pydatetime()
                active_trade.exit_price = float(hit["exit_price"])
                active_trade.result_r = compute_r_result(active_trade.direction, active_trade.entry, active_trade.sl, active_trade.exit_price)
                journal.update_trade(active_trade)

                telegram.send_message(
                    f"Trade Update [{active_trade.status}]\n"
                    f"{active_trade.direction} {symbol_label}\n"
                    f"Exit: {active_trade.exit_price:.2f}\n"
                    f"R: {active_trade.result_r:.2f}\n"
                    f"TradeID: {active_trade.trade_id}"
                )
                logger.info("Trade closed: %s status=%s R=%.2f", active_trade.trade_id, active_trade.status, active_trade.result_r)

                # Cooldown starts after closure too
                last_signal_time = active_trade.exit_time
                active_trade = None

        # Invalidation (only if still open)
        if active_trade is not None and active_trade.status == "OPEN":
            # compute BB + RSI quickly for invalidation rules
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

            # Only allow invalidation within deadline
            if active_trade.invalidation_deadline and now_utc <= active_trade.invalidation_deadline and reason:
                active_trade.status = "INVALIDATED"
                active_trade.invalidated_reason = reason
                active_trade.exit_time = last_closed_candle_time.to_pydatetime()
                active_trade.exit_price = float(last_candle["close"])
                active_trade.result_r = compute_r_result(active_trade.direction, active_trade.entry, active_trade.sl, active_trade.exit_price)
                journal.update_trade(active_trade)

                telegram.send_message(
                    f"Signal Invalidated ❌\n"
                    f"TradeID: {active_trade.trade_id}\n"
                    f"Reason: {reason}\n"
                    f"Close: {active_trade.exit_price:.2f}\n"
                    f"R (if entered): {active_trade.result_r:.2f}"
                )
                logger.info("Trade invalidated: %s reason=%s", active_trade.trade_id, reason)

                last_signal_time = active_trade.exit_time
                active_trade = None

        # If a trade is still open, we do NOT open new signals (keeps system clean)
        if active_trade is not None:
            logger.info("Active trade open, skipping new entries.")
            time.sleep(sleep_seconds)
            continue

        # Cooldown protocol
        if in_cooldown(now_utc, last_signal_time, minutes=20):
            logger.info("Cooldown active, skipping.")
            time.sleep(sleep_seconds)
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
                time.sleep(sleep_seconds)
                continue
            trend_dir = m15_dir
            trend_source = "M15"
        else:
            trend_dir = h1_trend
            _ = confirm_trend_m15(m15_df, trend_dir)

        trend_label = "LONG" if trend_dir == "LONG" else "SHORT"

        # News flag
        try:
            high_news = has_high_impact_news_near(symbol_label, now_utc, window_minutes=60)
        except Exception:
            high_news = False

        # Trigger on M5 candle close
        signal = trigger_signal_m5(cached_m5, trend_dir)
        if not signal:
            logger.info("No M5 trigger signal on candle close.")
            time.sleep(sleep_seconds)
            continue

        apply_dynamic_tp_sl(signal, h1_df)

        setup_type = signal.extra.get("setup_type", "GENERIC")
        adx_m5 = float(signal.extra.get("adx_m5", 0.0))
        m5_rsi = float(signal.extra.get("m5_rsi", 50.0))
        plus_di_m5 = float(signal.extra.get("plus_di_m5", 0.0))
        minus_di_m5 = float(signal.extra.get("minus_di_m5", 0.0))

        conf = compute_confidence_adaptive(
            trend_source=trend_source,
            setup_type=setup_type,
            adx_h1=adx_h1,
            adx_m5=adx_m5,
            high_news=high_news,
            m5_rsi=m5_rsi,
            plus_di_m5=plus_di_m5,
            minus_di_m5=minus_di_m5,
        )
        conf_text = confidence_label(conf)

        market_state = market_state_from_adx(adx_h1)
        trend_strength = trend_strength_from_adx(adx_h1)
        regime = market_regime(h1_df)

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
        logger.info("Signal sent: dir=%s setup=%s conf=%s", trend_label, setup_type, conf)

        # Create active trade + journal
        sl = float(signal.extra["sl"])
        tp1 = float(signal.extra["tp1"])
        tp2 = float(signal.extra["tp2"])

        trade = ActiveTrade(
            trade_id=new_trade_id(),
            opened_time=signal.time if signal.time.tzinfo else signal.time.replace(tzinfo=dt.timezone.utc),
            direction=signal.direction,
            setup_type=setup_type,
            entry=float(signal.price),
            sl=sl,
            tp1=tp1,
            tp2=tp2,
            confidence=conf,
            trend_source=trend_source,
            invalidation_deadline=(now_utc + dt.timedelta(minutes=10)),  # 2 candles on M5-ish
        )
        journal.append_open(trade)

        active_trade = trade
        last_signal_time = now_utc

        time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()
