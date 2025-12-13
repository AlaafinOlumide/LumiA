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
# Logging
# -------------------------
logger = logging.getLogger("xauusd_bot")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# -------------------------
# Optional news module (safe import)
# -------------------------
def _safe_has_high_impact_news(now_utc: dt.datetime) -> bool:
    """
    Safe wrapper: if module/function missing or broken, we continue without news filter.
    """
    try:
        import high_impact_news  # noqa
        fn = None
        if hasattr(high_impact_news, "has_high_impact_news_nearby"):
            fn = getattr(high_impact_news, "has_high_impact_news_nearby")
        elif hasattr(high_impact_news, "has_high_impact_news_near"):
            fn = getattr(high_impact_news, "has_high_impact_news_near")

        if fn is None:
            return False

        return bool(fn(now_utc))
    except Exception:
        return False


# -------------------------
# Resampling (M5 -> M15/H1)
# -------------------------
def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    rule: "15min" or "1h"
    Expects df with a datetime column (UTC) or datetime index.
    """
    tmp = df.copy()
    if "datetime" in tmp.columns:
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
    """
    Simple regime using Bollinger width + ATR magnitude (H1).
    """
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
    if h1_df is None or len(h1_df) < 60:
        return

    high = h1_df["high"]
    low = h1_df["low"]
    close = h1_df["close"]

    atr_series = indicators.atr(high, low, close, period=14)
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
# Trend Exhaustion + M15 Veto Filters
# -------------------------
def h1_exhaustion_veto(h1_df: pd.DataFrame, trend_dir: str, setup_type: str) -> Tuple[bool, str]:
    """
    Blocks 'late trend' pullbacks that are prone to sharp reversal.
    """
    if h1_df is None or len(h1_df) < 60:
        return False, ""

    close = h1_df["close"]
    rsi_h1 = indicators.rsi(close, period=14)
    bb_u, bb_m, bb_l = indicators.bollinger_bands(close, period=20, std_factor=2.0)

    c = float(close.iloc[-1])
    r = float(rsi_h1.iloc[-1]) if pd.notna(rsi_h1.iloc[-1]) else 50.0
    u = float(bb_u.iloc[-1])
    l = float(bb_l.iloc[-1])

    # We mainly block PULLBACK entries when trend is stretched
    if setup_type.startswith("PULLBACK"):
        if trend_dir == "LONG":
            stretched = (r >= 70) or (c >= u * 0.995)
            if stretched:
                return True, "H1 exhaustion veto: LONG stretched (RSI>=70 or near/above upper BB)."
        if trend_dir == "SHORT":
            stretched = (r <= 30) or (c <= l * 1.005)
            if stretched:
                return True, "H1 exhaustion veto: SHORT stretched (RSI<=30 or near/below lower BB)."

    return False, ""


def m15_veto(m15_df: pd.DataFrame, trend_dir: str, setup_type: str) -> Tuple[bool, str]:
    """
    Stops M5 setups when M15 structure is not supporting the direction.
    Strongest veto for Pullbacks + Continuations (most sensitive to micro flips).
    """
    if m15_df is None or len(m15_df) < 60:
        return False, ""

    close = m15_df["close"]
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    rsi15 = indicators.rsi(close, period=14)

    c = float(close.iloc[-1])
    e20 = float(ema20.iloc[-1])
    e50 = float(ema50.iloc[-1])
    r = float(rsi15.iloc[-1]) if pd.notna(rsi15.iloc[-1]) else 50.0

    if trend_dir == "LONG":
        ok = (e20 > e50) and (c >= e20) and (r >= 50)
        if not ok and (setup_type.startswith("PULLBACK") or setup_type.startswith("BREAKOUT_CONT")):
            return True, "M15 veto: LONG not supported (EMA/RSI structure weak)."

    if trend_dir == "SHORT":
        ok = (e20 < e50) and (c <= e20) and (r <= 50)
        if not ok and (setup_type.startswith("PULLBACK") or setup_type.startswith("BREAKOUT_CONT")):
            return True, "M15 veto: SHORT not supported (EMA/RSI structure weak)."

    return False, ""


# -------------------------
# Confidence Score Engine (Upgraded)
# -------------------------
def compute_confidence(
    trend_source: str,
    setup_type: str,
    adx_h1: float,
    adx_m5: float,
    high_news: bool,
    market_regime_text: str,
    m15_confirmed: bool,
    h1_exhausted: bool,
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

    # Trend strength ADX(H1)
    if adx_h1 >= 30:
        score += 12
    elif adx_h1 >= 20:
        score += 7
    else:
        score -= 6

    # Micro ADX (M5)
    if adx_m5 >= 25:
        score += 6
    elif adx_m5 < 15:
        score -= 5

    # M15 confirmation bonus
    score += 6 if m15_confirmed else -5

    # Regime adjustments
    if market_regime_text == "Low Volatility / Compression":
        # Breakouts are more meaningful during compression
        if setup_type.startswith("BREAKOUT"):
            score += 6
    if market_regime_text == "High Volatility":
        # Pullbacks are more dangerous in high vol (whipsaws)
        if setup_type.startswith("PULLBACK"):
            score -= 5

    # Exhaustion penalty
    if h1_exhausted:
        score -= 12

    # News penalty
    if high_news:
        score -= 12

    return max(0, min(100, score))


# -------------------------
# Cooldown Protocol
# -------------------------
def cooldown_seconds_for_setup(setup_type: str) -> int:
    """
    Time-based cooldown to avoid rapid re-entries.
    """
    if setup_type.startswith("PULLBACK"):
        return 12 * 60  # 12 min
    if setup_type.startswith("BREAKOUT"):
        return 9 * 60   # 9 min
    if setup_type.startswith("BREAKOUT_CONT"):
        return 6 * 60   # 6 min
    return 10 * 60


# -------------------------
# Telegram message (PLAIN TEXT) — EXACT TEMPLATE YOU GAVE
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

        if risk > 0:
            rr_line = f"RR to TP1: {rr1:.2f}R | RR to TP2: {rr2:.2f}R"

    lines = []
    lines.append(f"XAUUSD Signal [{risk_tag}]")
    lines.append(f"{arrow} {symbol_label} at {entry:.2f}")
    lines.append("")

    # SL/TP block (as per your template)
    if sl is not None:
        lines.append(f"SL: {float(sl):.2f}")
    if tp1 is not None:
        lines.append(f"TP1: {float(tp1):.2f}")
    if tp2 is not None:
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
    lines.append(
        f"ADX(M5): {float(signal.extra.get('adx_m5', 0.0)):.2f} "
        f"(+DI: {float(signal.extra.get('plus_di_m5', 0.0)):.2f}, -DI: {float(signal.extra.get('minus_di_m5', 0.0)):.2f})"
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
# Main loop (NEW candle detection + cooldown)
# -------------------------
def main():
    settings = load_settings()

    symbol_td = settings.xau_symbol_td  # "XAU/USD"
    symbol_label = "XAUUSD"

    s1_start = settings.session_1_start  # 700
    s1_end = settings.session_1_end      # 2000

    # Polling + API calls:
    sleep_seconds = 30                 # check more frequently
    fetch_interval_seconds = 120       # fetch every 2 minutes (as you asked)

    telegram = TelegramClient(settings.telegram_bot_token, settings.telegram_chat_id)

    logger.info("Starting XAUUSD bot (H1 primary, M15 fallback) + NEW M5 candle detection + cooldown.")

    last_fetch_ts: float = 0.0
    cached_m5: Optional[pd.DataFrame] = None

    # New candle detection state
    last_seen_m5_candle_time: Optional[pd.Timestamp] = None

    # Cooldown state
    last_signal_sent_ts: Optional[float] = None
    last_signal_key: Optional[str] = None

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

        # Fetch/cached M5
        need_fetch = (time.time() - last_fetch_ts) >= fetch_interval_seconds or cached_m5 is None
        if need_fetch:
            logger.info("Fetching fresh M5 OHLCV data from Twelve Data...")
            cached_m5 = fetch_m5_ohlcv_twelvedata(
                symbol=symbol_td,
                api_key=settings.twelvedata_api_key,
            )
            last_fetch_ts = time.time()
        else:
            logger.info("Using cached M5 OHLCV data.")

        if cached_m5 is None or len(cached_m5) < 300:
            logger.warning("Not enough M5 data yet. Sleeping %ss...", sleep_seconds)
            time.sleep(sleep_seconds)
            continue

        # New candle detection: only evaluate on new M5 close
        current_candle_time = pd.to_datetime(cached_m5["datetime"].iloc[-1], utc=True, errors="coerce")
        if pd.isna(current_candle_time):
            logger.info("Bad candle timestamp received; sleeping.")
            time.sleep(sleep_seconds)
            continue

        if last_seen_m5_candle_time is not None and current_candle_time <= last_seen_m5_candle_time:
            logger.info(
                "No new M5 candle closed yet (last=%s, current=%s). Sleeping %ss...",
                last_seen_m5_candle_time,
                current_candle_time,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)
            continue

        last_seen_m5_candle_time = current_candle_time
        logger.info("New M5 candle detected: %s — evaluating signal...", current_candle_time)

        # Resample
        m15_df = resample_ohlc(cached_m5, "15min")
        h1_df = resample_ohlc(cached_m5, "1h")

        if h1_df is None or len(h1_df) < 60 or m15_df is None or len(m15_df) < 60:
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
            m15_dir = detect_trend_m15_direction(m15_df)
            if m15_dir is None:
                logger.info("No clear H1 trend and no clear M15 direction, skipping.")
                time.sleep(sleep_seconds)
                continue
            trend_dir = m15_dir
            trend_source = "M15"
        else:
            trend_dir = h1_trend

        trend_label = "LONG" if trend_dir == "LONG" else "SHORT"

        # Market classifiers
        market_state = market_state_from_adx(adx_h1)
        trend_strength = trend_strength_from_adx(adx_h1)
        regime = market_regime(h1_df)

        # News flag (safe)
        high_news = _safe_has_high_impact_news(now_utc)

        # Trigger
        signal = trigger_signal_m5(cached_m5, trend_dir)
        if not signal:
            logger.info("No M5 trigger signal on candle close, sleeping %ss.", sleep_seconds)
            time.sleep(sleep_seconds)
            continue

        # Dynamic TP/SL
        apply_dynamic_tp_sl(signal, h1_df)

        setup_type = signal.extra.get("setup_type", "GENERIC")
        adx_m5 = float(signal.extra.get("adx_m5", 0.0))

        # --- Filters (Exhaustion + M15 veto) ---
        exhausted, exhausted_reason = h1_exhaustion_veto(h1_df, trend_dir, setup_type)
        vetoed, veto_reason = m15_veto(m15_df, trend_dir, setup_type)

        if exhausted:
            logger.info("Signal blocked: %s", exhausted_reason)
            time.sleep(sleep_seconds)
            continue

        if vetoed:
            logger.info("Signal blocked: %s", veto_reason)
            time.sleep(sleep_seconds)
            continue

        # M15 confirmation (bonus/penalty in confidence)
        m15_confirmed = confirm_trend_m15(m15_df, trend_dir)

        # --- Cooldown protocol ---
        cooldown_sec = cooldown_seconds_for_setup(setup_type)
        if last_signal_sent_ts is not None:
            elapsed = time.time() - last_signal_sent_ts
            if elapsed < cooldown_sec:
                logger.info(
                    "Cooldown active (%ss remaining). Skipping signal.",
                    int(cooldown_sec - elapsed),
                )
                time.sleep(sleep_seconds)
                continue

        # Dedup (avoid duplicate sends for same candle/setup/dir)
        signal_key = f"{setup_type}|{signal.direction}|{signal.time.isoformat()}"
        if last_signal_key == signal_key:
            logger.info("Duplicate signal key detected; skipping.")
            time.sleep(sleep_seconds)
            continue

        # Confidence (Upgraded engine)
        conf = compute_confidence(
            trend_source=trend_source,
            setup_type=setup_type,
            adx_h1=adx_h1,
            adx_m5=adx_m5,
            high_news=high_news,
            market_regime_text=regime,
            m15_confirmed=m15_confirmed,
            h1_exhausted=exhausted,
        )
        conf_text = confidence_label(conf)

        # Message (YOUR template)
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

        last_signal_sent_ts = time.time()
        last_signal_key = signal_key

        logger.info(
            "Signal sent. Trend source=%s direction=%s setup=%s confidence=%s (%s)",
            trend_source,
            trend_label,
            setup_type,
            conf,
            conf_text,
        )

        time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()
