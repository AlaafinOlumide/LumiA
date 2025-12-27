import time
import logging
import pandas as pd

from config import Config
from data_fetcher import fetch_ohlcv
from strategy import generate_signal
from telegram_client import send_message
from storage import load_state, save_state
from journal import open_journal_entry, update_journal_status, append_csv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
log = logging.getLogger("xauusd_bot")

def fmt_signal(sig) -> str:
    return (
        f"<b>{sig.symbol} Signal [SCALP]</b>\n"
        f"<b>{sig.direction}</b> {sig.symbol} at <b>{sig.entry:.2f}</b>\n\n"
        f"SL: {sig.sl:.2f}\n"
        f"TP1: {sig.tp1:.2f}\n"
        f"TP2: {sig.tp2:.2f}\n\n"
        f"Setup: {sig.setup}\n"
        f"Entry Score: {sig.score}\n"
        f"Confidence: {sig.confidence}\n\n"
        f"Time (UTC): {sig.candle_time.to_pydatetime().isoformat()}\n"
        f"Trend Bias: {sig.trend_bias}\n"
        f"{sig.notes}"
    )

def _dedupe_key(sig) -> str:
    # prevents “2 in 1” (same candle re-sent)
    return f"{sig.symbol}|{sig.tf}|{sig.candle_time.isoformat()}|{sig.setup}|{sig.direction}"

def run_loop() -> None:
    cfg = Config()
    state = load_state(cfg.STATE_PATH)

    last_seen_candle = pd.to_datetime(state.get("last_seen_candle")) if state.get("last_seen_candle") else None
    last_sent_ts = float(state.get("last_sent_ts", 0.0))
    sent_keys = set(state.get("sent_keys", []))  # small set persisted
    open_trade = state.get("open_trade")         # dict version of JournalEntry

    while True:
        try:
            df5 = fetch_ohlcv(cfg.SYMBOL, cfg.TF_TRIGGER, cfg.TWELVEDATA_API_KEY, cfg.LOOKBACK)
            dftrend = fetch_ohlcv(cfg.SYMBOL, cfg.TF_TREND, cfg.TWELVEDATA_API_KEY, cfg.LOOKBACK)

            if len(df5) < 60:
                log.info("Not enough data yet. Sleeping 60s...")
                time.sleep(60)
                continue

            candle_time = df5.index[-1]
            last_close = float(df5["close"].iloc[-1])

            # journal update on every new candle close (or even same, harmless)
            if open_trade:
                from journal import JournalEntry
                je = JournalEntry(**open_trade)
                je = update_journal_status(
                    je,
                    last_close_time=candle_time,
                    last_close=last_close,
                    min_risk_points=cfg.MIN_RISK_POINTS,
                    max_abs_r=cfg.MAX_ABS_R_MULTIPLE
                )
                if je.status != "OPEN":
                    # Send journal update + write CSV
                    msg = f"📒 Journal: {je.trade_id} -> {je.status} | pnl ≈ {je.r_multiple:.2f}R"
                    send_message(cfg.TELEGRAM_BOT_TOKEN, cfg.TELEGRAM_CHAT_ID, msg)
                    append_csv(cfg.JOURNAL_CSV, je.__dict__)
                    open_trade = None
                    state["open_trade"] = None
                    save_state(cfg.STATE_PATH, state)

            # Only evaluate on a NEW candle close
            if last_seen_candle is not None and candle_time <= last_seen_candle:
                log.info("No new candle closed yet (last=%s). Sleeping 60s...", last_seen_candle)
                time.sleep(60)
                continue

            log.info("New candle detected: %s — evaluating...", candle_time)

            sig = generate_signal(
                symbol=cfg.SYMBOL,
                tf_trigger=cfg.TF_TRIGGER,
                df_trigger=df5,
                df_trend=dftrend,
                min_atr_points=cfg.MIN_ATR_POINTS,
                min_bb_width_points=cfg.MIN_BB_WIDTH_POINTS,
                rr_tp1=cfg.RR_TP1,
                rr_tp2=cfg.RR_TP2,
                min_risk_points=cfg.MIN_RISK_POINTS
            )

            # mark candle as processed
            last_seen_candle = candle_time
            state["last_seen_candle"] = candle_time.isoformat()

            # If no signal, persist and sleep
            if sig is None:
                save_state(cfg.STATE_PATH, state)
                time.sleep(60)
                continue

            # Cooldown (prevents spam)
            now = time.time()
            if (now - last_sent_ts) < cfg.COOLDOWN_SECONDS:
                log.info("Cooldown active; skipping signal send.")
                save_state(cfg.STATE_PATH, state)
                time.sleep(60)
                continue

            # Dedupe: one per candle / per setup / per direction
            key = _dedupe_key(sig)
            if key in sent_keys:
                log.info("Duplicate signal key; skipping send.")
                save_state(cfg.STATE_PATH, state)
                time.sleep(60)
                continue

            # Also: if a trade is already open, don’t open another (reduces overtrading)
            if open_trade is not None:
                log.info("Trade still OPEN; skipping new signal.")
                save_state(cfg.STATE_PATH, state)
                time.sleep(60)
                continue

            # Send signal
            send_message(cfg.TELEGRAM_BOT_TOKEN, cfg.TELEGRAM_CHAT_ID, fmt_signal(sig))

            # Open journal entry immediately (signal-only tracking)
            je = open_journal_entry(sig, cfg.SIGNAL_EXPIRY_MINUTES)
            open_trade = je.__dict__
            state["open_trade"] = open_trade

            # update state (cap sent_keys size)
            sent_keys.add(key)
            if len(sent_keys) > 500:
                sent_keys = set(list(sent_keys)[-300:])
            state["sent_keys"] = list(sent_keys)

            state["last_sent_ts"] = now

            save_state(cfg.STATE_PATH, state)
            time.sleep(60)

        except Exception as e:
            log.exception("Loop error: %s", e)
            time.sleep(60)

if __name__ == "__main__":
    run_loop()
