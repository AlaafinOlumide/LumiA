# backtest.py
import argparse
import datetime as dt
import pandas as pd

import indicators
from strategy import (
    detect_trend_h1,
    detect_trend_m15_direction,
    confirm_trend_m15,
    trigger_signal_m5,
)

from trade_manager import ActiveTrade, new_trade_id, check_tp_sl_hit, should_invalidate, compute_r_result
from data_logger import TradeJournal
from cooldown import in_cooldown


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    tmp = df.copy()
    tmp["datetime"] = pd.to_datetime(tmp["datetime"], utc=True)
    tmp = tmp.set_index("datetime")

    agg = (
        tmp.resample(rule)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .dropna()
    )
    return agg.reset_index()


def tp_sl_multipliers(setup_type: str):
    if setup_type.startswith("PULLBACK"):
        return (0.90, 1.60, 2.60)
    if setup_type.startswith("BREAKOUT_CONT"):
        return (0.55, 0.90, 1.50)
    if setup_type.startswith("BREAKOUT"):
        return (0.70, 1.20, 2.00)
    return (0.75, 1.10, 1.80)


def apply_dynamic_tp_sl(signal, h1_df: pd.DataFrame) -> None:
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


def compute_confidence(trend_source: str, setup_type: str, adx_h1: float, adx_m5: float, high_news: bool) -> int:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to historical M5 CSV with columns: datetime,open,high,low,close,volume")
    ap.add_argument("--journal", default="trades_backtest.csv", help="Output journal CSV path")
    ap.add_argument("--cooldown", type=int, default=20, help="Cooldown minutes between trades")
    ap.add_argument("--invalidate", type=int, default=20, help="Invalidation minutes after entry")
    ap.add_argument("--rolling", type=int, default=60, help="Rolling N for adaptive confidence")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)

    journal = TradeJournal(csv_path=args.journal, rolling_n=args.rolling)

    active_trade: ActiveTrade | None = None
    last_signal_time: dt.datetime | None = None

    # Walk forward candle-by-candle
    for i in range(200, len(df)):  # start once we have enough history
        slice_m5 = df.iloc[: i + 1].copy()
        last_row = slice_m5.iloc[-1]
        now_utc = last_row["datetime"].to_pydatetime()

        # cooldown
        if in_cooldown(now_utc, last_signal_time, minutes=args.cooldown):
            continue

        # manage active trade
        if active_trade is not None and active_trade.status == "OPEN":
            hit = check_tp_sl_hit(active_trade, last_row)
            if hit:
                active_trade.status = hit["status"]
                active_trade.exit_time = now_utc
                active_trade.exit_price = float(hit["exit_price"])
                active_trade.result_r = compute_r_result(
                    active_trade.direction, active_trade.entry, active_trade.sl, active_trade.exit_price
                )
                journal.update_trade(active_trade)
                last_signal_time = now_utc
                active_trade = None
                continue

            # invalidation window
            if active_trade.invalidation_deadline and now_utc <= active_trade.invalidation_deadline:
                close_series = slice_m5["close"]
                bb_u, bb_m, bb_l = indicators.bollinger_bands(close_series, period=20, std_factor=2.0)
                rsi_series = indicators.rsi(close_series, period=14)

                inv_reason = should_invalidate(
                    active_trade,
                    last_row,
                    bb_mid=float(bb_m.iloc[-1]),
                    bb_upper=float(bb_u.iloc[-1]),
                    bb_lower=float(bb_l.iloc[-1]),
                    rsi_m5=float(rsi_series.iloc[-1]) if pd.notna(rsi_series.iloc[-1]) else 50.0,
                )
                if inv_reason:
                    active_trade.status = "INVALIDATED"
                    active_trade.exit_time = now_utc
                    active_trade.exit_price = float(last_row["close"])
                    active_trade.result_r = compute_r_result(
                        active_trade.direction, active_trade.entry, active_trade.sl, active_trade.exit_price
                    )
                    active_trade.invalidated_reason = inv_reason
                    journal.update_trade(active_trade)
                    last_signal_time = now_utc
                    active_trade = None
                    continue

        # Build HTFs
        m15_df = resample_ohlc(slice_m5, "15min")
        h1_df = resample_ohlc(slice_m5, "1h")
        if len(m15_df) < 60 or len(h1_df) < 60:
            continue

        # trend
        h1_trend = detect_trend_h1(h1_df)
        trend_source = "H1"
        if h1_trend is None:
            m15_dir = detect_trend_m15_direction(m15_df)
            if m15_dir is None:
                continue
            trend_dir = m15_dir
            trend_source = "M15"
        else:
            trend_dir = h1_trend
            _ = confirm_trend_m15(m15_df, trend_dir)

        # trigger
        signal = trigger_signal_m5(slice_m5, trend_dir)
        if not signal:
            continue

        apply_dynamic_tp_sl(signal, h1_df)

        # ADX(H1)
        adx_h1_series, _, _ = indicators.adx(h1_df["high"], h1_df["low"], h1_df["close"], period=14)
        adx_h1 = float(adx_h1_series.iloc[-1]) if pd.notna(adx_h1_series.iloc[-1]) else 0.0

        setup_type = signal.extra.get("setup_type", "GENERIC")
        adx_m5 = float(signal.extra.get("adx_m5", 0.0))

        base_conf = compute_confidence(trend_source, setup_type, adx_h1, adx_m5, high_news=False)
        adj = journal.adaptive_confidence_adjustment(setup_type, trend_source)
        conf = max(0, min(100, base_conf + adj))
        signal.extra["confidence"] = conf

        # open trade
        entry = float(signal.price)
        sl = float(signal.extra["sl"])
        tp1 = float(signal.extra["tp1"])
        tp2 = float(signal.extra["tp2"])

        opened_time = signal.time if signal.time.tzinfo else signal.time.replace(tzinfo=dt.timezone.utc)

        active_trade = ActiveTrade(
            trade_id=new_trade_id(),
            opened_time=opened_time,
            direction=signal.direction,
            setup_type=setup_type,
            entry=entry,
            sl=sl,
            tp1=tp1,
            tp2=tp2,
            confidence=conf,
            trend_source=trend_source,
        )
        active_trade.invalidation_deadline = opened_time + dt.timedelta(minutes=args.invalidate)

        journal.append_open(active_trade)
        last_signal_time = opened_time

    # summary
    print(f"Backtest complete. Journal written to: {args.journal}")
    print("Tip: compute win-rate by reading the journal and counting TP1/TP2 vs SL/INVALIDATED.")


if __name__ == "__main__":
    main()
