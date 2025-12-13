# backtest.py
import datetime as dt
import pandas as pd

import indicators
from strategy import detect_trend_h1, detect_trend_m15_direction, confirm_trend_m15, trigger_signal_m5
from trade_manager import ActiveTrade, new_trade_id, check_tp_sl_hit, should_invalidate, compute_r_result


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


def apply_dynamic_tp_sl(signal, h1_df):
    atr_series = indicators.atr(h1_df["high"], h1_df["low"], h1_df["close"], period=14)
    atr_h1 = float(atr_series.iloc[-1]) if pd.notna(atr_series.iloc[-1]) else None
    if atr_h1 is None or atr_h1 <= 0:
        return False

    setup_type = signal.extra.get("setup_type", "GENERIC")
    if setup_type.startswith("PULLBACK"):
        sl_mult, tp1_mult, tp2_mult = (0.90, 1.60, 2.60)
    elif setup_type.startswith("BREAKOUT_CONT"):
        sl_mult, tp1_mult, tp2_mult = (0.55, 0.90, 1.50)
    elif setup_type.startswith("BREAKOUT"):
        sl_mult, tp1_mult, tp2_mult = (0.70, 1.20, 2.00)
    else:
        sl_mult, tp1_mult, tp2_mult = (0.75, 1.10, 1.80)

    entry = float(signal.price)
    if signal.direction == "LONG":
        sl = entry - (atr_h1 * sl_mult)
        tp1 = entry + (atr_h1 * tp1_mult)
        tp2 = entry + (atr_h1 * tp2_mult)
    else:
        sl = entry + (atr_h1 * sl_mult)
        tp1 = entry - (atr_h1 * tp1_mult)
        tp2 = entry - (atr_h1 * tp2_mult)

    signal.extra["sl"] = sl
    signal.extra["tp1"] = tp1
    signal.extra["tp2"] = tp2
    signal.extra["atr_h1"] = atr_h1
    return True


def backtest(csv_path="m5.csv", cooldown_minutes=20):
    df = pd.read_csv(csv_path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)

    trades = []
    active = None
    last_signal_time = None

    for i in range(300, len(df)):
        window = df.iloc[: i + 1].copy()  # candles up to i
        last_candle = window.iloc[-1]
        now = last_candle["datetime"].to_pydatetime()

        # Update active
        if active is not None and active.status == "OPEN":
            hit = check_tp_sl_hit(active, last_candle)
            if hit:
                active.status = hit["status"]
                active.exit_time = now
                active.exit_price = float(hit["exit_price"])
                active.result_r = compute_r_result(active.direction, active.entry, active.sl, active.exit_price)
                trades.append(active)
                active = None
                last_signal_time = now
                continue

            close = window["close"]
            bb_u, bb_m, bb_l = indicators.bollinger_bands(close, period=20, std_factor=2.0)
            rsi_m5 = float(indicators.rsi(close, period=14).iloc[-1]) if len(close) >= 20 else 50.0

            reason = should_invalidate(
                trade=active,
                last_closed_m5=last_candle,
                bb_mid=float(bb_m.iloc[-1]),
                bb_upper=float(bb_u.iloc[-1]),
                bb_lower=float(bb_l.iloc[-1]),
                rsi_m5=rsi_m5,
            )
            if active.invalidation_deadline and now <= active.invalidation_deadline and reason:
                active.status = "INVALIDATED"
                active.invalidated_reason = reason
                active.exit_time = now
                active.exit_price = float(last_candle["close"])
                active.result_r = compute_r_result(active.direction, active.entry, active.sl, active.exit_price)
                trades.append(active)
                active = None
                last_signal_time = now
                continue

        # no new trades if one open
        if active is not None:
            continue

        # cooldown
        if last_signal_time is not None:
            if (now - last_signal_time).total_seconds() < cooldown_minutes * 60:
                continue

        # resample
        m15 = resample_ohlc(window, "15min")
        h1 = resample_ohlc(window, "1h")
        if len(m15) < 60 or len(h1) < 60:
            continue

        h1_trend = detect_trend_h1(h1)
        trend_source = "H1"
        if h1_trend is None:
            m15_dir = detect_trend_m15_direction(m15)
            if m15_dir is None:
                continue
            trend_dir = m15_dir
            trend_source = "M15"
        else:
            trend_dir = h1_trend
            _ = confirm_trend_m15(m15, trend_dir)

        sig = trigger_signal_m5(window, trend_dir)
        if not sig:
            continue

        ok = apply_dynamic_tp_sl(sig, h1)
        if not ok:
            continue

        trade = ActiveTrade(
            trade_id=new_trade_id(),
            opened_time=sig.time,
            direction=sig.direction,
            setup_type=sig.extra.get("setup_type", "GENERIC"),
            entry=float(sig.price),
            sl=float(sig.extra["sl"]),
            tp1=float(sig.extra["tp1"]),
            tp2=float(sig.extra["tp2"]),
            confidence=0,
            trend_source=trend_source,
            invalidation_deadline=(now + dt.timedelta(minutes=10)),
        )
        active = trade
        last_signal_time = now

    # If ends with open, close at last close
    if active is not None and active.status == "OPEN":
        last = df.iloc[-1]
        now = last["datetime"].to_pydatetime()
        active.exit_time = now
        active.exit_price = float(last["close"])
        active.result_r = compute_r_result(active.direction, active.entry, active.sl, active.exit_price)
        active.status = "EOD"
        trades.append(active)

    out = pd.DataFrame([t.__dict__ for t in trades])
    out.to_csv("backtest_trades.csv", index=False)

    # Summary
    wins = out[out["result_r"] > 0]
    losses = out[out["result_r"] <= 0]
    print("Trades:", len(out))
    print("Win rate:", (len(wins) / len(out) * 100) if len(out) else 0, "%")
    print("Avg R:", out["result_r"].mean() if len(out) else 0)
    print("Total R:", out["result_r"].sum() if len(out) else 0)
    print("Saved: backtest_trades.csv")


if __name__ == "__main__":
    backtest("m5.csv")
