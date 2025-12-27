from dataclasses import dataclass
import pandas as pd
from indicators import ema, rsi, stoch_kd, atr, bollinger, adx

@dataclass(frozen=True)
class Signal:
    symbol: str
    tf: str
    candle_time: pd.Timestamp  # close time of trigger candle
    direction: str             # "BUY" or "SELL"
    entry: float
    sl: float
    tp1: float
    tp2: float
    setup: str
    score: int
    confidence: int
    trend_bias: str
    notes: str

def _score(conf_items: list[bool]) -> int:
    return int(round(100 * sum(conf_items) / max(1, len(conf_items))))

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["rsi14"] = rsi(df["close"], 14)
    df["stoch_k"], df["stoch_d"] = stoch_kd(df["high"], df["low"], df["close"], 14, 3, 3)
    df["atr14"] = atr(df["high"], df["low"], df["close"], 14)
    df["bb_u"], df["bb_m"], df["bb_l"] = bollinger(df["close"], 20, 2)
    df["adx14"], df["pdi"], df["mdi"] = adx(df["high"], df["low"], df["close"], 14)
    df["bb_width"] = (df["bb_u"] - df["bb_l"])
    return df

def infer_trend_bias(df_trend: pd.DataFrame) -> str:
    """
    Simple & robust: use EMA20 vs EMA50 on trend TF.
    """
    f = build_features(df_trend)
    last = f.iloc[-1]
    if last["ema20"] > last["ema50"]:
        return "LONG"
    if last["ema20"] < last["ema50"]:
        return "SHORT"
    return "NEUTRAL"

def generate_signal(
    symbol: str,
    tf_trigger: str,
    df_trigger: pd.DataFrame,
    df_trend: pd.DataFrame,
    min_atr_points: float,
    min_bb_width_points: float,
    rr_tp1: float,
    rr_tp2: float,
    min_risk_points: float
) -> Signal | None:
    """
    Pullback-with-confluence, but now:
    - refuses micro volatility (fixes your 4479 spam)
    - requires BB width + ATR floors
    - uses trend TF as a gate
    """
    ft = build_features(df_trigger)
    last = ft.iloc[-1]
    prev = ft.iloc[-2]

    trend_bias = infer_trend_bias(df_trend)

    # Volatility filters (stop micro spam)
    if float(last["atr14"]) < min_atr_points:
        return None
    if float(last["bb_width"]) < min_bb_width_points:
        return None

    close = float(last["close"])
    bb_m = float(last["bb_m"])
    bb_u = float(last["bb_u"])
    bb_l = float(last["bb_l"])
    st_k = float(last["stoch_k"])
    st_d = float(last["stoch_d"])
    r = float(last["rsi14"])

    # Candle direction simple proxy (avoid counter-candle noise)
    bullish = close > float(last["open"])
    bearish = close < float(last["open"])

    # Define pullback zones
    near_lower_or_mid = close <= bb_m and close >= bb_l
    near_upper_or_mid = close >= bb_m and close <= bb_u

    # Stoch crosses
    stoch_cross_up = float(prev["stoch_k"]) < float(prev["stoch_d"]) and st_k > st_d
    stoch_cross_down = float(prev["stoch_k"]) > float(prev["stoch_d"]) and st_k < st_d

    ema_bull = float(last["ema20"]) > float(last["ema50"])
    ema_bear = float(last["ema20"]) < float(last["ema50"])

    # LONG setup
    long_conds = [
        trend_bias in ("LONG", "NEUTRAL"),
        near_lower_or_mid,
        (stoch_cross_up or (st_k > st_d and st_k < 30)),
        bullish,
        ema_bull,
        r >= 45,   # avoid buying when RSI is dead
    ]

    # SHORT setup
    short_conds = [
        trend_bias in ("SHORT", "NEUTRAL"),
        near_upper_or_mid,
        (stoch_cross_down or (st_k < st_d and st_k > 70)),
        bearish,
        ema_bear,
        r <= 55,   # avoid shorting when RSI is too strong
    ]

    long_score = _score(long_conds)
    short_score = _score(short_conds)

    # Require strong enough confluence to reduce spam
    if max(long_score, short_score) < 67:
        return None

    direction = "BUY" if long_score >= short_score else "SELL"
    score = max(long_score, short_score)

    # Confidence scaled off score + ADX regime (simple)
    adx_v = float(last["adx14"])
    conf = score
    if adx_v >= 25:
        conf += 5
    conf = min(95, conf)

    # SL/TP sizing based on ATR (safer than tiny band-based stops)
    atr_points = float(last["atr14"])
    risk = max(min_risk_points, atr_points * 1.2)

    if direction == "BUY":
        entry = close
        sl = entry - risk
        tp1 = entry + risk * rr_tp1
        tp2 = entry + risk * rr_tp2
        setup = "PULLBACK_LONG"
        notes = f"Pullback LONG: BB zone + Stoch + EMA + RSI | ATR={atr_points:.2f} BBW={float(last['bb_width']):.2f}"
    else:
        entry = close
        sl = entry + risk
        tp1 = entry - risk * rr_tp1
        tp2 = entry - risk * rr_tp2
        setup = "PULLBACK_SHORT"
        notes = f"Pullback SHORT: BB zone + Stoch + EMA + RSI | ATR={atr_points:.2f} BBW={float(last['bb_width']):.2f}"

    candle_time = ft.index[-1]

    return Signal(
        symbol=symbol,
        tf=tf_trigger,
        candle_time=candle_time,
        direction=direction,
        entry=entry,
        sl=sl,
        tp1=tp1,
        tp2=tp2,
        setup=setup,
        score=int(round(score)),
        confidence=int(round(conf)),
        trend_bias=trend_bias,
        notes=notes,
    )
