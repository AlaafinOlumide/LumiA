from dataclasses import dataclass
import os

def _env(name: str, default: str | None = None) -> str:
    v = os.getenv(name, default)
    if v is None or v == "":
        raise RuntimeError(f"Missing required env var: {name}")
    return v

@dataclass(frozen=True)
class Config:
    # Data
    TWELVEDATA_API_KEY: str = _env("TWELVEDATA_API_KEY")
    SYMBOL: str = os.getenv("SYMBOL", "XAU/USD")
    # Candle timeframe used to trigger entries
    TF_TRIGGER: str = os.getenv("TF_TRIGGER", "5min")  # 5min
    # Confirmation timeframe for trend bias
    TF_TREND: str = os.getenv("TF_TREND", "15min")     # 15min or 1h
    LOOKBACK: int = int(os.getenv("LOOKBACK", "300"))

    # Telegram
    TELEGRAM_BOT_TOKEN: str = _env("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID: str = _env("TELEGRAM_CHAT_ID")

    # Risk model (signal-only; journaling uses these)
    RR_TP1: float = float(os.getenv("RR_TP1", "1.2"))
    RR_TP2: float = float(os.getenv("RR_TP2", "2.0"))

    # Anti-spam / filtering (these fix your “4479.6 spam” issue)
    COOLDOWN_SECONDS: int = int(os.getenv("COOLDOWN_SECONDS", "900"))  # 15 min
    MIN_ATR_POINTS: float = float(os.getenv("MIN_ATR_POINTS", "4.0"))  # stop micro volatility
    MIN_BB_WIDTH_POINTS: float = float(os.getenv("MIN_BB_WIDTH_POINTS", "6.0"))

    # Expiry (if TP/SL not hit, close journal at expiry using last close)
    SIGNAL_EXPIRY_MINUTES: int = int(os.getenv("SIGNAL_EXPIRY_MINUTES", "60"))

    # Journal + storage paths (Render-friendly)
    STATE_PATH: str = os.getenv("STATE_PATH", "state.json")
    JOURNAL_CSV: str = os.getenv("JOURNAL_CSV", "journal.csv")

    # Safety clamps
    MIN_RISK_POINTS: float = float(os.getenv("MIN_RISK_POINTS", "2.0"))  # prevents -75R nonsense
    MAX_ABS_R_MULTIPLE: float = float(os.getenv("MAX_ABS_R_MULTIPLE", "6.0"))
