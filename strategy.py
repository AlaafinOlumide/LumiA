import datetime as dt

def active_session(
    now_utc: dt.datetime,
    enable_asia: bool,
    trade_weekends: bool,
    asia_start_hour_utc: int = 0,
    asia_end_hour_utc: int = 3,
    london_start_hour_utc: int = 6,
    london_end_hour_utc: int = 21,
) -> str | None:
    # weekend guard
    if not trade_weekends:
        # Saturday=5, Sunday=6
        if now_utc.weekday() in (5, 6):
            return None

    h = now_utc.hour

    if enable_asia and asia_start_hour_utc <= h < asia_end_hour_utc:
        return "ASIA"

    if london_start_hour_utc <= h < london_end_hour_utc:
        return "LONDON_NY"

    return None