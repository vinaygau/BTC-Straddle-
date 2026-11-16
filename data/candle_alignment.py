"""Align available data to candles that have fully closed by prediction time."""
import pandas as pd


def completed_candles(frame: pd.DataFrame, as_of, *, timeframe: str | None = None) -> pd.DataFrame:
    if "close_time_utc" in frame:
        closes = frame.close_time_utc
    else:
        minutes = {"5m": 5, "15m": 15, "1h": 60, "daily": 1440}
        interval = timeframe or frame.attrs.get("timeframe")
        if interval not in minutes: raise ValueError("close_time_utc or timeframe is required")
        closes = frame.timestamp_utc + pd.to_timedelta(minutes[interval], unit="m")
    cutoff = pd.Timestamp(as_of)
    if cutoff.tzinfo is None: cutoff = cutoff.tz_localize("UTC")
    return frame.loc[closes <= cutoff.tz_convert("UTC")].copy()

