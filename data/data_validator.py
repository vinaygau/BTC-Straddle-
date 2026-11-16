"""Candle row validation and timestamp-grid gap detection."""
import numpy as np
import pandas as pd

REQUIRED = ("timestamp_utc", "open", "high", "low", "close", "volume")


def validate_candles(frame: pd.DataFrame) -> dict:
    missing_columns = [c for c in REQUIRED if c not in frame]
    if missing_columns: return {"missing_required_columns": missing_columns, "invalid_ohlc_rows": None, "invalid_volume_rows": None}
    px = frame[["open", "high", "low", "close"]]
    invalid_ohlc = (~np.isfinite(px).all(axis=1) | (px.le(0).any(axis=1)) |
                    (frame.high < frame[["open", "close", "low"]].max(axis=1)) |
                    (frame.low > frame[["open", "close", "high"]].min(axis=1)))
    invalid_volume = ~np.isfinite(frame.volume) | frame.volume.lt(0)
    return {"missing_required_columns": [], "invalid_ohlc_rows": int(invalid_ohlc.sum()),
            "zero_or_negative_price_rows": int((px.le(0).any(axis=1)).sum()),
            "invalid_volume_rows": int(invalid_volume.sum())}


def valid_candle_mask(frame: pd.DataFrame) -> pd.Series:
    """Rows safe for the processed OHLCV output; never fills or repairs values."""
    missing = [c for c in REQUIRED if c not in frame]
    if missing: raise ValueError(f"Cannot preprocess candles; required columns missing: {missing}")
    px = frame[["open", "high", "low", "close"]]
    valid = frame.timestamp_utc.notna() & px.notna().all(axis=1) & np.isfinite(px).all(axis=1)
    valid &= px.gt(0).all(axis=1) & (frame.high >= px[["open", "close", "low"]].max(axis=1))
    valid &= frame.low <= px[["open", "close", "high"]].min(axis=1)
    valid &= frame.volume.notna() & np.isfinite(frame.volume) & frame.volume.ge(0)
    return valid


def missing_intervals(frame: pd.DataFrame, timeframe: str) -> list[dict]:
    minutes = {"5m": 5, "15m": 15, "1h": 60, "daily": 1440}.get(timeframe)
    if not minutes or "timestamp_utc" not in frame: return []
    times = pd.DatetimeIndex(frame.timestamp_utc.dropna().drop_duplicates().sort_values())
    if len(times) < 2: return []
    expected = pd.Timedelta(minutes=minutes)
    gaps = []
    for left, right in zip(times[:-1], times[1:]):
        delta = right - left
        if delta > expected:
            count = int(delta / expected) - 1
            gaps.append({"after": left.isoformat(), "before": right.isoformat(), "missing_candles": count})
    return gaps

