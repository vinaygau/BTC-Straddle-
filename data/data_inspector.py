"""Dataset-level diagnostics for the inspection report."""
from .data_validator import missing_intervals, validate_candles


def inspect_frame(frame, *, file_name: str, timeframe: str | None) -> dict:
    ts = frame.get("timestamp_utc")
    report = {
        "file_name": file_name, "timeframe": timeframe, "records": int(len(frame)),
        "columns_count": int(len(frame.columns)), "columns": list(frame.columns),
        "dtypes": {k: str(v) for k, v in frame.dtypes.items()},
        "first_timestamp_utc": None if ts is None or ts.dropna().empty else ts.min().isoformat(),
        "last_timestamp_utc": None if ts is None or ts.dropna().empty else ts.max().isoformat(),
        "missing_values": {k: int(v) for k, v in frame.isna().sum().items() if v},
        "invalid_timestamp_rows": int(ts.isna().sum()) if ts is not None else len(frame),
        "duplicate_records": int(frame.duplicated().sum()),
        "duplicate_timestamps": int(ts.duplicated().sum()) if ts is not None else None,
        "validation": validate_candles(frame),
        "missing_candle_intervals": missing_intervals(frame, timeframe) if timeframe else [],
    }
    if "close" in frame and frame.close.notna().any():
        report["min_price"] = float(frame[["open", "high", "low", "close"]].min().min())
        report["max_price"] = float(frame[["open", "high", "low", "close"]].max().max())
    return report

