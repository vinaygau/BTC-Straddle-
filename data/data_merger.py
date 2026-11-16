"""Deterministic candle merge with documented source-priority deduplication."""
import pandas as pd


def merge_datasets(frames: list[pd.DataFrame]) -> pd.DataFrame:
    valid = [f for f in frames if not f.empty and "timestamp_utc" in f]
    if not valid: return pd.DataFrame()
    merged = pd.concat(valid, ignore_index=True, sort=False)
    if "source_priority" not in merged: merged["source_priority"] = 99
    # Fetched Binance rows outrank raw monthly Binance, which outrank third-party exports.
    merged = merged.sort_values(["timestamp_utc", "source_priority", "source_file"], kind="stable")
    merged = merged.drop_duplicates(subset=["timestamp_utc"], keep="first")
    return merged.sort_values("timestamp_utc", kind="stable").reset_index(drop=True)

