"""Causal candle-derived features and point-in-time multi-timeframe alignment."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config.data_config import SUPPORTED_TIMEFRAMES, TIMEFRAME_MINUTES

PROCESSED_FILES = {
    "5m": "btc_5m.parquet",
    "15m": "btc_15m.parquet",
    "1h": "btc_1h.parquet",
    "daily": "btc_daily.parquet",
}
RETURN_WINDOWS = (1, 3, 6, 12, 24)
VOLATILITY_WINDOWS = (6, 12, 24, 48)


def candle_availability_times(frame: pd.DataFrame, timeframe: str) -> pd.Series:
    """Return inclusive UTC close times, deriving them only when absent."""
    if timeframe not in TIMEFRAME_MINUTES:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    inferred = frame["timestamp_utc"] + pd.to_timedelta(TIMEFRAME_MINUTES[timeframe], unit="m") - pd.Timedelta(microseconds=1)
    if "close_time_utc" in frame:
        known = frame.close_time_utc
        return known.where(known.notna(), inferred)
    return inferred


def engineer_timeframe_features(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Compute causal candle, momentum, volatility, volume, and UTC calendar features."""
    required = {"timestamp_utc", "open", "high", "low", "close", "volume"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{timeframe} candle data is missing required columns: {missing}")
    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    candles = frame.sort_values("timestamp_utc", kind="stable").drop_duplicates("timestamp_utc", keep="first").reset_index(drop=True).copy()
    close = candles.close.astype("float64")
    log_close = np.log(close.where(close > 0))
    one_bar_log_return = log_close.diff()
    candle_range = (candles.high - candles.low).replace(0, np.nan)
    upper_wick = candles.high - candles[["open", "close"]].max(axis=1)
    lower_wick = candles[["open", "close"]].min(axis=1) - candles.low

    result = pd.DataFrame(index=candles.index)
    result["timestamp_utc"] = candles.timestamp_utc
    result["available_at_utc"] = candle_availability_times(candles, timeframe)
    result["close"] = close
    result["log_return_1"] = one_bar_log_return
    result["body_return"] = candles.close / candles.open.where(candles.open != 0) - 1.0
    result["range_pct"] = (candles.high - candles.low) / close.where(close != 0)
    result["body_to_range"] = (candles.close - candles.open) / candle_range
    result["upper_wick_to_range"] = upper_wick / candle_range
    result["lower_wick_to_range"] = lower_wick / candle_range

    for window in RETURN_WINDOWS:
        result[f"momentum_return_{window}"] = close / close.shift(window) - 1.0
    for window in VOLATILITY_WINDOWS:
        result[f"realized_volatility_{window}"] = one_bar_log_return.rolling(window, min_periods=window).std(ddof=0)

    volume = candles.volume.astype("float64")
    result["volume_log1p"] = np.log1p(volume.where(volume >= 0))
    result["volume_change_1"] = volume.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    volume_mean = volume.rolling(24, min_periods=24).mean()
    volume_std = volume.rolling(24, min_periods=24).std(ddof=0).replace(0, np.nan)
    result["volume_zscore_24"] = (volume - volume_mean) / volume_std

    if "quote_volume" in candles:
        result["quote_volume_log1p"] = np.log1p(candles.quote_volume.where(candles.quote_volume >= 0))
    if "trade_count" in candles:
        trades = candles.trade_count.astype("float64")
        result["trade_count_change_1"] = trades.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    if "taker_buy_base_volume" in candles:
        result["taker_buy_base_ratio"] = candles.taker_buy_base_volume / volume.where(volume > 0)
    if "taker_buy_quote_volume" in candles and "quote_volume" in candles:
        result["taker_buy_quote_ratio"] = candles.taker_buy_quote_volume / candles.quote_volume.where(candles.quote_volume > 0)

    utc = candles.timestamp_utc.dt.tz_convert("UTC")
    hour_fraction = (utc.dt.hour + utc.dt.minute / 60.0) / 24.0
    weekday_fraction = utc.dt.dayofweek / 7.0
    result["utc_hour_sin"] = np.sin(2 * np.pi * hour_fraction)
    result["utc_hour_cos"] = np.cos(2 * np.pi * hour_fraction)
    result["utc_weekday_sin"] = np.sin(2 * np.pi * weekday_fraction)
    result["utc_weekday_cos"] = np.cos(2 * np.pi * weekday_fraction)
    return result


def build_multitimeframe_features(
    datasets: dict[str, pd.DataFrame], *, base_timeframe: str = "5m",
    timeframes: tuple[str, ...] = SUPPORTED_TIMEFRAMES,
) -> pd.DataFrame:
    """Align the latest fully closed timeframe features to each base candle close.

    The output row timestamp is the prediction/as-of time. Higher-timeframe rows
    are backward-as-of joined with a bounded age; future and overly stale rows
    remain null instead of leaking or being forward-filled indefinitely.
    """
    if base_timeframe not in datasets:
        raise ValueError(f"Base timeframe {base_timeframe!r} was not supplied")
    if base_timeframe not in timeframes:
        raise ValueError("base_timeframe must be included in timeframes")
    engineered = {}
    for timeframe in timeframes:
        if timeframe in datasets and not datasets[timeframe].empty:
            engineered[timeframe] = engineer_timeframe_features(datasets[timeframe], timeframe)

    base = engineered.get(base_timeframe)
    if base is None or base.empty:
        raise ValueError("Base timeframe contains no usable candles")
    result = pd.DataFrame({
        "prediction_time_utc": base.available_at_utc,
        "base_candle_time_utc": base.timestamp_utc,
    })
    for column in base.columns:
        if column not in {"timestamp_utc", "available_at_utc"}:
            result[f"{base_timeframe}_{column}"] = base[column].to_numpy()

    for timeframe in timeframes:
        if timeframe == base_timeframe or timeframe not in engineered:
            continue
        right = engineered[timeframe].copy()
        right = right.rename(columns={
            "timestamp_utc": f"{timeframe}_source_candle_time_utc",
            "available_at_utc": f"{timeframe}_available_at_utc",
            **{col: f"{timeframe}_{col}" for col in engineered[timeframe].columns
               if col not in {"timestamp_utc", "available_at_utc"}},
        })
        right = right.sort_values(f"{timeframe}_available_at_utc", kind="stable")
        result = pd.merge_asof(
            result.sort_values("prediction_time_utc", kind="stable"),
            right,
            left_on="prediction_time_utc",
            right_on=f"{timeframe}_available_at_utc",
            direction="backward",
            tolerance=pd.Timedelta(minutes=2 * TIMEFRAME_MINUTES[timeframe]),
            allow_exact_matches=True,
        )
        matched = result[f"{timeframe}_available_at_utc"].notna()
        if (result.loc[matched, f"{timeframe}_available_at_utc"] > result.loc[matched, "prediction_time_utc"]).any():
            raise RuntimeError(f"Future {timeframe} data leaked into a prediction row")
    return result.sort_values("prediction_time_utc", kind="stable").reset_index(drop=True)


def load_processed_datasets(processed_dir: str | Path, *, timeframes: tuple[str, ...] = SUPPORTED_TIMEFRAMES) -> dict[str, pd.DataFrame]:
    directory = Path(processed_dir)
    datasets = {}
    for timeframe in timeframes:
        path = directory / PROCESSED_FILES[timeframe]
        if path.exists():
            datasets[timeframe] = pd.read_parquet(path)
    return datasets


def write_multitimeframe_features(
    processed_dir: str | Path, *, output_path: str | Path | None = None,
    base_timeframe: str = "5m", timeframes: tuple[str, ...] = SUPPORTED_TIMEFRAMES,
) -> Path:
    directory = Path(processed_dir)
    features = build_multitimeframe_features(
        load_processed_datasets(directory, timeframes=timeframes),
        base_timeframe=base_timeframe, timeframes=timeframes,
    )
    target = Path(output_path) if output_path else directory / "btc_multitimeframe_features.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(target, index=False)
    return target

