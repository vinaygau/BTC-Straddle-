"""Leakage-aware labels, numeric features, and chronological dataset splits."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .training_config import ModelTrainingConfig


@dataclass(frozen=True)
class TemporalSplits:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    feature_columns: tuple[str, ...]


def prepare_training_frame(features: pd.DataFrame, config: ModelTrainingConfig | None = None) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Add forward-only targets and drop rows without observed inputs/targets."""
    config = config or ModelTrainingConfig()
    if "prediction_time_utc" not in features or "5m_close" not in features:
        raise ValueError("Feature input requires prediction_time_utc and 5m_close")
    excluded = {"prediction_time_utc", "base_candle_time_utc"}
    excluded |= {col for col in features if col.endswith(("_available_at_utc", "_source_candle_time_utc", "_close"))}
    feature_columns = tuple(
        col for col in features.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(features[col])
    )
    if not feature_columns:
        raise ValueError("No numeric model features were found")

    ordered = features.sort_values("prediction_time_utc", kind="stable").reset_index(drop=True).copy()
    close = pd.to_numeric(ordered["5m_close"], errors="coerce")
    future_close = close.shift(-config.horizon_steps)
    ordered["target_return"] = future_close / close - 1.0

    log_close = np.log(close.where(close > 0))
    future_bar_returns = [log_close.shift(-step) - log_close.shift(-(step - 1))
                          for step in range(1, config.horizon_steps + 1)]
    future_returns = pd.concat(future_bar_returns, axis=1)
    full_horizon = future_returns.notna().all(axis=1)
    ordered["target_volatility"] = future_returns.std(axis=1, ddof=0).where(full_horizon)
    timestamps = pd.to_datetime(ordered.prediction_time_utc, utc=True)
    horizon_end = timestamps.shift(-config.horizon_steps)
    exact_horizon = horizon_end - timestamps == pd.Timedelta(minutes=5 * config.horizon_steps)
    ordered.loc[~exact_horizon, ["target_return", "target_volatility"]] = np.nan
    ordered["target_direction"] = np.select(
        [ordered.target_return >= config.direction_threshold,
         ordered.target_return <= -config.direction_threshold],
        [2, 0], default=1,
    ).astype("int8")  # 0=bearish, 1=sideways, 2=bullish
    ordered["target_extreme_move"] = (ordered.target_return.abs() >= config.extreme_move_threshold).astype("int8")

    numeric = ordered.loc[:, feature_columns].replace([np.inf, -np.inf], np.nan)
    ordered.loc[:, feature_columns] = numeric
    valid = numeric.notna().all(axis=1)
    valid &= ordered[["target_return", "target_volatility", "target_direction", "target_extreme_move"]].notna().all(axis=1)
    ordered = ordered.loc[valid].reset_index(drop=True)
    if ordered.empty:
        raise ValueError("No rows remain after feature warm-up and forward target alignment")
    return ordered, feature_columns


def chronological_splits(frame: pd.DataFrame, feature_columns: tuple[str, ...],
                         config: ModelTrainingConfig | None = None) -> TemporalSplits:
    """Create chronological splits with a horizon-sized purge between each split."""
    config = config or ModelTrainingConfig()
    n = len(frame)
    train_boundary = (config.train_boundary_index if config.train_boundary_index is not None
                      else int(n * config.train_fraction))
    validation_boundary = (config.validation_boundary_index if config.validation_boundary_index is not None
                           else int(n * (config.train_fraction + config.validation_fraction)))
    if not (0 < train_boundary < validation_boundary < n):
        raise ValueError("Configured chronological split boundaries must be within the labeled frame")
    purge = config.horizon_steps
    train = np.arange(0, max(0, train_boundary - purge))
    validation = np.arange(train_boundary, max(train_boundary, validation_boundary - purge))
    test = np.arange(validation_boundary, n)
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError(f"Not enough labeled rows ({n}) for configured splits and purge gap")
    times = pd.to_datetime(frame.prediction_time_utc, utc=True)
    if not (times.iloc[train[-1]] < times.iloc[validation[0]] < times.iloc[test[0]]):
        raise RuntimeError("Temporal split ordering invariant failed")
    return TemporalSplits(train, validation, test, feature_columns)


def fit_feature_scaler(frame: pd.DataFrame, train_indices: np.ndarray, feature_columns: tuple[str, ...]):
    """Fit standardization on training rows only; return train-fitted parameters."""
    matrix = frame.loc[:, feature_columns].to_numpy(dtype=np.float64)
    mean = matrix[train_indices].mean(axis=0)
    scale = matrix[train_indices].std(axis=0)
    scale[~np.isfinite(scale) | (scale == 0)] = 1.0
    return mean, scale


def transform_features(frame: pd.DataFrame, feature_columns: tuple[str, ...], mean, scale) -> np.ndarray:
    values = frame.loc[:, feature_columns].to_numpy(dtype=np.float64)
    return np.clip((values - mean) / scale, -10.0, 10.0).astype(np.float32)

