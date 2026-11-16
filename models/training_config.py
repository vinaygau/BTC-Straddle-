"""Explicit research defaults for the first supervised market-model pass."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelTrainingConfig:
    horizon_steps: int = 12                 # 1 hour on the 5-minute base timeframe
    sequence_length: int = 48               # 4 hours of completed 5-minute candles
    sequence_stride: int = 12               # evaluate/train one non-overlapping 1-hour target at a time
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    train_boundary_index: int | None = None
    validation_boundary_index: int | None = None
    direction_threshold: float = 0.0025     # ±0.25% over the prediction horizon
    extreme_move_threshold: float = 0.01    # absolute 1% over the prediction horizon
    random_seed: int = 42
    lstm_epochs: int = 20
    lstm_batch_size: int = 128
    xgboost_estimators: int = 300

