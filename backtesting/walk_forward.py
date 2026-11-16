"""Chronological expanding-window fold construction with explicit purges."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WalkForwardFold:
    fold: int
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    train_start_utc: str
    train_end_utc: str
    validation_start_utc: str
    validation_end_utc: str
    test_start_utc: str
    test_end_utc: str


def expanding_walk_forward_folds(
    timestamps,
    *,
    initial_train_size: int,
    validation_size: int,
    test_size: int,
    step_size: int | None = None,
    purge_rows: int = 12,
) -> tuple[WalkForwardFold, ...]:
    """Create expanding train folds followed by validation/test blocks.

    The helper does not fit or tune a model. Callers must fit preprocessing and
    estimators inside each fold using only that fold's training rows.
    """
    times = pd.to_datetime(pd.Series(timestamps), utc=True, errors="coerce")
    if times.isna().any() or not times.is_monotonic_increasing:
        raise ValueError("timestamps must be valid and sorted in chronological order")
    if times.duplicated().any():
        raise ValueError("timestamps must be unique")
    if min(initial_train_size, validation_size, test_size) <= 0 or purge_rows < 0:
        raise ValueError("fold sizes must be positive and purge_rows cannot be negative")
    step_size = test_size if step_size is None else step_size
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    n = len(times)
    folds: list[WalkForwardFold] = []
    train_end = initial_train_size
    while True:
        val_start = train_end + purge_rows
        val_end = val_start + validation_size
        test_start = val_end + purge_rows
        test_end = test_start + test_size
        if test_end > n:
            break
        train = np.arange(0, train_end, dtype=int)
        validation = np.arange(val_start, val_end, dtype=int)
        test = np.arange(test_start, test_end, dtype=int)
        fold_number = len(folds) + 1
        folds.append(WalkForwardFold(
            fold=fold_number,
            train_indices=train,
            validation_indices=validation,
            test_indices=test,
            train_start_utc=times.iloc[train[0]].isoformat(),
            train_end_utc=times.iloc[train[-1]].isoformat(),
            validation_start_utc=times.iloc[validation[0]].isoformat(),
            validation_end_utc=times.iloc[validation[-1]].isoformat(),
            test_start_utc=times.iloc[test[0]].isoformat(),
            test_end_utc=times.iloc[test[-1]].isoformat(),
        ))
        train_end += step_size
    if not folds:
        raise ValueError("Not enough observations to construct one purged walk-forward fold")
    return tuple(folds)
