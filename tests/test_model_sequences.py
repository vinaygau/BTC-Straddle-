import numpy as np
import pandas as pd
import pytest

from models.train import _sequence_batch


def test_sequence_batch_strides_examples_and_keeps_each_window_contiguous():
    values = np.arange(40, dtype=np.float32).reshape(20, 2)
    targets = {"return": np.arange(20, dtype=np.float32)}
    times = pd.Series(pd.date_range("2025-01-01", periods=20, freq="5min", tz="UTC"))

    sequences, kept, labels = _sequence_batch(
        np.arange(5, 18), values, targets, times, sequence_length=4, sequence_stride=3
    )

    assert kept.tolist() == [5, 8, 11, 14, 17]
    assert sequences.shape == (5, 4, 2)
    assert labels["return"].tolist() == kept.astype(float).tolist()
    assert np.array_equal(sequences[0], values[2:6])
    assert np.array_equal(sequences[-1], values[14:18])


def test_sequence_batch_rejects_nonpositive_stride():
    with pytest.raises(ValueError, match="sequence_stride must be positive"):
        _sequence_batch(
            np.arange(5), np.zeros((5, 1)), {"return": np.zeros(5)},
            pd.Series(pd.date_range("2025-01-01", periods=5, freq="5min", tz="UTC")),
            sequence_length=2, sequence_stride=0,
        )
