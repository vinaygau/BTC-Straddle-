import numpy as np
import pandas as pd

from scripts.evaluate_model_reliability import evaluate
from scripts.evaluate_walk_forward import evaluate_walk_forward_predictions
import pytest


def test_reliability_evaluation_reports_class_metrics_calibration_and_regimes():
    times = pd.date_range("2026-01-01", periods=40, freq="h", tz="UTC")
    actual = np.tile([0, 1, 2, 1], 10)
    probabilities = np.tile([.2, .6, .2], (40, 1))
    predictions = pd.DataFrame({
        "prediction_time_utc": times[20:],
        "actual_return": np.zeros(20),
        "actual_direction": actual[20:],
        "actual_extreme_move": np.tile([0, 0, 1, 0], 5),
        "hybrid_expected_return": np.zeros(20),
        "hybrid_return_q10": np.full(20, -.01),
        "hybrid_return_q90": np.full(20, .01),
        "bearish_probability": probabilities[20:, 0],
        "sideways_probability": probabilities[20:, 1],
        "bullish_probability": probabilities[20:, 2],
        "extreme_movement_probability": np.full(20, .1),
    })
    features = pd.DataFrame({
        "prediction_time_utc": times,
        "5m_realized_volatility_12": np.r_[np.full(20, .001), np.linspace(.0005, .002, 20)],
        "5m_momentum_return_12": np.r_[np.zeros(20), np.tile([-.003, 0, .003, 0], 5)],
    })

    report = evaluate(predictions, features=features)
    metrics = report["overall"]
    assert "direction_precision_recall_f1_by_class" in metrics
    assert "direction_brier_score" in metrics
    assert "direction_calibration_bins" in metrics
    assert "extreme_precision_at_0_5" in metrics
    assert "extreme_recall_at_0_5" in metrics
    assert "extreme_brier_score" in metrics
    assert "extreme_calibration_bins" in metrics
    assert metrics["zero_baseline_mae"] == 0
    assert report["by_market_regime"]


def test_walk_forward_evaluator_requires_predictions_after_train_cutoff():
    rows = pd.DataFrame({
        "walk_forward_fold": [1, 1],
        "train_end_utc": ["2026-01-01T00:00:00Z"] * 2,
        "prediction_time_utc": ["2026-01-02T00:00:00Z", "2026-01-02T01:00:00Z"],
    })
    with pytest.raises(ValueError, match="Held-out predictions lack columns"):
        evaluate_walk_forward_predictions(rows)
