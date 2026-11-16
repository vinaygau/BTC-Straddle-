"""Write day-by-day held-out diagnostics for the hybrid BTC research model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config.data_config import DataPipelineConfig


def _metrics(frame: pd.DataFrame) -> dict[str, float | int | None]:
    actual = frame["actual_return"].to_numpy(dtype=float)
    predicted = frame["hybrid_expected_return"].to_numpy(dtype=float)
    baseline = np.zeros_like(actual)
    direction_columns = ["bearish_probability", "sideways_probability", "bullish_probability"]
    probabilities = np.clip(frame[direction_columns].to_numpy(dtype=float), 1e-12, 1.0)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    direction = frame["actual_direction"].to_numpy(dtype=int)
    direction_predicted = probabilities.argmax(axis=1)
    direction_confusion = np.zeros((3, 3), dtype=int)
    for actual_class in range(3):
        for predicted_class in range(3):
            direction_confusion[actual_class, predicted_class] = int(
                np.sum((direction == actual_class) & (direction_predicted == predicted_class))
            )
    direction_by_class = {}
    for label, name in enumerate(("bearish", "sideways", "bullish")):
        tp = direction_confusion[label, label]
        precision = tp / max(1, direction_confusion[:, label].sum())
        recall = tp / max(1, direction_confusion[label, :].sum())
        direction_by_class[name] = {
            "precision": float(precision), "recall": float(recall),
            "f1": float(2 * precision * recall / max(precision + recall, 1e-12)),
            "support": int(direction_confusion[label, :].sum()),
        }
    one_hot = np.eye(3)[direction]
    direction_brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    direction_confidence = probabilities.max(axis=1)
    direction_correct = direction_predicted == direction
    direction_calibration = []
    for lower, upper in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (direction_confidence >= lower) & (direction_confidence < upper if upper < 1 else direction_confidence <= upper)
        if mask.any():
            direction_calibration.append({"lower": float(lower), "upper": float(upper), "count": int(mask.sum()),
                                          "mean_confidence": float(direction_confidence[mask].mean()),
                                          "accuracy": float(direction_correct[mask].mean())})
    direction_ece = float(sum(item["count"] / len(direction) * abs(item["mean_confidence"] - item["accuracy"])
                              for item in direction_calibration))
    extreme_actual = frame["actual_extreme_move"].to_numpy(dtype=int)
    extreme_probability = np.clip(frame["extreme_movement_probability"].to_numpy(dtype=float), 1e-12, 1 - 1e-12)
    q10 = frame["hybrid_return_q10"].to_numpy(dtype=float)
    q90 = frame["hybrid_return_q90"].to_numpy(dtype=float)
    extreme_predicted = extreme_probability >= .5
    extreme_tp = int(np.sum(extreme_predicted & (extreme_actual == 1)))
    extreme_precision = extreme_tp / max(1, int(extreme_predicted.sum()))
    extreme_recall = extreme_tp / max(1, int((extreme_actual == 1).sum()))
    extreme_calibration = []
    for lower, upper in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (extreme_probability >= lower) & (extreme_probability < upper if upper < 1 else extreme_probability <= upper)
        if mask.any():
            extreme_calibration.append({"lower": float(lower), "upper": float(upper), "count": int(mask.sum()),
                                        "mean_probability": float(extreme_probability[mask].mean()),
                                        "observed_rate": float(extreme_actual[mask].mean())})
    extreme_ece = float(sum(item["count"] / len(extreme_actual) * abs(item["mean_probability"] - item["observed_rate"])
                           for item in extreme_calibration))
    return {
        "rows": int(len(frame)),
        "return_mae": float(np.mean(np.abs(actual - predicted))),
        "zero_baseline_mae": float(np.mean(np.abs(actual - baseline))),
        "return_rmse": float(np.sqrt(np.mean((actual - predicted) ** 2))),
        "zero_baseline_rmse": float(np.sqrt(np.mean(actual ** 2))),
        "direction_accuracy": float(np.mean(direction_predicted == direction)),
        "direction_precision_recall_f1_by_class": direction_by_class,
        "direction_brier_score": direction_brier,
        "direction_expected_calibration_error": direction_ece,
        "direction_calibration_bins": direction_calibration,
        "direction_majority_baseline_accuracy": float(max(np.mean(direction == label) for label in (0, 1, 2))),
        "direction_log_loss": float(-np.log(probabilities[np.arange(len(direction)), direction]).mean()),
        "extreme_brier_score": float(np.mean((extreme_probability - extreme_actual) ** 2)),
        "extreme_precision_at_0_5": float(extreme_precision),
        "extreme_recall_at_0_5": float(extreme_recall),
        "extreme_f1_at_0_5": float(2 * extreme_precision * extreme_recall / max(extreme_precision + extreme_recall, 1e-12)),
        "extreme_expected_calibration_error": extreme_ece,
        "extreme_calibration_bins": extreme_calibration,
        "extreme_prevalence_baseline_brier": float(np.mean(extreme_actual) * (1.0 - np.mean(extreme_actual))),
        "extreme_log_loss": float(-np.mean(extreme_actual * np.log(extreme_probability) + (1 - extreme_actual) * np.log(1 - extreme_probability))),
        "q10_q90_interval_coverage": float(np.mean((actual >= q10) & (actual <= q90))),
        "extreme_rate": float(np.mean(extreme_actual)),
    }


def evaluate(predictions: pd.DataFrame, *, features: pd.DataFrame | None = None) -> dict[str, object]:
    required = {
        "prediction_time_utc", "actual_return", "actual_direction", "actual_extreme_move",
        "hybrid_expected_return", "hybrid_return_q10", "hybrid_return_q90",
        "bearish_probability", "sideways_probability", "bullish_probability",
        "extreme_movement_probability",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Held-out predictions lack columns: {sorted(missing)}")
    optional_regime = {"market_regime"}
    frame = predictions.loc[:, sorted(required | (optional_regime & set(predictions.columns)))].copy()
    frame["prediction_time_utc"] = pd.to_datetime(frame["prediction_time_utc"], utc=True)
    numeric = sorted(required - {"prediction_time_utc"})
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=numeric).sort_values("prediction_time_utc", kind="stable")
    if frame.empty:
        raise ValueError("No complete held-out prediction rows remain.")
    frame["utc_date"] = frame["prediction_time_utc"].dt.strftime("%Y-%m-%d")
    by_day = [{"utc_date": day, **_metrics(group)} for day, group in frame.groupby("utc_date", sort=True)]
    by_regime = []
    if "market_regime" not in frame and features is not None:
        regime_columns = {"prediction_time_utc", "5m_realized_volatility_12", "5m_momentum_return_12"}
        if regime_columns.issubset(features.columns):
            source = features.loc[:, sorted(regime_columns)].copy()
            source["prediction_time_utc"] = pd.to_datetime(source.prediction_time_utc, utc=True, errors="coerce")
            source["5m_realized_volatility_12"] = pd.to_numeric(source["5m_realized_volatility_12"], errors="coerce")
            source["5m_momentum_return_12"] = pd.to_numeric(source["5m_momentum_return_12"], errors="coerce")
            train_period = source.loc[source.prediction_time_utc < frame.prediction_time_utc.min(), "5m_realized_volatility_12"].dropna()
            if len(train_period) >= 30:
                low, high = train_period.quantile([1/3, 2/3]).tolist()
                aligned = frame[["prediction_time_utc"]].merge(source, on="prediction_time_utc", how="left", validate="one_to_one")
                def label_regime(row):
                    rv, momentum = row["5m_realized_volatility_12"], row["5m_momentum_return_12"]
                    if pd.isna(rv) or pd.isna(momentum): return "unknown"
                    if momentum >= .0025: return "bullish_trend"
                    if momentum <= -.0025: return "bearish_trend"
                    return "high_volatility_sideways" if rv >= high else "low_volatility_sideways"
                frame["market_regime"] = aligned.apply(label_regime, axis=1)
            else:
                frame["market_regime"] = "unknown"
        else:
            frame["market_regime"] = "unknown"
    if "market_regime" in frame:
        by_regime = [{"market_regime": str(regime), **_metrics(group)}
                     for regime, group in frame.groupby("market_regime", sort=True)]
    return {
        "scope": "chronological held-out diagnostics; not a live signal validation",
        "rows": len(frame),
        "first_prediction_utc": frame["prediction_time_utc"].min().isoformat(),
        "last_prediction_utc": frame["prediction_time_utc"].max().isoformat(),
        "overall": _metrics(frame),
        "by_utc_day": by_day,
        "by_market_regime": by_regime,
        "interpretation": [
            "Compare return errors with the zero-return forecast; lower is better.",
            "Compare direction accuracy with the majority-class baseline and log loss with the training-prior baseline.",
            "The LSTM/XGBoost predictions are sampled at the one-hour target stride; daily rows group those observations and are not separate walk-forward folds.",
            "Quantile coverage is evaluated against the test labels and must not be used to retune these same held-out predictions.",
        ],
    }


def main() -> None:
    config = DataPipelineConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=config.processed_dir / "model_artifacts" / "heldout_test_predictions.parquet")
    parser.add_argument("--output", type=Path, default=config.reports_dir / "model_reliability.json")
    parser.add_argument("--features", type=Path, default=config.processed_dir / "btc_multitimeframe_features.parquet",
                        help="Optional aligned features used for test-period regime breakdown.")
    args = parser.parse_args()
    if not args.predictions.is_file():
        parser.error(f"Held-out prediction file does not exist: {args.predictions}")
    feature_frame = pd.read_parquet(args.features) if args.features.is_file() else None
    report = evaluate(pd.read_parquet(args.predictions), features=feature_frame)
    report["prediction_file"] = str(args.predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
