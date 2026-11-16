"""Generate/evaluate chronological walk-forward forecasts without promotion.

If --predictions is given, evaluate supplied out-of-fold predictions. If a
features file is given instead, train the existing hybrid architecture on
expanding historical folds and save every fold under a separate experiment
directory, never into the promoted model_artifacts directory.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from scripts.evaluate_model_reliability import evaluate


def evaluate_walk_forward_predictions(predictions: pd.DataFrame, *, features: pd.DataFrame | None = None):
    required = {"walk_forward_fold", "train_end_utc", "prediction_time_utc"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Walk-forward predictions require columns: {sorted(missing)}")
    frame = predictions.copy()
    frame["prediction_time_utc"] = pd.to_datetime(frame.prediction_time_utc, utc=True, errors="coerce")
    frame["train_end_utc"] = pd.to_datetime(frame.train_end_utc, utc=True, errors="coerce")
    if frame[["prediction_time_utc", "train_end_utc"]].isna().any().any():
        raise ValueError("Walk-forward timestamps must be valid and timezone-aware")
    if not (frame.prediction_time_utc > frame.train_end_utc).all():
        raise ValueError("Every walk-forward prediction must occur strictly after its recorded training cutoff")
    folds = []
    for fold_id, rows in frame.groupby("walk_forward_fold", sort=True):
        if rows.prediction_time_utc.duplicated().any():
            raise ValueError(f"Duplicate prediction timestamps in walk-forward fold {fold_id}")
        fold_report = evaluate(rows.drop(columns=["walk_forward_fold", "train_end_utc"]), features=features)
        folds.append({
            "fold": str(fold_id),
            "train_end_utc": rows.train_end_utc.max().isoformat(),
            "first_prediction_utc": rows.prediction_time_utc.min().isoformat(),
            "last_prediction_utc": rows.prediction_time_utc.max().isoformat(),
            "metrics": fold_report["overall"],
            "by_market_regime": fold_report["by_market_regime"],
        })
    if not folds:
        raise ValueError("No walk-forward folds were supplied")
    aggregate = evaluate(
        frame.drop(columns=["walk_forward_fold", "train_end_utc"]),
        features=features,
    )
    return {
        "scope": "walk-forward diagnostics on supplied out-of-fold predictions",
        "model_retrained_by_this_command": False,
        "fold_count": len(folds),
        "total_prediction_rows": int(len(frame)),
        "aggregate_metrics": aggregate["overall"],
        "aggregate_period_utc": {
            "first_prediction": frame.prediction_time_utc.min().isoformat(),
            "last_prediction": frame.prediction_time_utc.max().isoformat(),
        },
        "folds": folds,
        "limitations": [
            "This evaluator checks timestamps and metrics but cannot independently verify that each fold's model and scaler were fit only on the recorded training period.",
            "The project's current held-out prediction artifact is not a walk-forward prediction file.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, help="Optional Parquet out-of-fold prediction file")
    parser.add_argument("--features", type=Path, help="Aligned feature input for walk-forward model training")
    parser.add_argument("--initial-train-rows", type=int, default=105120,
                        help="Initial training window in 5m rows (default: 365 days)")
    parser.add_argument("--validation-rows", type=int, default=25920,
                        help="Validation window in 5m rows (default: 90 days)")
    parser.add_argument("--test-rows", type=int, default=8640,
                        help="Test window in 5m rows (default: 30 days)")
    parser.add_argument("--step-rows", type=int, default=8640,
                        help="Advance the fold by this many rows (default: non-overlapping 30 days)")
    parser.add_argument("--purge-rows", type=int, default=12)
    parser.add_argument("--max-folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sequence-stride", type=int, default=12)
    parser.add_argument("--output", type=Path,
                        default=Path("data/reports/walk_forward_runs") /
                        f"{pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%SZ')}" / "walk_forward_reliability.json")
    args = parser.parse_args()
    if args.max_folds <= 0:
        parser.error("--max-folds must be positive")
    if args.predictions:
        predictions = pd.read_parquet(args.predictions)
        feature_data = pd.read_parquet(args.features) if args.features else None
        report = evaluate_walk_forward_predictions(predictions, features=feature_data)
        report["prediction_file"] = str(args.predictions)
    elif args.features:
        from config.data_config import DataPipelineConfig
        from models.dataset import prepare_training_frame
        from models.train import train_models
        from models.training_config import ModelTrainingConfig
        from backtesting.walk_forward import expanding_walk_forward_folds

        feature_data = pd.read_parquet(args.features)
        base_config = ModelTrainingConfig(sequence_stride=args.sequence_stride,
                                          lstm_epochs=args.epochs, lstm_batch_size=args.batch_size)
        labeled, _ = prepare_training_frame(feature_data, base_config)
        folds = expanding_walk_forward_folds(
            labeled.prediction_time_utc,
            initial_train_size=args.initial_train_rows,
            validation_size=args.validation_rows,
            test_size=args.test_rows,
            step_size=args.step_rows,
            purge_rows=args.purge_rows,
        )[-args.max_folds:]
        fold_rows = []
        for fold in folds:
            last_test = pd.Timestamp(fold.test_end_utc)
            input_end = last_test + pd.Timedelta(minutes=5 * base_config.horizon_steps)
            fold_features = feature_data.loc[
                pd.to_datetime(feature_data.prediction_time_utc, utc=True) <= input_end
            ].copy()
            fold_labeled, _ = prepare_training_frame(fold_features, base_config)
            expected_rows = int(fold.test_indices[-1]) + 1
            if len(fold_labeled) != expected_rows:
                raise ValueError(f"Walk-forward fold {fold.fold} data cut produced {len(fold_labeled)} labels; expected {expected_rows}.")
            n = len(fold_labeled)
            train_boundary = int(fold.train_indices[-1]) + 1 + args.purge_rows
            validation_boundary = int(fold.validation_indices[-1]) + 1 + args.purge_rows
            fold_config = replace(
                base_config,
                train_boundary_index=train_boundary,
                validation_boundary_index=validation_boundary,
            )
            fold_dir = args.output.parent / "walk_forward_artifacts" / f"fold_{fold.fold:02d}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            fold_input = fold_dir / "features.parquet"
            fold_features.to_parquet(fold_input, index=False)
            report_one = train_models(fold_input, fold_dir / "artifacts", fold_config, include_xgboost=True)
            report_path = fold_dir / "artifacts" / "training_report.json"
            persisted_report = json.loads(report_path.read_text(encoding="utf-8"))
            persisted_report["walk_forward_fold"] = fold.fold
            persisted_report["walk_forward_train_end_utc"] = fold.train_end_utc
            persisted_report["walk_forward_validation_window_utc"] = [fold.validation_start_utc, fold.validation_end_utc]
            persisted_report["walk_forward_test_window_utc"] = [fold.test_start_utc, fold.test_end_utc]
            report_path.write_text(json.dumps(persisted_report, indent=2), encoding="utf-8")
            predictions_one = pd.read_parquet(fold_dir / "artifacts" / "heldout_test_predictions.parquet")
            predictions_one["walk_forward_fold"] = fold.fold
            predictions_one["train_end_utc"] = fold.train_end_utc
            predictions_one.to_parquet(fold_dir / "predictions.parquet", index=False)
            fold_rows.append({
                "fold": fold.fold,
                "train_start_utc": fold.train_start_utc,
                "train_end_utc": fold.train_end_utc,
                "validation_start_utc": fold.validation_start_utc,
                "validation_end_utc": fold.validation_end_utc,
                "test_start_utc": fold.test_start_utc,
                "test_end_utc": fold.test_end_utc,
                "prediction_rows": len(predictions_one),
                "training_report": report_one,
                "predictions_file": str(fold_dir / "predictions.parquet"),
                "artifacts_dir": str(fold_dir / "artifacts"),
            })
        out_of_fold = pd.concat([pd.read_parquet(row["predictions_file"]) for row in fold_rows], ignore_index=True)
        aggregate_predictions = args.output.parent / "walk_forward_predictions.parquet"
        out_of_fold.to_parquet(aggregate_predictions, index=False)
        report = evaluate_walk_forward_predictions(out_of_fold, features=feature_data)
        report["fold_schedule"] = [{k: v for k, v in row.items() if k not in {"training_report", "predictions_file"}}
                                   for row in fold_rows]
        report["fold_artifacts"] = [row["artifacts_dir"] for row in fold_rows]
        report["prediction_file"] = str(aggregate_predictions)
        report["experiment_artifacts_root"] = str(args.output.parent / "walk_forward_artifacts")
        report["all_fold_models_use_existing_lstm_xgboost_architecture"] = True
        report["model_retrained_by_this_command"] = True
    else:
        parser.error("Provide --features to train walk-forward folds or --predictions to evaluate out-of-fold data.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
