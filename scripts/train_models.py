"""Train offline LSTM and XGBoost research models from generated features."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

from config.data_config import DataPipelineConfig
from models.train import train_models
from models.training_config import ModelTrainingConfig


def main() -> None:
    config = DataPipelineConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=config.processed_dir / "btc_multitimeframe_features.parquet")
    parser.add_argument("--output-dir", type=Path,
                        help="Artifact destination. Defaults to a timestamped experiment directory, preserving promoted models.")
    parser.add_argument("--promote-artifacts", action="store_true",
                        help="Explicitly allow writing into the promoted model_artifacts directory.")
    parser.add_argument("--horizon-steps", type=int, default=12)
    parser.add_argument("--sequence-length", type=int, default=48)
    parser.add_argument("--sequence-stride", type=int, default=12,
                        help="Train/evaluate LSTM on one observation per N five-minute rows (default: 12, non-overlapping one-hour targets).")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--xgboost-estimators", type=int, default=300)
    parser.add_argument("--direction-threshold", type=float, default=0.0025)
    parser.add_argument("--extreme-threshold", type=float, default=0.01)
    parser.add_argument("--lstm-only", action="store_true",
                        help="Train only the multitask LSTM; do not produce or claim hybrid results.")
    args = parser.parse_args()
    output_dir = args.output_dir or (config.processed_dir / "model_artifacts_experiments" /
                                     datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    promoted_dir = (config.processed_dir / "model_artifacts").resolve()
    if output_dir.resolve() == promoted_dir and not args.promote_artifacts:
        parser.error("Refusing to overwrite promoted artifacts. Use a separate --output-dir; pass --promote-artifacts only for an intentional promotion.")
    training = ModelTrainingConfig(horizon_steps=args.horizon_steps, sequence_length=args.sequence_length,
                                   sequence_stride=args.sequence_stride,
                                   lstm_epochs=args.epochs, lstm_batch_size=args.batch_size,
                                   xgboost_estimators=args.xgboost_estimators,
                                   direction_threshold=args.direction_threshold,
                                   extreme_move_threshold=args.extreme_threshold)
    report = train_models(args.features, output_dir, training, include_xgboost=not args.lstm_only)
    metric_name = "hybrid_return" if not args.lstm_only else "lstm_return"
    print(f"Offline {report['model_mode']} training complete. Held-out return metrics: {report['test_metrics'][metric_name]}")
    print(f"Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
