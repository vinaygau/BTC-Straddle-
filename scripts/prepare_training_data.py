"""Create labeled, chronological research samples without fitting models."""
import argparse
import json
from pathlib import Path

import pandas as pd

from config.data_config import DataPipelineConfig
from models.dataset import chronological_splits, prepare_training_frame
from models.training_config import ModelTrainingConfig


def main() -> None:
    config = DataPipelineConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=config.processed_dir / "btc_multitimeframe_features.parquet")
    parser.add_argument("--output-dir", type=Path, default=config.processed_dir)
    parser.add_argument("--horizon-steps", type=int, default=12)
    parser.add_argument("--direction-threshold", type=float, default=0.0025)
    parser.add_argument("--extreme-threshold", type=float, default=0.01)
    args = parser.parse_args()
    training = ModelTrainingConfig(horizon_steps=args.horizon_steps,
                                   direction_threshold=args.direction_threshold,
                                   extreme_move_threshold=args.extreme_threshold)
    data = pd.read_parquet(args.features)
    labeled, columns = prepare_training_frame(data, training)
    splits = chronological_splits(labeled, columns, training)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "btc_model_training.parquet"
    labeled.to_parquet(target, index=False)
    manifest = {
        "source_features": str(args.features), "training_data": str(target),
        "rows": len(labeled), "feature_count": len(columns), "feature_columns": list(columns),
        "horizon_steps": training.horizon_steps, "horizon_minutes": training.horizon_steps * 5,
        "direction_threshold": training.direction_threshold,
        "extreme_move_threshold": training.extreme_move_threshold,
        "direction_labels": {"0": "bearish", "1": "sideways", "2": "bullish"},
        "class_counts": {str(k): int(v) for k, v in labeled.target_direction.value_counts().sort_index().items()},
        "extreme_counts": {str(k): int(v) for k, v in labeled.target_extreme_move.value_counts().sort_index().items()},
        "splits": {
            "train": {"rows": len(splits.train), "first": labeled.prediction_time_utc.iloc[splits.train[0]].isoformat(), "last": labeled.prediction_time_utc.iloc[splits.train[-1]].isoformat()},
            "validation": {"rows": len(splits.validation), "first": labeled.prediction_time_utc.iloc[splits.validation[0]].isoformat(), "last": labeled.prediction_time_utc.iloc[splits.validation[-1]].isoformat()},
            "test": {"rows": len(splits.test), "first": labeled.prediction_time_utc.iloc[splits.test[0]].isoformat(), "last": labeled.prediction_time_utc.iloc[splits.test[-1]].isoformat()},
            "purge_gap_rows": training.horizon_steps,
        },
    }
    (args.output_dir / "btc_model_training_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared {len(labeled):,} labeled rows with {len(columns)} features: {target}")
    print(f"Time splits: train={len(splits.train):,}, validation={len(splits.validation):,}, test={len(splits.test):,}")


if __name__ == "__main__":
    main()
