"""Build causal multi-timeframe features from processed candle Parquet files."""
import argparse
from pathlib import Path

from config.data_config import DataPipelineConfig
from data.feature_engineering import write_multitimeframe_features


def main() -> None:
    config = DataPipelineConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=config.processed_dir)
    parser.add_argument("--output", type=Path, default=config.processed_dir / "btc_multitimeframe_features.parquet")
    parser.add_argument("--base-timeframe", choices=("5m", "15m", "1h", "daily"), default="5m")
    args = parser.parse_args()
    path = write_multitimeframe_features(args.processed_dir, output_path=args.output, base_timeframe=args.base_timeframe)
    import pandas as pd
    result = pd.read_parquet(path)
    print(f"Wrote {len(result):,} aligned rows and {len(result.columns):,} columns to {path}")
    print(f"Prediction-time range: {result.prediction_time_utc.min()} through {result.prediction_time_utc.max()}")


if __name__ == "__main__":
    main()
