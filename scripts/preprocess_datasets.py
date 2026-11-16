"""Merge and export separate BTCUSDT Parquet candle datasets."""
import argparse
from pathlib import Path
from config.data_config import DataPipelineConfig
from data.data_preprocessor import preprocess


def main():
    config = DataPipelineConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=config.raw_dir)
    parser.add_argument("--processed-dir", type=Path, default=config.processed_dir)
    parser.add_argument("--reports-dir", type=Path, default=config.reports_dir)
    args = parser.parse_args()
    _, datasets = preprocess(args.raw_dir, args.processed_dir, args.reports_dir)
    for timeframe, frame in datasets.items(): print(f"{timeframe}: {len(frame):,} rows -> {args.processed_dir / {'5m':'btc_5m.parquet','15m':'btc_15m.parquet','1h':'btc_1h.parquet','daily':'btc_daily.parquet'}[timeframe]}")


if __name__ == "__main__": main()
