"""Run one offline hybrid prediction from prepared BTC candle Parquet files."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from config.data_config import DataPipelineConfig, SUPPORTED_TIMEFRAMES
from models.hybrid_inference import HybridMarketPredictionProvider


def main() -> None:
    config = DataPipelineConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=config.processed_dir)
    parser.add_argument("--artifact-dir", type=Path, default=config.processed_dir / "model_artifacts")
    parser.add_argument(
        "--as-of",
        help="UTC timestamp of a completed 5m candle close; defaults to the latest prepared 5m close.",
    )
    args = parser.parse_args()

    candles: dict[str, pd.DataFrame] = {}
    for timeframe in SUPPORTED_TIMEFRAMES:
        path = args.data_dir / f"btc_{timeframe}.parquet"
        if not path.is_file():
            parser.error(f"Missing prepared candle data: {path}")
        candles[timeframe] = pd.read_parquet(path)

    if args.as_of:
        as_of = args.as_of
    else:
        latest_close = candles["5m"].get("close_time_utc")
        if latest_close is None or candles["5m"].empty:
            parser.error("5m candle data must include close_time_utc to infer the latest as-of time.")
        as_of = pd.to_datetime(latest_close, utc=True).max().isoformat()

    provider = HybridMarketPredictionProvider(args.artifact_dir)
    prediction = provider.predict(candles, as_of_utc=as_of)
    print(json.dumps(asdict(prediction), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
