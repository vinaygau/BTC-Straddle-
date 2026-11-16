"""Download reproducible public BTCUSDT candle history for model refreshes."""
from __future__ import annotations

import argparse

from config.data_config import DataPipelineConfig
from data.binance_history import download_btcusdt_history, save_btcusdt_history


def main() -> None:
    config = DataPipelineConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="First UTC date, YYYY-MM-DD, inclusive")
    parser.add_argument("--end", required=True, help="Last UTC date, YYYY-MM-DD, inclusive")
    parser.add_argument("--timeframes", nargs="+", choices=("5m", "15m", "1h", "daily"),
                        default=("5m", "15m", "1h", "daily"))
    parser.add_argument("--output-dir", default=str(config.raw_dir / "binance_history"))
    args = parser.parse_args()

    for timeframe in args.timeframes:
        frame = download_btcusdt_history(
            timeframe,
            args.start,
            args.end,
            on_page=lambda pages, rows: print(
                f"{timeframe}: received page {pages}, {rows:,} candles so far", flush=True
            ),
        )
        output = save_btcusdt_history(frame, timeframe, args.start, args.end, args.output_dir)
        print(f"{timeframe}: saved {len(frame):,} candles to {output}")


if __name__ == "__main__":
    main()
