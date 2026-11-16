"""Download real Deribit BTC option trades to a raw CSV with a provenance manifest."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from config.data_config import DataPipelineConfig
from data.deribit_historical_options import fetch_btc_option_trades, parse_utc_date


def main() -> None:
    config = DataPipelineConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="UTC start date/time (inclusive), e.g. 2025-01-01")
    parser.add_argument("--end", required=True, help="UTC end date/time (inclusive); a date includes its full UTC day")
    parser.add_argument("--window-hours", type=int, default=24, help="Initial request window; dense windows are subdivided automatically")
    parser.add_argument("--output-dir", type=Path, default=config.raw_dir / "deribit_option_trades")
    parser.add_argument("--overwrite", action="store_true", help="Redownload chunks that already have a CSV and manifest")
    args = parser.parse_args()

    try:
        start = parse_utc_date(args.start)
        end = parse_utc_date(args.end, end_of_day=True)
        if start > end:
            raise ValueError("--start must be earlier than or equal to --end.")
        if args.window_hours <= 0 or args.window_hours > 24 * 31:
            raise ValueError("--window-hours must be between 1 and 744.")
    except ValueError as exc:
        parser.error(str(exc))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    day_cursor = start.replace(hour=0, minute=0, second=0, microsecond=0)
    chunk_number = 0
    while day_cursor <= end:
        chunk_start = max(start, day_cursor)
        chunk_end = min(end, day_cursor + timedelta(days=1, milliseconds=-1))
        start_tag = chunk_start.strftime("%Y%m%dT%H%M%SZ")
        end_tag = chunk_end.strftime("%Y%m%dT%H%M%SZ")
        csv_path = args.output_dir / f"btc_option_trades_{start_tag}_{end_tag}.csv"
        manifest_path = csv_path.with_suffix(".manifest.json")
        chunk_number += 1
        if csv_path.is_file() and manifest_path.is_file() and not args.overwrite:
            print(f"[{chunk_number}] Skipping completed chunk: {csv_path.name}", flush=True)
            day_cursor += timedelta(days=1)
            continue
        try:
            trades = fetch_btc_option_trades(chunk_start, chunk_end, window_hours=args.window_hours)
        except (RuntimeError, OSError) as exc:
            parser.error(f"{chunk_start.date()} download failed: {exc}")

        fields = sorted({key for trade in trades for key in trade})
        if fields:
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
                writer.writeheader()
                writer.writerows(trades)
        else:
            csv_path.write_text("", encoding="utf-8")
        manifest = {
            "source": "Deribit history API public/get_last_trades_by_currency_and_time",
            "api_root": "https://history.deribit.com/api/v2",
            "currency": "BTC",
            "kind": "option",
            "start_utc": chunk_start.isoformat().replace("+00:00", "Z"),
            "end_utc": chunk_end.isoformat().replace("+00:00", "Z"),
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "trade_rows": len(trades),
            "distinct_instruments": len({row["instrument_name"] for row in trades}),
            "csv_file": csv_path.name,
            "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            "data_semantics": "Executed public trades, not historical bid/ask order-book snapshots or guaranteed fill quotes.",
            "limitations": [
                "No historical best bid/ask or displayed size is included in this trade tape.",
                "A trade tape alone is insufficient to claim realistic limit-order fills or a quote-based backtest.",
                "Trade amount and price units must be interpreted using the instrument contract metadata.",
            ],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[{chunk_number}] {chunk_start.date()}: {len(trades)} trades, {manifest['distinct_instruments']} instruments", flush=True)
        day_cursor += timedelta(days=1)


if __name__ == "__main__":
    main()
