"""Validate downloaded daily trade files and build a processed Parquet dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from config.data_config import DataPipelineConfig


REQUIRED_COLUMNS = {"timestamp", "trade_id", "instrument_name", "price", "amount"}


def main() -> None:
    config = DataPipelineConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=config.raw_dir / "deribit_option_trades")
    parser.add_argument("--output", type=Path, default=config.processed_dir / "btc_option_trades.parquet")
    parser.add_argument("--report", type=Path, default=config.reports_dir / "deribit_option_trade_coverage.json")
    args = parser.parse_args()

    manifests = sorted(args.input_dir.glob("*.manifest.json"))
    if not manifests:
        parser.error(f"No trade provenance manifests found in {args.input_dir}")

    frames: list[pd.DataFrame] = []
    coverage = []
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        csv_path = args.input_dir / manifest["csv_file"]
        if not csv_path.is_file():
            parser.error(f"Manifest references a missing CSV: {csv_path}")
        digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        if digest != manifest.get("csv_sha256"):
            parser.error(f"SHA-256 mismatch for {csv_path}")
        frame = pd.read_csv(csv_path, low_memory=False)
        missing = REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            parser.error(f"{csv_path.name} is missing required columns: {sorted(missing)}")
        if len(frame) != int(manifest["trade_rows"]):
            parser.error(f"Trade row count does not match manifest for {csv_path.name}")
        frame["timestamp_utc"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True, errors="coerce")
        if frame["timestamp_utc"].isna().any():
            parser.error(f"Invalid trade timestamps in {csv_path.name}")
        if (frame["price"] <= 0).any() or (frame["amount"] <= 0).any():
            parser.error(f"Nonpositive trade price or amount in {csv_path.name}")
        if not frame["timestamp_utc"].between(
            pd.Timestamp(manifest["start_utc"]), pd.Timestamp(manifest["end_utc"]), inclusive="both"
        ).all():
            parser.error(f"Trade timestamps fall outside manifest bounds in {csv_path.name}")
        coverage.append({
            "start": pd.Timestamp(manifest["start_utc"]),
            "end": pd.Timestamp(manifest["end_utc"]),
            "manifest": manifest_path.name,
            "csv": csv_path.name,
            "rows": len(frame),
        })
        frames.append(frame)

    coverage.sort(key=lambda item: item["start"])
    coverage_gaps = []
    for previous, current in zip(coverage, coverage[1:]):
        if current["start"] <= previous["end"]:
            parser.error(f"Overlapping raw trade chunks: {previous['csv']} and {current['csv']}")
        if current["start"] > previous["end"] + pd.Timedelta(milliseconds=1):
            coverage_gaps.append({
                "after_utc": previous["end"].isoformat().replace("+00:00", "Z"),
                "before_utc": current["start"].isoformat().replace("+00:00", "Z"),
            })
    combined = pd.concat(frames, ignore_index=True)
    duplicate_mask = combined.duplicated(["instrument_name", "trade_id"], keep=False)
    if duplicate_mask.any():
        parser.error(f"Found {int(duplicate_mask.sum())} duplicate instrument/trade IDs across raw chunks")
    combined = combined.sort_values(["timestamp_utc", "instrument_name", "trade_seq"], kind="stable").reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(args.output, index=False)
    covered_dates = {item["start"].date() for item in coverage}
    all_dates = set(pd.date_range(coverage[0]["start"].date(), coverage[-1]["end"].date(), freq="D").date)
    report = {
        "source": "Deribit public historical BTC option trade tape",
        "input_chunks": len(coverage),
        "covered_start_utc": coverage[0]["start"].isoformat().replace("+00:00", "Z"),
        "covered_end_utc": coverage[-1]["end"].isoformat().replace("+00:00", "Z"),
        "covered_utc_dates": sorted(date.isoformat() for date in covered_dates),
        "missing_utc_dates_between_chunks": sorted(date.isoformat() for date in all_dates - covered_dates),
        "coverage_gaps_between_chunks": coverage_gaps,
        "coverage_contiguous": not coverage_gaps,
        "trade_rows": len(combined),
        "distinct_instruments": int(combined["instrument_name"].nunique()),
        "duplicate_instrument_trade_ids": 0,
        "timestamp_min_utc": combined["timestamp_utc"].min().isoformat().replace("+00:00", "Z"),
        "timestamp_max_utc": combined["timestamp_utc"].max().isoformat().replace("+00:00", "Z"),
        "field_completeness": {name: float(combined[name].notna().mean()) for name in
                               ("price", "amount", "mark_price", "index_price", "iv", "contracts") if name in combined},
        "parquet_file": str(args.output),
        "execution_data_limit": "Trade prints are not historical bid/ask quotes, book depth, or guaranteed fills.",
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"trades": len(combined), "instruments": report["distinct_instruments"],
                      "parquet": str(args.output), "report": str(args.report)}, indent=2))


if __name__ == "__main__":
    main()
