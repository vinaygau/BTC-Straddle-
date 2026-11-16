"""Inspection and preprocessing orchestration for historical BTC candles."""
from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd

from .data_inspector import inspect_frame
from .data_loader import discover_datasets, infer_timeframe, load_csv
from .data_merger import merge_datasets
from .data_quality_report import write_quality_reports
from .data_validator import valid_candle_mask

LOGGER = logging.getLogger(__name__)
OUTPUTS = {"5m": "btc_5m.parquet", "15m": "btc_15m.parquet", "1h": "btc_1h.parquet", "daily": "btc_daily.parquet"}


def inspect_directory(raw_dir: str | Path) -> tuple[list[dict], dict[str, list[pd.DataFrame]]]:
    reports, grouped = [], {k: [] for k in OUTPUTS}
    for path, member in discover_datasets(raw_dir):
        timeframe = infer_timeframe(f"{path.parent.name}/{path.name}/{member or ''}")
        try:
            frame = load_csv(path, archive_member=member)
            item = inspect_frame(frame, file_name=f"{path.name}{'::' + member if member else ''}", timeframe=timeframe)
            item["source_market"] = frame.source_market.iloc[0] if len(frame) else "unknown"
            item["included_in_btcusdt_pipeline"] = item["source_market"] == "BTCUSDT" and timeframe in grouped
            if item["included_in_btcusdt_pipeline"]: grouped[timeframe].append(frame)
            reports.append(item)
        except Exception as exc:
            LOGGER.exception("Could not inspect %s", path)
            reports.append({"file_name": f"{path.name}{'::' + member if member else ''}", "timeframe": timeframe, "error": str(exc), "included_in_btcusdt_pipeline": False})
    return reports, grouped


def preprocess(raw_dir: str | Path, processed_dir: str | Path, reports_dir: str | Path) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    file_reports, grouped = inspect_directory(raw_dir)
    processed_dir = Path(processed_dir); processed_dir.mkdir(parents=True, exist_ok=True)
    combined = {}
    for timeframe, frames in grouped.items():
        merged = merge_datasets(frames)
        if not merged.empty:
            valid = valid_candle_mask(merged)
            dropped_invalid = int((~valid).sum())
            merged = merged.loc[valid].reset_index(drop=True)
            merged = merged.drop(columns=["source_priority"], errors="ignore")
            target = processed_dir / OUTPUTS[timeframe]
            merged.to_parquet(target, index=False)
            combined[timeframe] = merged
            file_reports.append(inspect_frame(merged, file_name=target.name, timeframe=timeframe) | {"source_market": "BTCUSDT", "included_in_btcusdt_pipeline": True, "dropped_invalid_candle_rows": dropped_invalid})
            LOGGER.info("Wrote %s (%d candles)", target, len(merged))
    write_quality_reports(file_reports, reports_dir, "dataset_quality")
    return file_reports, combined

