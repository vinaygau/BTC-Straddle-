from datetime import datetime, timezone
import hashlib
from pathlib import Path
import zipfile

import pandas as pd
import pytest

from data.data_loader import load_csv
from data.data_merger import merge_datasets
from data.data_validator import missing_intervals, validate_candles
from data.data_preprocessor import preprocess
from data.candle_alignment import completed_candles


def test_csv_headerless_kline_mapping_and_microseconds(tmp_path):
    row = "1794528000000000,100,105,99,103,4,1794528299999999,410,20,2,205,0\n"
    path = tmp_path / "BTCUSDT-5m.csv"; path.write_text(row)
    frame = load_csv(path)
    assert frame.loc[0, "open"] == 100
    assert frame.loc[0, "trade_count"] == 20
    assert isinstance(frame.timestamp_utc.dtype, pd.DatetimeTZDtype)
    assert str(frame.timestamp_utc.dt.tz) == "UTC"


def test_csv_preamble_and_zip(tmp_path):
    payload = "https://example.test/data\nUnix,Date,Symbol,Open,High,Low,Close,Volume BTC,Volume USDT,tradecount\n1794528000000,2026-11-13,BTCUSDT,100,105,99,103,4,410,20\n"
    path = tmp_path / "archive.zip"
    with zipfile.ZipFile(path, "w") as zf: zf.writestr("bars.csv", payload)
    frame = load_csv(path, archive_member="bars.csv")
    assert frame.loc[0, "symbol"] == "BTCUSDT"
    assert frame.loc[0, "quote_volume"] == 410


def test_mixed_millisecond_and_microsecond_timestamp_units(tmp_path):
    payload = "Unix,Open,High,Low,Close,Volume\n1502956800000,10,11,9,10,1\n1741734000000000,11,12,10,11,1\n"
    path = tmp_path / "mixed.csv"; path.write_text(payload)
    frame = load_csv(path)
    assert frame.timestamp_utc.dt.year.tolist() == [2017, 2025]


def test_duplicates_gap_and_ohlcv_validation():
    frame = pd.DataFrame({"timestamp_utc": pd.to_datetime(["2025-01-01T00:00Z", "2025-01-01T00:10Z"], utc=True),
                          "open": [10, 10], "high": [11, 12], "low": [9, 9], "close": [10, 11], "volume": [1, -1]})
    assert missing_intervals(frame, "5m")[0]["missing_candles"] == 1
    assert validate_candles(frame)["invalid_volume_rows"] == 1
    assert frame.timestamp_utc.duplicated().sum() == 0


def test_merge_priority_and_parquet_preserves_raw(tmp_path):
    raw = tmp_path / "raw"; source_dir = raw / "BTCUSDT-5m-2025-01"; source_dir.mkdir(parents=True)
    a = source_dir / "bars.csv"; a.write_text("1735689600000,10,11,9,10,1,1735689899999,10,1,0.5,5,0\n")
    before = hashlib.sha256(a.read_bytes()).hexdigest()
    reports, result = preprocess(raw, tmp_path / "processed", tmp_path / "reports")
    assert result["5m"].iloc[0].close == 10
    assert (tmp_path / "processed" / "btc_5m.parquet").exists()
    assert hashlib.sha256(a.read_bytes()).hexdigest() == before
    assert (tmp_path / "reports" / "dataset_quality.json").exists()


def test_merge_uses_first_priority_for_duplicate_timestamp():
    stamp = pd.Timestamp("2025-01-01", tz="UTC")
    frames = [pd.DataFrame({"timestamp_utc": [stamp], "close": [1], "source_priority": [2], "source_file": ["later"]}),
              pd.DataFrame({"timestamp_utc": [stamp], "close": [2], "source_priority": [0], "source_file": ["preferred"]})]
    assert merge_datasets(frames).iloc[0].close == 2


def test_alignment_excludes_candle_not_completed_at_prediction_time():
    frame = pd.DataFrame({"timestamp_utc": pd.to_datetime(["2025-01-01T00:00Z", "2025-01-01T00:05Z"], utc=True)})
    aligned = completed_candles(frame, "2025-01-01T00:07:00Z", timeframe="5m")
    assert len(aligned) == 1

