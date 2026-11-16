"""Load downloaded candle CSV/ZIP files into a normalized UTC schema."""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import pandas as pd

BINANCE_KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume", "ignore",
]
ALIASES = {
    "open_time": "timestamp", "open_time_ms": "timestamp", "open_time_us": "timestamp",
    "unix": "timestamp", "timestamp": "timestamp", "time": "timestamp",
    "date": "source_date", "open": "open", "high": "high", "low": "low",
    "close": "close", "volume": "volume", "volume btc": "volume",
    "quote_asset_volume": "quote_volume", "volume usdt": "quote_volume",
    "volume usd": "quote_volume", "quote volume": "quote_volume",
    "close_time": "close_time", "close_time_ms": "close_time",
    "number_of_trades": "trade_count", "tradecount": "trade_count",
    "trades": "trade_count", "taker_buy_base_asset_volume": "taker_buy_base_volume",
    "taker_buy_quote_asset_volume": "taker_buy_quote_volume", "symbol": "symbol",
    "ignore": "ignore",
}


def _parse_epoch_series(values: pd.Series) -> pd.Series:
    """Parse epoch fields row-by-row because some exports mix ms and us."""
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns, UTC]")
    magnitude = numeric.abs()
    for mask, unit in ((magnitude >= 1e17, "ns"), (magnitude >= 1e14, "us"),
                       (magnitude >= 1e11, "ms"), (numeric.notna(), "s")):
        selected = mask & result.isna()
        if selected.any():
            result.loc[selected] = pd.to_datetime(numeric.loc[selected], unit=unit, utc=True, errors="coerce")
    return result


def infer_timeframe(path: str | Path) -> str | None:
    text = str(path).lower()
    if re.search(r"(^|[^a-z0-9])5m([^a-z0-9]|$)", text): return "5m"
    if re.search(r"(^|[^a-z0-9])15m([^a-z0-9]|$)", text): return "15m"
    if re.search(r"(^|[^a-z0-9])1h([^a-z0-9]|$)", text): return "1h"
    if "daily" in text or re.search(r"(?:_|-)d(?:\.|_|-)", text): return "daily"
    return None


def _read_csv_bytes(blob: bytes) -> pd.DataFrame:
    text = blob.decode("utf-8-sig", errors="replace")
    lines = text.splitlines()
    while lines and (not lines[0].strip() or lines[0].strip().lower().startswith(("http://", "https://"))):
        lines.pop(0)
    if not lines: return pd.DataFrame()
    first = next(csv for csv in [lines[0]] if csv)
    tokens = [part.strip().strip('"').lower() for part in first.split(",")]
    header_words = {"unix", "date", "open_time", "open_time_ms", "timestamp", "symbol", "open"}
    header = bool(set(tokens) & header_words) and any(x in tokens for x in ("open", "timestamp", "unix"))
    if header:
        frame = pd.read_csv(io.StringIO("\n".join(lines)))
    else:
        frame = pd.read_csv(io.StringIO("\n".join(lines)), header=None)
        if frame.shape[1] == len(BINANCE_KLINE_COLUMNS): frame.columns = BINANCE_KLINE_COLUMNS
        else: frame.columns = [f"column_{i}" for i in range(frame.shape[1])]
    return frame


def load_csv(path: str | Path, *, archive_member: str | None = None) -> pd.DataFrame:
    path = Path(path)
    if archive_member:
        with zipfile.ZipFile(path) as archive: blob = archive.read(archive_member)
    else: blob = path.read_bytes()
    frame = _read_csv_bytes(blob)
    original = list(frame.columns)
    is_binance_kline = len(original) == len(BINANCE_KLINE_COLUMNS) and all(
        str(a).lower() == b for a, b in zip(original, BINANCE_KLINE_COLUMNS)
    )
    renamed = {}
    for col in original:
        key = re.sub(r"\s+", " ", str(col).strip().lower())
        renamed[col] = ALIASES.get(key, key.replace(" ", "_").replace("-", "_"))
    frame = frame.rename(columns=renamed)
    # Numeric Unix fields are epoch values; text timestamps are parsed directly.
    if "timestamp" not in frame and "source_date" in frame:
        frame = frame.rename(columns={"source_date": "timestamp"})
    if "timestamp" in frame:
        raw = frame["timestamp"]
        numeric = pd.to_numeric(raw, errors="coerce")
        if numeric.notna().mean() > 0.9:
            frame["timestamp_utc"] = _parse_epoch_series(numeric)
        else:
            frame["timestamp_utc"] = pd.to_datetime(raw, utc=True, errors="coerce")
    if "close_time" in frame:
        close = pd.to_numeric(frame["close_time"], errors="coerce")
        if close.notna().any(): frame["close_time_utc"] = _parse_epoch_series(close)
    for col in ("open", "high", "low", "close", "volume", "quote_volume", "trade_count", "taker_buy_base_volume", "taker_buy_quote_volume"):
        if col in frame: frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["source_file"] = path.name + (f"::{archive_member}" if archive_member else "")
    frame["source_priority"] = 0 if "fetched_" in path.name.lower() else 1 if is_binance_kline else 2
    frame["source_market"] = "BTCUSD" if "bitstamp" in path.name.lower() else "BTCUSDT"
    return frame


def discover_datasets(raw_dir: str | Path) -> list[tuple[Path, str | None]]:
    result = []
    for path in sorted(Path(raw_dir).rglob("*")):
        if path.suffix.lower() == ".csv": result.append((path, None))
        elif path.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(path) as zf:
                    result.extend((path, n) for n in zf.namelist() if n.lower().endswith(".csv"))
            except zipfile.BadZipFile: continue
    return result

