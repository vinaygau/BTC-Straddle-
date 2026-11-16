"""Download paginated, public Binance BTCUSDT historical klines to CSV."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
import json
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


API_ROOTS = ("https://api.binance.com", "https://data-api.binance.vision")
INTERVALS = {"5m": "5m", "15m": "15m", "1h": "1h", "daily": "1d"}
KLINE_COLUMNS = (
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume", "ignore",
)


def _utc_date_start(value: str) -> datetime:
    parsed = date.fromisoformat(value)
    return datetime.combine(parsed, time.min, tzinfo=timezone.utc)


def _request_page(timeframe: str, start_ms: int, end_ms: int) -> list[Any]:
    params = urlencode({
        "symbol": "BTCUSDT", "interval": INTERVALS[timeframe],
        "startTime": start_ms, "endTime": end_ms, "limit": 1000,
    })
    errors = []
    for root in API_ROOTS:
        request = Request(f"{root}/api/v3/klines?{params}",
                          headers={"User-Agent": "BTCOptionsStraddleResearch/1.0"})
        try:
            with urlopen(request, timeout=20) as response:
                result = json.loads(response.read().decode("utf-8"))
            if isinstance(result, list):
                return result
            errors.append(f"{root}: unexpected response")
        except HTTPError as exc:
            errors.append(f"{root}: HTTP {exc.code}")
        except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            errors.append(f"{root}: {exc}")
    raise RuntimeError("Binance public candle download failed (" + "; ".join(errors) + ")")


def download_btcusdt_history(
    timeframe: str,
    start_date: str,
    end_date: str,
    *,
    on_page: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Fetch every candle from start_date through end_date (both UTC dates inclusive)."""
    if timeframe not in INTERVALS:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    start = _utc_date_start(start_date)
    end = _utc_date_start(end_date) + timedelta(days=1) - timedelta(milliseconds=1)
    if end < start:
        raise ValueError("end_date must be on or after start_date")
    cursor_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    rows: list[list[Any]] = []
    page_count = 0
    while cursor_ms <= end_ms:
        page = _request_page(timeframe, cursor_ms, end_ms)
        if not page:
            break
        valid = [row for row in page if isinstance(row, list) and len(row) >= len(KLINE_COLUMNS)]
        if not valid:
            raise RuntimeError("Binance returned a page without valid kline rows.")
        rows.extend(valid)
        page_count += 1
        if on_page is not None:
            on_page(page_count, len(rows))
        last_open_ms = int(valid[-1][0])
        next_cursor_ms = last_open_ms + 1
        if next_cursor_ms <= cursor_ms:
            raise RuntimeError("Binance pagination did not advance; stopping to avoid a repeated download loop.")
        cursor_ms = next_cursor_ms
        if len(valid) < 1000:
            break
    frame = pd.DataFrame(rows, columns=KLINE_COLUMNS)
    if frame.empty:
        raise RuntimeError("Binance returned no candles for the requested interval.")
    frame["open_time"] = pd.to_numeric(frame["open_time"], errors="coerce")
    frame = frame.dropna(subset=["open_time"]).drop_duplicates("open_time", keep="last")
    frame = frame.loc[frame["open_time"].between(int(start.timestamp() * 1000), end_ms)]
    frame = frame.sort_values("open_time", kind="stable").reset_index(drop=True)
    return frame


def save_btcusdt_history(frame: pd.DataFrame, timeframe: str, start_date: str,
                         end_date: str, output_dir: str | Path) -> Path:
    """Write a source-priority Binance CSV consumable by the existing pipeline."""
    if timeframe not in INTERVALS:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"fetched_btcusdt_{timeframe}_{start_date}_{end_date}.csv"
    frame.to_csv(path, index=False)
    return path
