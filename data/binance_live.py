"""Read-only public Binance BTCUSDT candles with the project's training schema."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


_API_ROOTS = ("https://api.binance.com", "https://data-api.binance.vision")
_TIMEFRAMES = {"5m": "5m", "15m": "15m", "1h": "1h", "daily": "1d"}


def fetch_btcusdt_candles(timeframe: str, *, limit: int = 240) -> pd.DataFrame:
    """Fetch completed BTCUSDT klines and preserve volume/trade-flow features."""
    if timeframe not in _TIMEFRAMES:
        raise ValueError(f"Unsupported model timeframe: {timeframe}")
    if not 60 <= limit <= 1000:
        raise ValueError("limit must be between 60 and 1000 to cover the model's rolling windows.")
    params = urlencode({"symbol": "BTCUSDT", "interval": _TIMEFRAMES[timeframe], "limit": limit})
    payload: list[Any] | None = None
    errors = []
    for root in _API_ROOTS:
        request = Request(f"{root}/api/v3/klines?{params}", headers={"User-Agent": "BTCOptionsStraddleResearch/1.0"})
        try:
            with urlopen(request, timeout=12) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except HTTPError as exc:
            errors.append(f"{root}: HTTP {exc.code}")
        except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            errors.append(f"{root}: {exc}")
    if not isinstance(payload, list) or not payload:
        raise RuntimeError("Could not fetch BTCUSDT candles from Binance public data (" + "; ".join(errors) + ")")

    columns = [
        "timestamp", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trade_count", "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
    ]
    rows = []
    for item in payload:
        if not isinstance(item, list) or len(item) < 12:
            continue
        try:
            rows.append({
                "timestamp": int(item[0]),
                "open": float(item[1]), "high": float(item[2]), "low": float(item[3]),
                "close": float(item[4]), "volume": float(item[5]), "close_time": int(item[6]),
                "quote_volume": float(item[7]), "trade_count": int(item[8]),
                "taker_buy_base_volume": float(item[9]), "taker_buy_quote_volume": float(item[10]),
                "ignore": item[11],
            })
        except (TypeError, ValueError, OverflowError):
            continue
    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        raise RuntimeError("Binance returned no valid BTCUSDT candle rows.")
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    frame["close_time_utc"] = pd.to_datetime(frame["close_time"], unit="ms", utc=True)
    now = pd.Timestamp(datetime.now(timezone.utc))
    frame = frame.loc[frame["close_time_utc"] <= now].sort_values("timestamp_utc").drop_duplicates("timestamp_utc", keep="last")
    frame = frame.reset_index(drop=True)
    if len(frame) < 60:
        raise RuntimeError(f"Only {len(frame)} completed {timeframe} candles were returned; at least 60 are required.")
    close_times = frame["close_time_utc"]
    expected = {"5m": pd.Timedelta(minutes=5), "15m": pd.Timedelta(minutes=15),
                "1h": pd.Timedelta(hours=1), "daily": pd.Timedelta(days=1)}[timeframe]
    if not close_times.diff().dropna().eq(expected).all():
        # Preserve actual missing intervals; the model input validator will reject them if its latest sequence is affected.
        frame.attrs["contains_gaps"] = True
    else:
        frame.attrs["contains_gaps"] = False
    return frame
