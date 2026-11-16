"""Read-only downloader for Deribit's public historical BTC option trades."""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


HISTORY_API_ROOT = "https://history.deribit.com/api/v2"
REQUEST_TIMEOUT_SECONDS = 20
MAX_TRADES_PER_REQUEST = 1000


def _rpc(method: str, params: dict[str, Any]) -> Any:
    query = urlencode(params)
    request = Request(
        f"{HISTORY_API_ROOT}/public/{method}?{query}",
        headers={"User-Agent": "BTCOptionsStraddleResearch/1.0"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if exc.code in {429, 500, 502, 503, 504}:
            raise OSError(f"Deribit historical API temporary HTTP {exc.code}: {detail[:200]}") from exc
        raise RuntimeError(f"Deribit historical HTTP {exc.code}: {detail[:500]}") from exc
    if payload.get("error"):
        error = payload["error"]
        raise RuntimeError(f"Deribit historical {method} failed ({error.get('code')}): {error.get('message')}")
    if "result" not in payload:
        raise RuntimeError(f"Deribit historical {method} returned no result.")
    return payload["result"]


def parse_utc_date(value: str, *, end_of_day: bool = False) -> datetime:
    """Parse an ISO date or timestamp and return an aware UTC datetime."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        if "T" in value or " " in value:
            raise ValueError("Datetime inputs must include a timezone, for example 2025-01-01T00:00:00Z.")
        parsed = parsed.replace(tzinfo=timezone.utc)
        if end_of_day:
            parsed += timedelta(days=1, milliseconds=-1)
    return parsed.astimezone(timezone.utc)


def _fetch_interval(start_ms: int, end_ms: int) -> list[dict[str, Any]]:
    """Fetch an inclusive interval, bisecting it when the API page is capped."""
    params = {
        "currency": "BTC",
        "kind": "option",
        "start_timestamp": start_ms,
        "end_timestamp": end_ms,
        "count": MAX_TRADES_PER_REQUEST,
        "sorting": "asc",
    }
    for attempt in range(3):
        try:
            result = _rpc("get_last_trades_by_currency_and_time", params)
            break
        except OSError:
            if attempt == 2:
                raise
            time.sleep(0.5 * (2 ** attempt))
    trades = result.get("trades") or []
    if not result.get("has_more", False):
        return trades
    if end_ms <= start_ms:
        raise RuntimeError(f"More than {MAX_TRADES_PER_REQUEST} trades share a 1ms interval; cannot page without losing records.")
    midpoint = (start_ms + end_ms) // 2
    time.sleep(0.05)
    return _fetch_interval(start_ms, midpoint) + _fetch_interval(midpoint + 1, end_ms)


def fetch_btc_option_trades(start: datetime, end: datetime, *, window_hours: int = 24) -> list[dict[str, Any]]:
    """Download actual BTC option trades in the inclusive UTC time range."""
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("start and end must be timezone-aware datetimes.")
    start_ms = int(start.astimezone(timezone.utc).timestamp() * 1000)
    end_ms = int(end.astimezone(timezone.utc).timestamp() * 1000)
    if start_ms > end_ms:
        raise ValueError("start must be earlier than or equal to end.")
    if window_hours <= 0 or window_hours > 24 * 31:
        raise ValueError("window_hours must be between 1 and 744.")

    interval_ms = window_hours * 60 * 60 * 1000
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    cursor = start_ms
    while cursor <= end_ms:
        interval_end = min(cursor + interval_ms - 1, end_ms)
        for trade in _fetch_interval(cursor, interval_end):
            name = str(trade.get("instrument_name", ""))
            trade_id = str(trade.get("trade_id", ""))
            if not name or not trade_id:
                raise RuntimeError("Historical trade is missing instrument_name or trade_id; refusing to save an incomplete record.")
            rows[(name, trade_id)] = trade
        cursor = interval_end + 1
        if cursor <= end_ms:
            time.sleep(0.05)
    return sorted(rows.values(), key=lambda row: (int(row.get("timestamp", 0)), row["instrument_name"], row["trade_id"]))
