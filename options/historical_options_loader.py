"""Load the project's historical Deribit trade tape without inventing quotes.

The available archive contains executed trade prints plus mark/IV fields. This
module deliberately labels it as proxy data; it is not an order-book loader.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


TRADE_FIELDS = ("instrument_name", "timestamp_utc", "price", "amount", "mark_price", "iv")
QUOTE_FIELDS = ("bid", "ask", "bid_size", "ask_size")


@dataclass(frozen=True)
class HistoricalOptionsData:
    trades: pd.DataFrame
    source_path: str
    first_timestamp_utc: str | None
    last_timestamp_utc: str | None
    row_count: int
    source_kind: str = "deribit_executed_trade_tape_with_mark_price_proxy"
    has_historical_bid_ask: bool = False
    has_displayed_size: bool = False
    executable_fill_evidence: bool = False
    warnings: tuple[str, ...] = (
        "Trade prices and mark prices are historical observations, not confirmed strategy fills.",
        "Historical bid/ask, displayed order-book size, queue position, and fills are unavailable.",
    )


def load_historical_option_trades(
    source: str | Path | pd.DataFrame,
    *,
    start_utc: str | pd.Timestamp | None = None,
    end_utc: str | pd.Timestamp | None = None,
) -> HistoricalOptionsData:
    """Read the archived tape and report its actual field coverage."""
    if isinstance(source, pd.DataFrame):
        trades = source.copy()
        source_name = "<in-memory trade tape>"
    else:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(path)
        trades = pd.read_parquet(path)
        source_name = str(path)
    missing = set(TRADE_FIELDS).difference(trades.columns)
    if missing:
        raise ValueError(f"Historical option tape lacks required trade fields: {sorted(missing)}")
    trades["timestamp_utc"] = pd.to_datetime(trades.timestamp_utc, utc=True, errors="coerce")
    for column in ("price", "amount", "mark_price", "iv"):
        trades[column] = pd.to_numeric(trades[column], errors="coerce")
    trades = trades.dropna(subset=["instrument_name", "timestamp_utc", "price", "amount"])
    trades = trades.loc[(trades.price > 0) & (trades.amount > 0)]
    if start_utc is not None:
        start = pd.Timestamp(start_utc)
        if start.tzinfo is None:
            raise ValueError("start_utc must be timezone-aware")
        trades = trades.loc[trades.timestamp_utc >= start.tz_convert("UTC")]
    if end_utc is not None:
        end = pd.Timestamp(end_utc)
        if end.tzinfo is None:
            raise ValueError("end_utc must be timezone-aware")
        trades = trades.loc[trades.timestamp_utc <= end.tz_convert("UTC")]
    trades = trades.sort_values("timestamp_utc", kind="stable").reset_index(drop=True)
    if trades.empty:
        raise ValueError("No valid historical option trade rows remain for the requested period")
    # The current Deribit tape processor's actual schema has no BBO/depth fields.
    has_bbo = set(QUOTE_FIELDS).issubset(trades.columns) and bool(trades.loc[:, list(QUOTE_FIELDS)].notna().all().all())
    size_fields = ("bid_size", "ask_size")
    has_size = set(size_fields).issubset(trades.columns) and bool(trades.loc[:, list(size_fields)].notna().all().all())
    return HistoricalOptionsData(
        trades=trades,
        source_path=source_name,
        first_timestamp_utc=trades.timestamp_utc.iloc[0].isoformat().replace("+00:00", "Z"),
        last_timestamp_utc=trades.timestamp_utc.iloc[-1].isoformat().replace("+00:00", "Z"),
        row_count=len(trades),
        source_kind="historical_order_book" if has_bbo else "deribit_executed_trade_tape_with_mark_price_proxy",
        has_historical_bid_ask=bool(has_bbo),
        has_displayed_size=bool(has_size),
        executable_fill_evidence=False,
    )
