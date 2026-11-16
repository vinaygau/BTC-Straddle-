"""Read-only Deribit current-chain adapter; this never requests credentials."""
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, Sequence

from .contracts import OptionQuoteSnapshot


class OptionChainLoader(Protocol):
    def load_current_chain(self, underlying: str = "BTC") -> Sequence[OptionQuoteSnapshot]:
        """Return exchange-sourced current snapshots."""


@dataclass(frozen=True)
class OptionChainSnapshot:
    underlying: str
    reference_price: float
    quotes: tuple[OptionQuoteSnapshot, ...]
    fetched_at: datetime
    environment: str


class DeribitPublicOptionChainLoader:
    """Adapt the app's existing Deribit public catalog into option snapshots.

    The catalog is read-only and unauthenticated. No historical options are
    created, and incomplete market quotes are omitted rather than fabricated.
    """

    def __init__(self, *, environment: str = "production") -> None:
        if environment not in {"production", "testnet"}:
            raise ValueError("environment must be 'production' or 'testnet'")
        self.environment = environment

    def load_snapshot(self, underlying: str = "BTC") -> OptionChainSnapshot:
        if underlying.upper() != "BTC":
            raise ValueError("This Deribit catalog adapter currently supports BTC only")
        from data.deribit_live import fetch_live_btc_option_catalog, fetch_testnet_btc_option_catalog

        catalog = (fetch_live_btc_option_catalog() if self.environment == "production"
                   else fetch_testnet_btc_option_catalog())
        if catalog.environment != self.environment:
            raise RuntimeError("Deribit returned a catalog from an unexpected environment")
        fetched_at = _parse_time(catalog.fetched_at)
        quotes = []
        for contract in (*catalog.calls, *catalog.puts):
            summary = catalog.summaries.get(contract.instrument_id, {})
            bid, ask = summary.get("bid"), summary.get("ask")
            quote_time = _parse_time(summary.get("timestamp"))
            # No quote timestamp, bid/ask, or known contract multiplier means
            # the snapshot cannot be safely valued or freshness-checked.
            if bid is None or ask is None or quote_time is None or contract.multiplier is None:
                continue
            quotes.append(OptionQuoteSnapshot(
                instrument_id=contract.instrument_id,
                underlying="BTC",
                option_type=contract.option_type,
                strike=float(contract.strike),
                expiry=contract.expiry,
                bid=float(bid),
                ask=float(ask),
                bid_size=summary.get("bid_size"),
                ask_size=summary.get("ask_size"),
                quote_time=quote_time,
                active=bool(contract.is_active),
                quantity=1.0,
                multiplier=float(contract.multiplier),
                implied_volatility=summary.get("implied_volatility"),
                delta=summary.get("delta"),
                gamma=summary.get("gamma"),
                theta=summary.get("theta"),
                vega=summary.get("vega"),
                settlement_currency=contract.settlement_currency,
                mark_price=summary.get("mark_price"),
            ))
        return OptionChainSnapshot("BTC", float(catalog.reference_price), tuple(quotes), fetched_at,
                                   catalog.environment)

    def load_current_chain(self, underlying: str = "BTC") -> Sequence[OptionQuoteSnapshot]:
        return self.load_snapshot(underlying).quotes


def _parse_time(value) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        from datetime import timezone
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed

