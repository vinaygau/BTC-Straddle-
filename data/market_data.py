"""Phase 3 BTC market-data discovery and quote validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class OptionQuote:
    """Normalized quote snapshot for one BTC option instrument."""

    instrument_id: str
    option_type: str
    strike: float
    expiry: str
    bid: float | None = None
    ask: float | None = None
    bid_size: float | None = None
    ask_size: float | None = None
    timestamp: str | None = None
    is_active: bool = True
    multiplier: float | None = None
    quote_currency: str | None = None
    settlement_currency: str | None = None

    def is_valid(self) -> bool:
        if not self.is_active:
            return False
        if not self.instrument_id:
            return False
        if self.option_type.upper() not in {"CALL", "PUT"}:
            return False
        if self.strike <= 0:
            return False
        if self.bid is None or self.ask is None:
            return False
        return self.bid > 0 and self.ask > 0


@dataclass(frozen=True)
class ReferencePrice:
    """BTC reference-price snapshot used for ATM selection."""

    value: float
    source: str
    timestamp: str | None = None

    def is_valid(self) -> bool:
        return self.value > 0


@dataclass(frozen=True)
class InstrumentMetadata:
    """Normalized exchange metadata for a BTC option instrument."""

    instrument_id: str
    underlying: str
    option_type: str
    strike: float
    expiry: str
    is_active: bool = True
    is_tradeable: bool = True
    multiplier: float | None = None
    tick_size: float | None = None
    min_quantity: float | None = None
    lot_size: float | None = None
    quote_currency: str | None = None
    settlement_currency: str | None = None
    timestamp: str | None = None

    def is_valid(self) -> bool:
        return (
            bool(self.instrument_id)
            and self.underlying.upper() == "BTC"
            and self.option_type.upper() in {"CALL", "PUT"}
            and self.strike > 0
            and bool(self.expiry)
            and self.is_active
            and self.is_tradeable
        )


def _parse_option_type(raw: Any) -> str:
    if raw is None:
        return ""
    value = str(raw).strip().upper()
    if value in {"CALL", "C", "CALL_OPTION", "OPTION_CALL"}:
        return "CALL"
    if value in {"PUT", "P", "PUT_OPTION", "OPTION_PUT"}:
        return "PUT"
    return value


def _parse_expiry(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, datetime):
        return raw.strftime("%Y-%m-%dT%H:%M:%SZ")
    value = str(raw).strip()
    if not value:
        return ""
    return value


def is_future_expiry(expiry: str | None) -> bool:
    """Return True when an expiry is in the future and not stale."""
    if not expiry:
        return False
    try:
        if "T" in expiry or " " in expiry:
            dt = datetime.fromisoformat(expiry.replace("Z", "+00:00"))
        else:
            dt = datetime.fromisoformat(expiry)
    except ValueError:
        return True
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc) > datetime.now(timezone.utc)


def _parse_strike(raw: Any, instrument_id: str) -> float:
    if raw not in (None, ""):
        try:
            return float(raw)
        except (TypeError, ValueError):
            pass

    match = None
    if isinstance(instrument_id, str):
        match = __import__("re").search(r"-(\d+(?:\.\d+)?)-?[CP]$", instrument_id.upper())
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass

    raise ValueError(f"Instrument strike could not be parsed for {instrument_id!r}.")


def _extract_underlying(raw: Any) -> str:
    if raw is None:
        return ""
    return str(raw).strip().upper()


def _to_iso_timestamp(raw: Any) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return str(raw)


def filter_btc_option_instruments(
    instruments: Iterable[Mapping[str, Any]],
    *,
    require_tradeable: bool = True,
) -> list[InstrumentMetadata]:
    """Filter a raw exchange instrument list to valid BTC Call/Put instruments."""
    results: list[InstrumentMetadata] = []

    for item in instruments:
        if not isinstance(item, Mapping):
            continue

        instrument_id = str(
            item.get("instrument_id")
            or item.get("instrument_name")
            or item.get("symbol")
            or ""
        ).strip()
        if not instrument_id:
            continue

        underlying = _extract_underlying(
            item.get("underlying")
            or item.get("base_currency")
            or item.get("currency")
            or item.get("underlying_currency")
        )

        option_type = _parse_option_type(
            item.get("option_type")
            or item.get("type")
            or item.get("kind")
            or ("C" if instrument_id.upper().endswith("-C") else "")
            or ("P" if instrument_id.upper().endswith("-P") else "")
        )
        if not option_type:
            continue

        if underlying and underlying != "BTC":
            continue

        is_active = bool(item.get("is_active", item.get("active", True)))
        is_tradeable = bool(item.get("is_tradeable", item.get("tradeable", True)))
        if require_tradeable and not is_tradeable:
            continue

        strike_raw = item.get("strike")
        try:
            strike = _parse_strike(strike_raw, instrument_id)
        except ValueError:
            continue

        expiry = _parse_expiry(
            item.get("expiry")
            or item.get("expiration")
            or item.get("expiry_time")
            or item.get("settlement_date")
            or ""
        )
        if not expiry or not is_future_expiry(expiry):
            continue

        metadata = InstrumentMetadata(
            instrument_id=instrument_id,
            underlying="BTC",
            option_type=option_type,
            strike=float(strike),
            expiry=expiry,
            is_active=bool(is_active),
            is_tradeable=bool(is_tradeable),
            multiplier=item.get("multiplier"),
            tick_size=item.get("tick_size"),
            min_quantity=item.get("min_quantity"),
            lot_size=item.get("lot_size"),
            quote_currency=item.get("quote_currency"),
            settlement_currency=item.get("settlement_currency"),
            timestamp=_to_iso_timestamp(item.get("timestamp") or item.get("updated_at")),
        )
        if metadata.is_valid():
            results.append(metadata)

    return results


def validate_reference_price(reference: ReferencePrice) -> bool:
    """Confirm the BTC reference price is present and positive."""
    return reference.is_valid()


def has_fresh_quote(quote: OptionQuote, *, max_age_seconds: int = 30) -> bool:
    """Return True when the quote timestamp is within the freshness threshold."""
    if not quote.is_valid():
        return False

    if quote.timestamp is None:
        return False

    try:
        timestamp = datetime.fromisoformat(quote.timestamp.replace("Z", "+00:00"))
    except ValueError:
        return False

    now = datetime.now(timezone.utc)
    return now - timestamp.astimezone(timezone.utc) <= timedelta(seconds=max_age_seconds)


def discover_btc_option_instruments(
    raw_instruments: Iterable[Mapping[str, Any]],
    *,
    require_tradeable: bool = True,
) -> list[InstrumentMetadata]:
    """Public wrapper for deriving valid BTC option metadata from a raw catalogue."""
    return filter_btc_option_instruments(raw_instruments, require_tradeable=require_tradeable)


def validate_quote(quote: OptionQuote) -> bool:
    """Validate a quote before it is used in strategy or risk calculations."""
    return quote.is_valid() and has_fresh_quote(quote)


def select_nearest_atm_strike(
    reference_price: float,
    strikes: Iterable[float],
    *,
    tie_breaker: str = "lower",
) -> float:
    """Return the strike closest to the BTC reference price."""
    values = sorted(float(strike) for strike in strikes if float(strike) > 0)
    if not values:
        raise ValueError("No valid strikes were provided for ATM selection.")

    if tie_breaker not in {"lower", "higher"}:
        raise ValueError("tie_breaker must be 'lower' or 'higher'.")

    closest = min(
        values,
        key=lambda strike: (
            abs(strike - reference_price),
            strike if tie_breaker == "lower" else -strike,
        ),
    )
    return float(closest)
