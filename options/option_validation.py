"""Strict validation for live-sourced option snapshots and straddle pairs."""
from datetime import datetime, timezone
from .contracts import OptionQuoteSnapshot


def validate_option_quote(quote: OptionQuoteSnapshot, *, now: datetime | None = None, max_age_seconds: int = 30) -> None:
    if quote.underlying.upper() != "BTC": raise ValueError("Option underlying must be BTC")
    if not quote.settlement_currency or quote.settlement_currency.upper() != "BTC":
        raise ValueError("BTC straddle analysis requires verified BTC-settled contracts")
    if quote.option_type.upper() not in {"CALL", "PUT"}: raise ValueError("Option type must be CALL or PUT")
    if not quote.instrument_id or not quote.active: raise ValueError("Option contract must be identified and active")
    if quote.strike <= 0 or quote.quantity <= 0 or quote.multiplier <= 0: raise ValueError("Strike, quantity and multiplier must be positive")
    if quote.bid <= 0 or quote.ask < quote.bid: raise ValueError("Quote must have a positive, non-crossed bid/ask")
    if quote.bid_size is not None and quote.bid_size < 0 or quote.ask_size is not None and quote.ask_size < 0:
        raise ValueError("Quote sizes cannot be negative")
    stamp = quote.quote_time.replace(tzinfo=timezone.utc) if quote.quote_time.tzinfo is None else quote.quote_time.astimezone(timezone.utc)
    current = now or datetime.now(timezone.utc)
    current = current.replace(tzinfo=timezone.utc) if current.tzinfo is None else current.astimezone(timezone.utc)
    age = (current - stamp).total_seconds()
    if age < 0 or age > max_age_seconds: raise ValueError("Option quote is stale or future-dated")


def validate_pair(call: OptionQuoteSnapshot, put: OptionQuoteSnapshot, *, now=None, max_age_seconds=30) -> None:
    validate_option_quote(call, now=now, max_age_seconds=max_age_seconds)
    validate_option_quote(put, now=now, max_age_seconds=max_age_seconds)
    if call.option_type.upper() != "CALL" or put.option_type.upper() != "PUT": raise ValueError("Pair must contain one Call and one Put")
    if call.underlying.upper() != put.underlying.upper(): raise ValueError("Pair underlyings must match")
    if call.strike != put.strike: raise ValueError("Call and Put strikes must match")
    if call.expiry != put.expiry: raise ValueError("Call and Put expiries must match")
    if call.quantity != put.quantity: raise ValueError("Call and Put quantities must match")
    if call.settlement_currency.upper() != put.settlement_currency.upper(): raise ValueError("Call and Put settlement currencies must match")

