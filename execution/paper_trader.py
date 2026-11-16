"""Paper-only, metadata-checked execution and position accounting for BTC options."""
from __future__ import annotations

import math
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from typing import Any

from data.market_data import OptionQuote


@dataclass(frozen=True)
class PaperPosition:
    # The first ten fields retain the former session-state shape for migration.
    call_instrument_id: str
    put_instrument_id: str
    strike: float
    expiry: str
    requested_amount: float
    open_amount: float
    entry_premium: float
    realized_pnl: float
    opened_at: str
    position_id: str = ""
    call_entry_price: float | None = None
    put_entry_price: float | None = None
    call_multiplier: float | None = None
    put_multiplier: float | None = None
    quote_currency: str | None = None
    settlement_currency: str | None = None
    initial_filled_amount: float | None = None
    call_open_amount: float | None = None
    put_open_amount: float | None = None
    entry_fees: float = 0.0
    entry_slippage: float = 0.0
    realized_gross_pnl: float = 0.0
    realized_exit_fees: float = 0.0
    realized_slippage: float = 0.0
    fee_rate_per_side: float = 0.0
    fee_cap_fraction: float = 0.125
    slippage_rate_per_side: float = 0.0
    status: str = "OPEN"
    last_valuation_at: str | None = None
    environment: str = "production"


@dataclass(frozen=True)
class PositionValuation:
    call_closing_cost: float
    put_closing_cost: float
    call_gross_pnl: float
    put_gross_pnl: float
    call_net_pnl: float
    put_net_pnl: float
    gross_unrealized_pnl: float
    allocated_entry_fees: float
    estimated_exit_fees: float
    allocated_entry_slippage: float
    estimated_exit_slippage: float
    net_unrealized_pnl: float
    total_fees: float
    total_slippage: float
    currency: str
    valued_at: str


def _settings(config: dict[str, Any] | None) -> dict[str, Any]:
    if config is None:
        return {
            "execution": {"allow_partial_fills": True, "require_both_legs": True},
            "quote_validation": {"maximum_quote_age_seconds": 5.0},
            "accounting": {"fee_rate_per_side": 0.0, "slippage_rate_per_side": 0.0,
                           "fee_cap_percent_of_premium": 12.5,
                           "include_entry_fees": True, "include_exit_fees": True, "include_slippage": True},
            "risk": {"stop_loss_enabled": True, "stop_loss_method": "percentage_of_entry_credit",
                     "stop_loss_percentage": 20.0, "take_profit_enabled": True,
                     "take_profit_percentage": 30.0},
            "expiry": {"validate_expiry": True},
        }
    return config


def _utc_now(now: datetime | None = None) -> datetime:
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise ValueError("Paper-trading time must include a timezone.")
    return current.astimezone(timezone.utc)


def _parse_datetime(value: str | None, label: str) -> datetime:
    if not value:
        raise ValueError(f"{label} timestamp is unavailable.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} timestamp is invalid.") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} timestamp has no timezone.")
    return parsed.astimezone(timezone.utc)


def quote_status(
    quote: OptionQuote,
    *,
    max_age_seconds: float = 5.0,
    now: datetime | None = None,
) -> tuple[str, str | None]:
    """Validate both sides of a live option quote and report a UI-ready status."""
    if not quote.is_active:
        return "Unavailable", "Instrument is not active."
    if quote.bid is None or quote.ask is None:
        return "Unavailable", "Bid or ask is missing."
    if not all(math.isfinite(value) and value > 0 for value in (quote.bid, quote.ask)):
        return "Invalid", "Bid and ask must be finite positive prices."
    if quote.bid > quote.ask:
        return "Invalid", "Bid exceeds ask (crossed quote)."
    if quote.bid_size is None or quote.ask_size is None:
        return "Unavailable", "Bid or ask size is missing."
    if not all(math.isfinite(value) and value >= 0 for value in (quote.bid_size, quote.ask_size)):
        return "Invalid", "Bid and ask sizes must be finite and non-negative."
    try:
        quote_time = _parse_datetime(quote.timestamp, "Quote")
    except ValueError as exc:
        return "Unavailable", str(exc)
    age = (_utc_now(now) - quote_time).total_seconds()
    if age < -2.0:
        return "Invalid", "Quote timestamp is in the future."
    if age > max_age_seconds:
        return "Stale", f"Quote is {age:.1f}s old; limit is {max_age_seconds:g}s."
    return "Fresh", None


def _validate_btc_inverse_pair(call: OptionQuote, put: OptionQuote) -> None:
    if call.option_type.upper() != "CALL" or put.option_type.upper() != "PUT":
        raise ValueError("Paper execution requires one Call and one Put.")
    if call.strike != put.strike or call.expiry != put.expiry:
        raise ValueError("The paper position must use the same strike and expiry.")
    if not call.instrument_id or not put.instrument_id or call.instrument_id == put.instrument_id:
        raise ValueError("Call and Put instrument identifiers are missing or duplicated.")
    for label, quote in (("Call", call), ("Put", put)):
        multiplier = quote.multiplier
        if multiplier is None or not math.isfinite(float(multiplier)) or float(multiplier) <= 0:
            raise ValueError(f"{label} contract multiplier is unavailable; accounting is disabled.")
        # Deribit's inverse BTC option amount is already expressed in BTC units.
        # The exchange metadata multiplier is 1 BTC per option contract; applying
        # it again to the BTC-denominated amount would double count the contract.
        if not math.isclose(float(multiplier), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{label} multiplier {multiplier:g} is unsupported by the BTC inverse paper calculator.")
        if (quote.quote_currency or "").upper() != "BTC":
            raise ValueError(f"{label} quote currency is unknown or not BTC; no conversion is configured.")
        if (quote.settlement_currency or "").upper() != "BTC":
            raise ValueError(f"{label} settlement currency is unknown or not BTC; no conversion is configured.")
    if call.quote_currency.upper() != put.quote_currency.upper():
        raise ValueError("Call and Put premium currencies differ.")
    if call.settlement_currency.upper() != put.settlement_currency.upper():
        raise ValueError("Call and Put settlement currencies differ.")


def _validate_quotes(call: OptionQuote, put: OptionQuote, *, config: dict[str, Any], now: datetime | None) -> None:
    _validate_btc_inverse_pair(call, put)
    max_age = float(config["quote_validation"]["maximum_quote_age_seconds"])
    for label, quote in (("Call", call), ("Put", put)):
        status, reason = quote_status(quote, max_age_seconds=max_age, now=now)
        if status != "Fresh":
            raise ValueError(f"{label} quote is {status.lower()}: {reason}")
    current = _utc_now(now)
    expiry = _parse_datetime(call.expiry, "Option expiry")
    if expiry <= current:
        raise ValueError("Option is expired; use the separate settlement workflow, not a stale-ask close.")


def validate_pair_for_execution(
    call: OptionQuote,
    put: OptionQuote,
    *,
    config: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> tuple[bool, str | None]:
    """Non-throwing pair/metadata/expiry/freshness check for the app's entry UI."""
    try:
        _validate_quotes(call, put, config=_settings(config), now=now)
    except (ValueError, TypeError) as exc:
        return False, str(exc)
    return True, None


def _cost_rates(config: dict[str, Any]) -> tuple[float, float]:
    accounting = config["accounting"]
    fee_rate = float(accounting.get("fee_rate_per_side", 0.0))
    slip_rate = float(accounting.get("slippage_rate_per_side", 0.0)) if accounting.get("include_slippage", True) else 0.0
    if not all(math.isfinite(value) and value >= 0 for value in (fee_rate, slip_rate)):
        raise ValueError("Configured paper fees and slippage must be finite and non-negative.")
    return fee_rate, slip_rate


def _option_fee(price: float, amount: float, fee_rate: float, cap_fraction: float) -> float:
    """Deribit BTC option fee estimate, capped as a share of option premium."""
    if fee_rate <= 0 or amount <= 0 or price <= 0:
        return 0.0
    return min(fee_rate * amount, cap_fraction * price * amount)


def _entry_notional(call_price: float, put_price: float, amount: float) -> float:
    # BTC inverse options quote BTC per 1 BTC underlying and Deribit option
    # amount is already the BTC underlying-equivalent amount (multiplier=1).
    return (call_price + put_price) * amount


def simulate_entry(
    call: OptionQuote,
    put: OptionQuote,
    requested_amount: float,
    *,
    config: dict[str, Any] | None = None,
    now: datetime | None = None,
    position_id: str | None = None,
    environment: str = "production",
) -> PaperPosition:
    """Sell both legs at displayed bids for a matched, metadata-validated amount."""
    config = _settings(config)
    if not math.isfinite(requested_amount) or requested_amount <= 0:
        raise ValueError("The simulated amount must be greater than zero.")
    _validate_quotes(call, put, config=config, now=now)
    if call.bid_size is None or put.bid_size is None:
        raise ValueError("Both Call and Put bid sizes are required for a paper fill.")
    if not all(math.isfinite(value) and value >= 0 for value in (call.bid_size, put.bid_size)):
        raise ValueError("Bid sizes must be finite and non-negative.")
    filled_amount = min(requested_amount, call.bid_size, put.bid_size)
    if filled_amount <= 0:
        raise ValueError("The displayed book has no matched amount available.")
    if not config["execution"]["allow_partial_fills"] and filled_amount < requested_amount:
        raise ValueError("Available matched bid size is below the requested amount; partial entry is disabled.")

    entry_premium = call.bid + put.bid
    fee_rate, slippage_rate = _cost_rates(config)
    accounting = config["accounting"]
    fee_cap = float(accounting.get("fee_cap_percent_of_premium", 12.5)) / 100.0
    entry_fees = (
        _option_fee(call.bid, filled_amount, fee_rate, fee_cap)
        + _option_fee(put.bid, filled_amount, fee_rate, fee_cap)
        if accounting.get("include_entry_fees", True) else 0.0
    )
    entry_slippage = _entry_notional(call.bid, put.bid, filled_amount) * slippage_rate
    partial = filled_amount + 1e-12 < requested_amount
    now_iso = _utc_now(now).strftime("%Y-%m-%dT%H:%M:%SZ")
    return PaperPosition(
        call_instrument_id=call.instrument_id,
        put_instrument_id=put.instrument_id,
        strike=call.strike,
        expiry=call.expiry,
        requested_amount=float(requested_amount),
        open_amount=float(filled_amount),
        entry_premium=float(entry_premium),
        realized_pnl=0.0,
        opened_at=now_iso,
        position_id=position_id or str(uuid.uuid4()),
        call_entry_price=float(call.bid),
        put_entry_price=float(put.bid),
        call_multiplier=float(call.multiplier),
        put_multiplier=float(put.multiplier),
        quote_currency=call.quote_currency.upper(),
        settlement_currency=call.settlement_currency.upper(),
        initial_filled_amount=float(filled_amount),
        call_open_amount=float(filled_amount),
        put_open_amount=float(filled_amount),
        entry_fees=float(entry_fees),
        entry_slippage=float(entry_slippage),
        fee_rate_per_side=fee_rate,
        fee_cap_fraction=fee_cap,
        slippage_rate_per_side=slippage_rate,
        status="PARTIALLY_FILLED" if partial else "OPEN",
        environment=environment,
    )


def value_position(
    position: PaperPosition,
    call: OptionQuote,
    put: OptionQuote,
    *,
    config: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> PositionValuation:
    """Estimate executable close value and net P&L from fresh asks in BTC."""
    config = _settings(config)
    _validate_quotes(call, put, config=config, now=now)
    if call.instrument_id != position.call_instrument_id or put.instrument_id != position.put_instrument_id:
        raise ValueError("The live quotes do not match the open paper position.")
    if call.multiplier != position.call_multiplier or put.multiplier != position.put_multiplier:
        raise ValueError("Live instrument multiplier differs from the position's stored metadata.")
    if call.ask_size is None or put.ask_size is None:
        raise ValueError("Both ask sizes are required to value the position at executable top-of-book depth.")
    if not all(math.isfinite(value) and value >= 0 for value in (call.ask_size, put.ask_size)):
        raise ValueError("Ask sizes must be finite and non-negative.")
    call_remaining = position.call_open_amount if position.call_open_amount is not None else position.open_amount
    put_remaining = position.put_open_amount if position.put_open_amount is not None else position.open_amount
    if abs(call_remaining - put_remaining) > 1e-12:
        raise ValueError("Call and Put open amounts differ; combined-straddle valuation is unavailable.")
    amount = float(call_remaining)
    if amount <= 0:
        raise ValueError("The paper position has no remaining open amount.")
    if min(call.ask_size, put.ask_size) < amount:
        raise ValueError("Displayed ask depth cannot close the full remaining amount; partial-close value only is available.")
    call_entry = position.call_entry_price
    put_entry = position.put_entry_price
    if call_entry is None or put_entry is None:
        raise ValueError("Stored per-leg entry prices are unavailable; migrate/reopen the paper position before valuation.")
    call_cost = call.ask * amount
    put_cost = put.ask * amount
    call_gross = (call_entry - call.ask) * amount
    put_gross = (put_entry - put.ask) * amount
    gross = call_gross + put_gross
    initial = position.initial_filled_amount or position.open_amount
    allocated_ratio = amount / initial if initial > 0 else 1.0
    allocated_entry_fees = position.entry_fees * allocated_ratio
    allocated_entry_slippage = position.entry_slippage * allocated_ratio
    fee_rate = position.fee_rate_per_side if config["accounting"].get("include_exit_fees", True) else 0.0
    slip_rate = position.slippage_rate_per_side if config["accounting"].get("include_slippage", True) else 0.0
    exit_cost = call_cost + put_cost
    exit_fees = (_option_fee(call.ask, amount, fee_rate, position.fee_cap_fraction)
                 + _option_fee(put.ask, amount, fee_rate, position.fee_cap_fraction))
    exit_slippage = _entry_notional(call.ask, put.ask, amount) * slip_rate
    net = gross - allocated_entry_fees - allocated_entry_slippage - exit_fees - exit_slippage
    entry_credit = call_entry + put_entry
    call_share = call_entry / entry_credit if entry_credit > 0 else 0.5
    put_share = 1.0 - call_share
    call_net = call_gross - allocated_entry_fees * call_share - allocated_entry_slippage * call_share - _option_fee(call.ask, amount, fee_rate, position.fee_cap_fraction) - call_cost * slip_rate
    put_net = put_gross - allocated_entry_fees * put_share - allocated_entry_slippage * put_share - _option_fee(put.ask, amount, fee_rate, position.fee_cap_fraction) - put_cost * slip_rate
    total_fees = position.entry_fees + position.realized_exit_fees + exit_fees
    total_slippage = position.entry_slippage + position.realized_slippage + exit_slippage
    return PositionValuation(
        call_closing_cost=float(call_cost), put_closing_cost=float(put_cost),
        call_gross_pnl=float(call_gross), put_gross_pnl=float(put_gross),
        call_net_pnl=float(call_net), put_net_pnl=float(put_net),
        gross_unrealized_pnl=float(gross), allocated_entry_fees=float(allocated_entry_fees),
        estimated_exit_fees=float(exit_fees), allocated_entry_slippage=float(allocated_entry_slippage),
        estimated_exit_slippage=float(exit_slippage), net_unrealized_pnl=float(net),
        total_fees=float(total_fees), total_slippage=float(total_slippage),
        currency=position.settlement_currency or "BTC",
        valued_at=_utc_now(now).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def risk_thresholds(position: PaperPosition, config: dict[str, Any]) -> dict[str, float | str | bool]:
    """Position-level net loss/profit thresholds based on initial received credit."""
    risk = config["risk"]
    remaining_amount = position.call_open_amount if position.call_open_amount is not None else position.open_amount
    gross_credit = _entry_notional(
        position.call_entry_price or 0.0,
        position.put_entry_price or 0.0,
        remaining_amount,
    )
    max_loss = gross_credit * float(risk["stop_loss_percentage"]) / 100.0
    target_profit = gross_credit * float(risk["take_profit_percentage"]) / 100.0
    return {
        "stop_loss_enabled": bool(risk["stop_loss_enabled"]),
        "take_profit_enabled": bool(risk["take_profit_enabled"]),
        "maximum_loss": max_loss,
        "target_profit": target_profit,
        "maximum_loss_percent": float(risk["stop_loss_percentage"]),
        "target_profit_percent": float(risk["take_profit_percentage"]),
        "method": str(risk["stop_loss_method"]),
    }


def evaluate_position_risk(
    position: PaperPosition,
    valuation: PositionValuation,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Compare net executable P&L with configured position-level loss/profit limits."""
    limits = risk_thresholds(position, config)
    stop = bool(limits["stop_loss_enabled"] and valuation.net_unrealized_pnl <= -float(limits["maximum_loss"]))
    take = bool(limits["take_profit_enabled"] and valuation.net_unrealized_pnl >= float(limits["target_profit"]))
    return {
        **limits,
        "stop_loss_triggered": stop,
        "take_profit_triggered": take,
        "status": "STOP_LOSS_SIGNAL" if stop else "TAKE_PROFIT_SIGNAL" if take else "WITHIN_LIMITS",
        "current_loss": max(0.0, -valuation.net_unrealized_pnl),
    }


def simulate_exit(
    position: PaperPosition,
    call: OptionQuote,
    put: OptionQuote,
    *,
    config: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> tuple[PaperPosition, float, float]:
    """Buy back matched Call/Put quantity at asks; partial close leaves both legs open."""
    config = _settings(config)
    _validate_quotes(call, put, config=config, now=now)
    if call.instrument_id != position.call_instrument_id or put.instrument_id != position.put_instrument_id:
        raise ValueError("The live quotes do not match the open paper position.")
    if call.multiplier != position.call_multiplier or put.multiplier != position.put_multiplier:
        raise ValueError("Live instrument multiplier differs from the position's stored metadata.")
    if call.ask_size is None or put.ask_size is None:
        raise ValueError("Both legs need an ask size for a paper close.")
    if not all(math.isfinite(value) and value >= 0 for value in (call.ask_size, put.ask_size)):
        raise ValueError("Ask sizes must be finite and non-negative.")
    call_remaining = position.call_open_amount if position.call_open_amount is not None else position.open_amount
    put_remaining = position.put_open_amount if position.put_open_amount is not None else position.open_amount
    if abs(call_remaining - put_remaining) > 1e-12:
        raise ValueError("Call and Put remaining quantities differ; paired close is blocked for safety.")
    remaining = float(call_remaining)
    if remaining <= 0:
        raise ValueError("The paper position is already closed.")
    closed_amount = min(remaining, call.ask_size, put.ask_size)
    if closed_amount <= 0:
        raise ValueError("The displayed ask book has no matched amount available.")
    if not config["execution"]["allow_partial_fills"] and closed_amount < remaining:
        raise ValueError("Available matched ask size is below the open amount; partial close is disabled.")
    call_entry = position.call_entry_price
    put_entry = position.put_entry_price
    if call_entry is None or put_entry is None:
        raise ValueError("Stored per-leg entry prices are unavailable; cannot calculate realized P&L.")

    gross_delta = ((call_entry - call.ask) + (put_entry - put.ask)) * closed_amount
    initial = position.initial_filled_amount or position.open_amount
    closed_ratio = closed_amount / initial if initial > 0 else 1.0
    allocated_entry_costs = (position.entry_fees + position.entry_slippage) * closed_ratio
    accounting = config["accounting"]
    fee_rate = position.fee_rate_per_side if accounting.get("include_exit_fees", True) else 0.0
    slip_rate = position.slippage_rate_per_side if accounting.get("include_slippage", True) else 0.0
    exit_notional = _entry_notional(call.ask, put.ask, closed_amount)
    exit_fees = (_option_fee(call.ask, closed_amount, fee_rate, position.fee_cap_fraction)
                 + _option_fee(put.ask, closed_amount, fee_rate, position.fee_cap_fraction))
    exit_slippage = exit_notional * slip_rate
    net_delta = gross_delta - allocated_entry_costs - exit_fees - exit_slippage
    new_remaining = max(remaining - closed_amount, 0.0)
    partial = new_remaining > 1e-12
    updated = replace(
        position,
        open_amount=new_remaining,
        call_open_amount=new_remaining,
        put_open_amount=new_remaining,
        realized_pnl=position.realized_pnl + net_delta,
        realized_gross_pnl=position.realized_gross_pnl + gross_delta,
        realized_exit_fees=position.realized_exit_fees + exit_fees,
        realized_slippage=position.realized_slippage + exit_slippage,
        status="PARTIALLY_CLOSED" if partial else "CLOSED",
        last_valuation_at=_utc_now(now).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    return updated, float(closed_amount), float(net_delta)


def position_to_dict(position: PaperPosition) -> dict[str, Any]:
    return asdict(position)


def position_from_dict(payload: dict[str, Any]) -> PaperPosition:
    """Restore current or pre-upgrade session/database positions safely."""
    known = PaperPosition.__dataclass_fields__
    clean = {key: value for key, value in payload.items() if key in known}
    return PaperPosition(**clean)
