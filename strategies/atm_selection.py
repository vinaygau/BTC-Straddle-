"""Phase 4 ATM-selection logic for BTC call/put pairing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class OptionContract:
    """Representative BTC option instrument metadata."""

    instrument_id: str
    option_type: str
    strike: float
    expiry: str
    is_active: bool = True
    bid: float | None = None
    ask: float | None = None
    min_trade_amount: float | None = None
    tick_size: float | None = None
    tick_size_steps: tuple[tuple[float, float], ...] = ()
    multiplier: float | None = None
    settlement_currency: str | None = None
    quote_currency: str | None = None

    def is_valid(self) -> bool:
        if not self.is_active:
            return False
        if self.strike <= 0:
            return False
        if self.option_type.upper() not in {"CALL", "PUT"}:
            return False
        return True


def nearest_atm_strike(reference_price: float, strikes: Iterable[float]) -> float:
    """Return the closest eligible strike to the BTC reference price."""
    values = sorted(float(strike) for strike in strikes if float(strike) > 0)
    if not values:
        raise ValueError("No valid strikes were supplied.")

    return min(values, key=lambda strike: (abs(strike - reference_price), strike))


def select_atm_pair(
    reference_price: float,
    calls: Iterable[OptionContract],
    puts: Iterable[OptionContract],
    *,
    expiry_policy: str = "nearest_future_expiry",
    tie_breaker: str = "lower",
) -> tuple[OptionContract, OptionContract]:
    """Find the nearest ATM Call/Put pair with identical strike and expiry.

    The selection is deterministic and matches the assignment’s requirement that
    both legs share exactly the same strike and expiry.
    """
    call_list = [contract for contract in calls if contract.is_valid()]
    put_list = [contract for contract in puts if contract.is_valid()]

    if not call_list or not put_list:
        raise ValueError("Both calls and puts must be available for ATM selection.")

    if expiry_policy not in {"nearest_future_expiry"}:
        raise ValueError("Unsupported expiry policy: expected 'nearest_future_expiry'.")
    if tie_breaker not in {"lower", "higher"}:
        raise ValueError("tie_breaker must be 'lower' or 'higher'.")

    best_pair: tuple[OptionContract, OptionContract] | None = None
    best_distance: float | None = None
    best_expiry: str | None = None

    for call in call_list:
        for put in put_list:
            if call.expiry != put.expiry:
                continue
            if call.strike != put.strike:
                continue
            distance = abs(call.strike - reference_price)
            candidate = (distance, call.expiry, call.strike)
            if best_distance is None or candidate[0] < best_distance:
                best_distance = candidate[0]
                best_expiry = call.expiry
                best_pair = (call, put)
            elif candidate[0] == best_distance and best_expiry is not None:
                if tie_breaker == "lower" and call.strike < best_pair[0].strike:
                    best_pair = (call, put)
                    best_expiry = call.expiry
                elif tie_breaker == "higher" and call.strike > best_pair[0].strike:
                    best_pair = (call, put)
                    best_expiry = call.expiry

    if best_pair is None:
        raise ValueError("No matching ATM call/put pair exists for the selected expiry.")

    return best_pair
