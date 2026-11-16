"""Order-management primitives for the Phase 1 BTC options workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class StrategyState(str, Enum):
    PREPARING = "preparing"
    RECONCILIATION_REQUIRED = "reconciliation_required"
    ENTRY_WORKING = "entry_working"
    PARTIALLY_OPEN = "partially_open"
    OPEN = "open"
    EXITING = "exiting"
    CLOSED = "closed"
    ENTRY_FAILED = "entry_failed"


class OrderStatus(str, Enum):
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass(frozen=True)
class OrderRequest:
    """Represents an exchange order request."""

    instrument_id: str
    side: str
    quantity: float
    order_type: str = "limit"
    limit_price: float | None = None

    def validate(self) -> None:
        if not self.instrument_id:
            raise ValueError("instrument_id is required.")

        if self.side.upper() not in {"BUY", "SELL"}:
            raise ValueError("side must be BUY or SELL.")

        if self.quantity <= 0:
            raise ValueError("quantity must be greater than zero.")

        if self.order_type.lower() == "limit" and self.limit_price is not None:
            if self.limit_price <= 0:
                raise ValueError("limit_price must be positive when provided.")


@dataclass(frozen=True)
class OrderEvent:
    """Venue-side event for an order lifecycle."""

    order_id: str
    instrument_id: str
    side: str
    status: OrderStatus
    quantity: float
    filled_quantity: float = 0.0
    average_fill_price: float | None = None
    timestamp: str | None = None

    @property
    def is_executed(self) -> bool:
        return self.status in {OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED}


@dataclass
class LegState:
    """Tracks the lifecycle of one option leg and its resulting open exposure."""

    instrument_id: str
    side: str
    order_id: str | None = None
    requested_quantity: float = 0.0
    filled_quantity: float = 0.0
    open_quantity: float = 0.0
    status: OrderStatus = OrderStatus.SUBMITTED
    average_fill_price: float | None = None
    last_event_time: str | None = None

    @property
    def remaining_quantity(self) -> float:
        return max(self.requested_quantity - self.filled_quantity, 0.0)

    def apply_event(self, event: OrderEvent) -> None:
        if event.instrument_id != self.instrument_id:
            raise ValueError("Event instrument_id does not match the leg state.")

        self.order_id = event.order_id
        self.status = event.status
        self.last_event_time = event.timestamp

        if event.is_executed:
            self.filled_quantity = max(self.filled_quantity, event.filled_quantity)
            if event.average_fill_price is not None:
                self.average_fill_price = event.average_fill_price
            self.open_quantity = self.filled_quantity

        if event.status == OrderStatus.CANCELLED:
            self.open_quantity = 0.0
        elif event.status == OrderStatus.REJECTED:
            self.open_quantity = 0.0


@dataclass
class StraddleLifecycle:
    """Coherent tracking for the short-straddle Call + Put legs."""

    call_leg: LegState
    put_leg: LegState
    entries: dict[str, LegState] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.entries = {
            self.call_leg.instrument_id: self.call_leg,
            self.put_leg.instrument_id: self.put_leg,
        }

    def handle_event(self, event: OrderEvent) -> LegState:
        if event.instrument_id not in self.entries:
            raise ValueError(f"No tracked leg for instrument {event.instrument_id}.")

        leg = self.entries[event.instrument_id]
        leg.apply_event(event)
        return leg

    @property
    def matched_quantity(self) -> float:
        return min(self.call_leg.open_quantity, self.put_leg.open_quantity)

    @property
    def has_unmatched_exposure(self) -> bool:
        return (
            (self.call_leg.open_quantity > 0 and self.put_leg.open_quantity == 0)
            or (self.put_leg.open_quantity > 0 and self.call_leg.open_quantity == 0)
            or (self.call_leg.open_quantity != self.put_leg.open_quantity)
        )

    def recovery_needed(self) -> bool:
        return self.has_unmatched_exposure and (
            self.call_leg.open_quantity > 0 or self.put_leg.open_quantity > 0
        )

    def exposure_summary(self) -> dict[str, float]:
        return {
            "call_open": self.call_leg.open_quantity,
            "put_open": self.put_leg.open_quantity,
            "matched_quantity": self.matched_quantity,
            "unmatched_exposure": max(
                abs(self.call_leg.open_quantity - self.put_leg.open_quantity),
                0.0,
            ),
        }


@dataclass
class StrategyStateMachine:
    """Stateful execution manager for single-cycle BTC short-straddle trading."""

    call_leg: LegState
    put_leg: LegState
    state: StrategyState = StrategyState.PREPARING
    entry_premium: float = 0.0
    current_buyback_premium: float | None = None
    exit_requested: bool = False
    trigger_reason: str | None = None
    history: list[tuple[StrategyState, StrategyState]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.lifecycle = StraddleLifecycle(self.call_leg, self.put_leg)

    @property
    def matched_quantity(self) -> float:
        return self.lifecycle.matched_quantity

    @property
    def has_unmatched_exposure(self) -> bool:
        return self.lifecycle.has_unmatched_exposure

    def _transition(self, new_state: StrategyState) -> StrategyState:
        if new_state != self.state:
            self.history.append((self.state, new_state))
            self.state = new_state
        return self.state

    def _evaluate_state(self) -> StrategyState:
        if self.call_leg.status == OrderStatus.REJECTED or self.put_leg.status == OrderStatus.REJECTED:
            return StrategyState.ENTRY_FAILED

        if self.exit_requested and self.call_leg.open_quantity == 0 and self.put_leg.open_quantity == 0:
            return StrategyState.CLOSED

        if self.call_leg.filled_quantity > 0 or self.put_leg.filled_quantity > 0:
            if self.call_leg.open_quantity > 0 and self.put_leg.open_quantity > 0:
                if self.call_leg.open_quantity == self.put_leg.open_quantity:
                    return StrategyState.OPEN
            return StrategyState.PARTIALLY_OPEN

        if self.call_leg.status in {OrderStatus.SUBMITTED, OrderStatus.ACCEPTED} or self.put_leg.status in {
            OrderStatus.SUBMITTED,
            OrderStatus.ACCEPTED,
        }:
            return StrategyState.ENTRY_WORKING

        if self.call_leg.open_quantity == 0 and self.put_leg.open_quantity == 0:
            return StrategyState.RECONCILIATION_REQUIRED

        if self.call_leg.open_quantity > 0 and self.put_leg.open_quantity > 0:
            if self.call_leg.open_quantity == self.put_leg.open_quantity:
                return StrategyState.OPEN
            return StrategyState.PARTIALLY_OPEN

        if self.lifecycle.recovery_needed():
            return StrategyState.PARTIALLY_OPEN

        return StrategyState.RECONCILIATION_REQUIRED

    def update_state(self) -> StrategyState:
        return self._transition(self._evaluate_state())

    def handle_event(self, event: OrderEvent) -> LegState:
        leg = self.lifecycle.handle_event(event)

        if event.is_executed and event.average_fill_price is not None:
            self.entry_premium += event.filled_quantity * event.average_fill_price

        self.update_state()
        return leg

    def begin_entry(self) -> StrategyState:
        self.exit_requested = False
        self.trigger_reason = None
        return self._transition(StrategyState.ENTRY_WORKING)

    def mark_reconciliation_required(self) -> StrategyState:
        return self._transition(StrategyState.RECONCILIATION_REQUIRED)

    def begin_exit(self, reason: str) -> StrategyState:
        self.exit_requested = True
        self.trigger_reason = reason
        return self._transition(StrategyState.EXITING)

    def mark_closed(self) -> StrategyState:
        self.exit_requested = False
        self.trigger_reason = None
        return self._transition(StrategyState.CLOSED)

    def reconcile_position(self) -> StrategyState:
        if self.call_leg.open_quantity == 0 and self.put_leg.open_quantity == 0:
            return self.mark_closed()
        return self.mark_reconciliation_required()


def validate_order_request(order: OrderRequest) -> None:
    """Validate an order request before it is submitted to the exchange."""
    order.validate()
