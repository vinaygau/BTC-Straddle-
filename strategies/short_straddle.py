"""Phase 5 short-straddle entry workflow and order validation."""

from __future__ import annotations

from dataclasses import dataclass

from execution.order_manager import OrderRequest, OrderStatus, validate_order_request


@dataclass(frozen=True)
class StraddleLeg:
    """Single side of the short straddle."""

    instrument_id: str
    option_type: str
    strike: float
    expiry: str
    quantity: float = 1.0

    def validate(self) -> None:
        if not self.instrument_id:
            raise ValueError("instrument_id is required.")
        if self.option_type.upper() not in {"CALL", "PUT"}:
            raise ValueError("option_type must be CALL or PUT.")
        if self.strike <= 0:
            raise ValueError("strike must be positive.")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive.")


@dataclass(frozen=True)
class ShortStraddle:
    """Represents a valid Call + Put pair that forms a short straddle."""

    call: StraddleLeg
    put: StraddleLeg
    order_quantity: float = 1.0
    strategy_state: str = "PREPARING"

    def validate(self) -> None:
        self.call.validate()
        self.put.validate()

        if self.call.option_type.upper() != "CALL":
            raise ValueError("call leg must be a CALL instrument.")
        if self.put.option_type.upper() != "PUT":
            raise ValueError("put leg must be a PUT instrument.")
        if self.call.strike != self.put.strike:
            raise ValueError("Call and Put strikes must match.")
        if self.call.expiry != self.put.expiry:
            raise ValueError("Call and Put expiries must match.")
        if self.order_quantity <= 0:
            raise ValueError("order_quantity must be positive.")
        if self.call.quantity != self.put.quantity:
            raise ValueError("Both short-straddle legs must use the same quantity.")
        if self.call.quantity != self.order_quantity:
            raise ValueError("Leg quantity and order quantity must match.")

    def build_sell_orders(self) -> tuple[OrderRequest, OrderRequest]:
        """Build the call and put sell orders for the straddle entry."""
        call_order = OrderRequest(
            instrument_id=self.call.instrument_id,
            side="SELL",
            quantity=self.order_quantity,
            order_type="limit",
            limit_price=None,
        )
        put_order = OrderRequest(
            instrument_id=self.put.instrument_id,
            side="SELL",
            quantity=self.order_quantity,
            order_type="limit",
            limit_price=None,
        )

        validate_order_request(call_order)
        validate_order_request(put_order)
        return call_order, put_order


@dataclass(frozen=True)
class LegExecution:
    """Execution record for one straddle leg."""

    instrument_id: str
    side: str
    filled_quantity: float
    fill_price: float
    status: OrderStatus

    def is_valid(self) -> bool:
        return self.filled_quantity > 0 and self.fill_price > 0


def create_short_straddle(
    call: StraddleLeg,
    put: StraddleLeg,
    *,
    order_quantity: float | None = None,
) -> ShortStraddle:
    """Create a valid short straddle after checking both legs align."""
    quantity = order_quantity if order_quantity is not None else call.quantity
    straddle = ShortStraddle(
        call=call,
        put=put,
        order_quantity=quantity,
    )
    straddle.validate()
    return straddle


def compute_entry_premium(fill_records: list[LegExecution]) -> float:
    """Calculate total premium received from actual fills."""
    total = 0.0
    for record in fill_records:
        if not record.is_valid():
            continue
        total += record.filled_quantity * record.fill_price
    return total
