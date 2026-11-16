from execution.order_manager import OrderStatus
from strategies.short_straddle import LegExecution, ShortStraddle, StraddleLeg, compute_entry_premium, create_short_straddle


def test_short_straddle_builds_sell_orders_for_both_legs():
    call = StraddleLeg("BTC-25DEC27-50000-C", "CALL", 50000.0, "2027-12-25", quantity=2.0)
    put = StraddleLeg("BTC-25DEC27-50000-P", "PUT", 50000.0, "2027-12-25", quantity=2.0)

    straddle = create_short_straddle(call, put, order_quantity=2.0)
    call_order, put_order = straddle.build_sell_orders()

    assert call_order.side == "SELL"
    assert put_order.side == "SELL"
    assert call_order.quantity == 2.0
    assert put_order.quantity == 2.0


def test_entry_premium_uses_actual_filled_quantity_and_price():
    fills = [
        LegExecution("BTC-25DEC27-50000-C", "SELL", 1.0, 95.0, OrderStatus.FILLED),
        LegExecution("BTC-25DEC27-50000-P", "SELL", 1.0, 90.0, OrderStatus.FILLED),
    ]

    assert compute_entry_premium(fills) == 185.0
