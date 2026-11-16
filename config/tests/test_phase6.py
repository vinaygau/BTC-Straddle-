from execution.order_manager import LegState, OrderEvent, OrderStatus, StraddleLifecycle


def test_straddle_lifecycle_tracks_open_quantity_and_recovery():
    call_leg = LegState("BTC-25DEC27-50000-C", "SELL", requested_quantity=2.0)
    put_leg = LegState("BTC-25DEC27-50000-P", "SELL", requested_quantity=2.0)
    lifecycle = StraddleLifecycle(call_leg, put_leg)

    call_fill = OrderEvent(
        order_id="call-1",
        instrument_id="BTC-25DEC27-50000-C",
        side="SELL",
        status=OrderStatus.FILLED,
        quantity=2.0,
        filled_quantity=2.0,
        average_fill_price=95.0,
        timestamp="2026-11-14T00:00:00Z",
    )
    lifecycle.handle_event(call_fill)

    assert lifecycle.call_leg.open_quantity == 2.0
    assert lifecycle.matched_quantity == 0.0
    assert lifecycle.recovery_needed() is True

    put_fill = OrderEvent(
        order_id="put-1",
        instrument_id="BTC-25DEC27-50000-P",
        side="SELL",
        status=OrderStatus.FILLED,
        quantity=2.0,
        filled_quantity=2.0,
        average_fill_price=90.0,
        timestamp="2026-11-14T00:00:01Z",
    )
    lifecycle.handle_event(put_fill)

    assert lifecycle.matched_quantity == 2.0
    assert lifecycle.recovery_needed() is False
