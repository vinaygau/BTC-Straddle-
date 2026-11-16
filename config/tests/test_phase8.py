from execution.order_manager import (
    LegState,
    OrderEvent,
    OrderStatus,
    StrategyState,
    StrategyStateMachine,
)


def test_strategy_state_machine_tracks_open_cycle():
    call_leg = LegState("BTC-25DEC27-50000-C", "SELL", requested_quantity=2.0)
    put_leg = LegState("BTC-25DEC27-50000-P", "SELL", requested_quantity=2.0)
    machine = StrategyStateMachine(call_leg, put_leg)

    assert machine.begin_entry() == StrategyState.ENTRY_WORKING

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

    machine.handle_event(call_fill)
    machine.handle_event(put_fill)

    assert machine.state == StrategyState.OPEN
    assert machine.matched_quantity == 2.0
    assert machine.entry_premium == 370.0


def test_strategy_state_machine_flags_partial_open_exposure():
    call_leg = LegState("BTC-25DEC27-50000-C", "SELL", requested_quantity=2.0)
    put_leg = LegState("BTC-25DEC27-50000-P", "SELL", requested_quantity=2.0)
    machine = StrategyStateMachine(call_leg, put_leg)

    call_fill = OrderEvent(
        order_id="call-1",
        instrument_id="BTC-25DEC27-50000-C",
        side="SELL",
        status=OrderStatus.FILLED,
        quantity=2.0,
        filled_quantity=2.0,
        average_fill_price=100.0,
        timestamp="2026-11-14T00:00:00Z",
    )

    machine.handle_event(call_fill)

    assert machine.state == StrategyState.PARTIALLY_OPEN
    assert machine.has_unmatched_exposure is True


def test_strategy_state_machine_handles_exit_and_closed_cycle():
    call_leg = LegState("BTC-25DEC27-50000-C", "SELL", requested_quantity=2.0)
    put_leg = LegState("BTC-25DEC27-50000-P", "SELL", requested_quantity=2.0)
    machine = StrategyStateMachine(call_leg, put_leg)

    machine.begin_exit("stop_loss")
    assert machine.state == StrategyState.EXITING

    call_leg.open_quantity = 0.0
    put_leg.open_quantity = 0.0

    assert machine.reconcile_position() == StrategyState.CLOSED
    assert machine.state == StrategyState.CLOSED
