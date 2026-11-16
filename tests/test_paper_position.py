from datetime import datetime, timedelta, timezone

import pytest

from config.paper_trading import load_paper_trading_config
from data.market_data import OptionQuote
from execution.paper_store import PaperTradeStore
from execution.paper_trader import (
    evaluate_position_risk,
    quote_status,
    risk_thresholds,
    simulate_entry,
    simulate_exit,
    value_position,
)


NOW = datetime(2026, 11, 16, 12, 0, tzinfo=timezone.utc)
EXPIRY = (NOW + timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ")


def quote(option_type, *, bid, ask, bid_size=1.0, ask_size=1.0, timestamp=None,
          multiplier=1.0, quote_currency="BTC", settlement_currency="BTC", expiry=EXPIRY):
    return OptionQuote(
        instrument_id=f"BTC-16NOV26-90000-{option_type[0]}", option_type=option_type,
        strike=90_000, expiry=expiry, bid=bid, ask=ask, bid_size=bid_size,
        ask_size=ask_size, timestamp=(timestamp or NOW).strftime("%Y-%m-%dT%H:%M:%SZ"),
        multiplier=multiplier, quote_currency=quote_currency,
        settlement_currency=settlement_currency,
    )


def config(*, fee=0.0, slippage=0.0):
    settings = load_paper_trading_config()
    settings = {key: dict(value) if isinstance(value, dict) else value for key, value in settings.items()}
    settings["accounting"] = dict(settings["accounting"])
    settings["accounting"]["fee_rate_per_side"] = fee
    settings["accounting"]["slippage_rate_per_side"] = slippage
    return settings


def make_position(*, call_size=1.0, put_size=1.0, amount=0.1, cfg=None):
    call = quote("CALL", bid=0.02, ask=0.021, bid_size=call_size)
    put = quote("PUT", bid=0.03, ask=0.04, bid_size=put_size)
    return simulate_entry(call, put, amount, config=cfg or config(), now=NOW), call, put


def test_entry_uses_matched_bid_liquidity_and_tracks_each_leg():
    position, call, put = make_position(call_size=0.06, put_size=0.09, amount=0.1)
    assert position.status == "PARTIALLY_FILLED"
    assert position.open_amount == pytest.approx(0.06)
    assert position.call_entry_price == pytest.approx(0.02)
    assert position.put_entry_price == pytest.approx(0.03)
    assert position.call_instrument_id == call.instrument_id
    assert position.put_instrument_id == put.instrument_id
    assert position.quote_currency == position.settlement_currency == "BTC"


def test_full_close_calculates_gross_and_net_pnl_in_btc():
    position, _, _ = make_position(amount=0.1)
    call = quote("CALL", bid=0.0205, ask=0.021, ask_size=0.1)
    put = quote("PUT", bid=0.031, ask=0.04, ask_size=0.1)
    valuation = value_position(position, call, put, config=config(), now=NOW)
    assert valuation.call_gross_pnl == pytest.approx(-0.0001)
    assert valuation.put_gross_pnl == pytest.approx(-0.001)
    assert valuation.gross_unrealized_pnl == pytest.approx(-0.0011)
    assert valuation.net_unrealized_pnl == pytest.approx(-0.0011)
    closed, amount, pnl = simulate_exit(position, call, put, config=config(), now=NOW)
    assert amount == pytest.approx(0.1)
    assert closed.open_amount == pytest.approx(0.0)
    assert closed.status == "CLOSED"
    assert pnl == pytest.approx(-0.0011)


def test_partial_close_preserves_matched_remaining_amount():
    position, _, _ = make_position(amount=0.1)
    call = quote("CALL", bid=0.02, ask=0.021, ask_size=0.03)
    put = quote("PUT", bid=0.03, ask=0.04, ask_size=0.07)
    updated, closed, _ = simulate_exit(position, call, put, config=config(), now=NOW)
    assert closed == pytest.approx(0.03)
    assert updated.call_open_amount == updated.put_open_amount == pytest.approx(0.07)
    assert updated.status == "PARTIALLY_CLOSED"


def test_entry_and_close_fees_follow_configured_rate_and_premium_cap():
    cfg = config(fee=0.0003)
    position, _, _ = make_position(amount=0.1, cfg=cfg)
    # 3 bps of BTC amount on each leg, capped by 12.5% of option premium.
    assert position.entry_fees == pytest.approx(0.00006)
    call = quote("CALL", bid=0.02, ask=0.021, ask_size=0.1)
    put = quote("PUT", bid=0.03, ask=0.04, ask_size=0.1)
    valuation = value_position(position, call, put, config=cfg, now=NOW)
    assert valuation.estimated_exit_fees == pytest.approx(0.00006)
    assert valuation.net_unrealized_pnl == pytest.approx(-0.00122)


def test_stale_missing_crossed_and_insufficient_metadata_quotes_are_rejected():
    stale_call = quote("CALL", bid=0.02, ask=0.021, timestamp=NOW - timedelta(seconds=9))
    put = quote("PUT", bid=0.03, ask=0.04)
    assert quote_status(stale_call, max_age_seconds=5, now=NOW)[0] == "Stale"
    with pytest.raises(ValueError, match="stale"):
        simulate_entry(stale_call, put, 0.1, config=config(), now=NOW)
    crossed = quote("CALL", bid=0.03, ask=0.02)
    with pytest.raises(ValueError, match="invalid"):
        simulate_entry(crossed, put, 0.1, config=config(), now=NOW)
    unknown_multiplier = quote("CALL", bid=0.02, ask=0.021, multiplier=None)
    with pytest.raises(ValueError, match="multiplier is unavailable"):
        simulate_entry(unknown_multiplier, put, 0.1, config=config(), now=NOW)
    wrong_currency = quote("CALL", bid=0.02, ask=0.021, settlement_currency=None)
    with pytest.raises(ValueError, match="settlement currency"):
        simulate_entry(wrong_currency, put, 0.1, config=config(), now=NOW)
    wrong_multiplier = quote("CALL", bid=0.02, ask=0.021, multiplier=2.0)
    with pytest.raises(ValueError, match="unsupported"):
        simulate_entry(wrong_multiplier, put, 0.1, config=config(), now=NOW)
    wrong_quote_currency = quote("CALL", bid=0.02, ask=0.021, quote_currency="USD")
    with pytest.raises(ValueError, match="quote currency"):
        simulate_entry(wrong_quote_currency, put, 0.1, config=config(), now=NOW)


def test_missing_quote_fields_and_zero_depth_are_not_executable():
    missing_ask = quote("CALL", bid=0.02, ask=None)
    missing_size = quote("CALL", bid=0.02, ask=0.021, ask_size=None)
    zero_size = quote("CALL", bid=0.02, ask=0.021, bid_size=0.0)
    assert quote_status(missing_ask, now=NOW)[0] == "Unavailable"
    assert quote_status(missing_size, now=NOW)[0] == "Unavailable"
    call = quote("CALL", bid=0.02, ask=0.021, bid_size=0.0)
    put = quote("PUT", bid=0.03, ask=0.04)
    with pytest.raises(ValueError, match="no matched amount"):
        simulate_entry(call, put, 0.1, config=config(), now=NOW)
    assert quote_status(zero_size, now=NOW)[0] == "Fresh"


def test_expired_contract_is_not_entered_or_closed_at_last_ask():
    expired = (NOW - timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    call = quote("CALL", bid=0.02, ask=0.021, expiry=expired)
    put = quote("PUT", bid=0.03, ask=0.04, expiry=expired)
    with pytest.raises(ValueError, match="expired"):
        simulate_entry(call, put, 0.1, config=config(), now=NOW)


def test_stop_threshold_uses_configured_percentage_of_position_credit():
    position, _, _ = make_position(amount=0.1)
    limits = risk_thresholds(position, config())
    assert limits["maximum_loss"] == pytest.approx(0.001)
    assert limits["target_profit"] == pytest.approx(0.0015)


@pytest.mark.parametrize(
    ("call_ask", "put_ask", "expected"),
    [(0.03, 0.04, "STOP_LOSS_SIGNAL"), (0.02, 0.03, "WITHIN_LIMITS"), (0.01, 0.02, "TAKE_PROFIT_SIGNAL")],
)
def test_net_pnl_drives_stop_loss_and_take_profit(call_ask, put_ask, expected):
    position, _, _ = make_position(amount=0.1)
    call = quote("CALL", bid=min(0.02, call_ask), ask=call_ask, ask_size=0.1)
    put = quote("PUT", bid=min(0.03, put_ask), ask=put_ask, ask_size=0.1)
    valuation = value_position(position, call, put, config=config(), now=NOW)
    risk = evaluate_position_risk(position, valuation, config())
    assert risk["status"] == expected


def test_sqlite_store_recovers_active_position_and_keeps_event_history(tmp_path):
    position, _, _ = make_position(amount=0.1)
    store = PaperTradeStore(tmp_path / "paper.sqlite3")
    store.save_position(position)
    store.append_event("Position opened", position.position_id, {"reason": "test"})
    reopened = PaperTradeStore(tmp_path / "paper.sqlite3")
    assert reopened.load_active_position() == position
    assert reopened.list_events()[0]["Event"] == "Position opened"
