from datetime import datetime, timezone
from dataclasses import replace
import pytest

from options.contracts import OptionQuoteSnapshot
from options.option_pair_matcher import match_pairs
from options.combined_straddle_analyzer import analyze_short_straddle


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def quote(kind, strike=100, expiry="2026-02-01", bid=4, delta=None):
    return OptionQuoteSnapshot(kind, "BTC", kind, strike, expiry, bid, bid + 1, 2, 2, NOW,
                               quantity=1, multiplier=1, delta=delta, gamma=0.1, theta=-0.2, vega=0.3,
                               settlement_currency="BTC")


def test_matches_only_same_strike_and_expiry():
    call, put = quote("CALL"), quote("PUT")
    assert len(match_pairs([call, put], now=NOW)) == 1
    assert match_pairs([call, quote("PUT", strike=101)], now=NOW) == []
    assert match_pairs([call, quote("PUT", expiry="2026-03-01")], now=NOW) == []


def test_combined_credit_and_short_greeks():
    call, put = quote("CALL", bid=4, delta=0.6), quote("PUT", bid=5, delta=-0.4)
    pair = match_pairs([call, put], now=NOW)[0]
    result = analyze_short_straddle(pair, expected_buyback_cost=3, now=NOW)
    assert result.entry_credit == 9
    assert result.settlement_currency == "BTC"
    assert result.expected_pnl == 6
    assert result.delta == pytest.approx(-0.2)
    assert result.max_margin_requirement is None


def test_combined_expected_pnl_accounts_for_fees_and_slippage():
    pair = match_pairs([quote("CALL", bid=4), quote("PUT", bid=5)], now=NOW)[0]
    result = analyze_short_straddle(pair, expected_buyback_cost=3, fee_rate=0.01,
                                    slippage_rate=0.02, now=NOW)
    assert result.fees_estimate == pytest.approx(0.12)
    assert result.slippage_estimate == pytest.approx(0.24)
    assert result.expected_pnl == pytest.approx(5.64)


def test_combined_cvar_uses_worst_five_percent_of_all_scenarios():
    call, put = quote("CALL", delta=0), quote("PUT", delta=0)
    pair = match_pairs([call, put], now=NOW)[0]
    result = analyze_short_straddle(pair, pnl_samples=[10] * 19 + [-100], now=NOW)
    assert result.cvar95 == 100


def test_pair_requires_verified_matching_btc_settlement_currency():
    call, put = quote("CALL"), quote("PUT")
    pair = match_pairs([call, replace(put, settlement_currency="USDC")], now=NOW)
    assert pair == []


def test_combined_credit_respects_contract_multiplier_and_quantity():
    call = replace(quote("CALL", bid=2), multiplier=10, quantity=2)
    put = replace(quote("PUT", bid=3), multiplier=10, quantity=2)
    pair = match_pairs([call, put], now=NOW)[0]
    result = analyze_short_straddle(pair, now=NOW)
    assert result.entry_credit == 100


def test_deribit_standard_inverse_margin_uses_exchange_mark_prices():
    call = replace(quote("CALL", bid=4), mark_price=0.05)
    put = replace(quote("PUT", bid=5), mark_price=0.05)
    pair = match_pairs([call, put], now=NOW)[0]

    result = analyze_short_straddle(pair, reference_price=100, now=NOW)

    assert result.initial_margin_estimate == pytest.approx(0.40)
    assert result.maintenance_margin_estimate == pytest.approx(0.25)
    assert result.max_margin_requirement == pytest.approx(0.40)
    assert result.margin_model == "deribit_standard_inverse_estimate"


def test_margin_stays_unavailable_without_exchange_mark_price():
    pair = match_pairs([quote("CALL"), quote("PUT")], now=NOW)[0]
    result = analyze_short_straddle(pair, reference_price=100, now=NOW)
    assert result.initial_margin_estimate is None
    assert result.maintenance_margin_estimate is None
    assert result.max_margin_requirement is None
