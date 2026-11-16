from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np

from config.risk_config import RiskConfig
from models.interfaces import MarketPrediction
from options.contracts import OptionQuoteSnapshot
from strategies.dynamic_straddle import rank_dynamic_straddles


def test_dynamic_ranker_scores_only_fresh_complete_same_strike_pairs():
    now = datetime(2026, 11, 15, 12, 0, tzinfo=timezone.utc)
    expiry = (now + timedelta(days=7)).isoformat().replace("+00:00", "Z")
    quotes = [
        OptionQuoteSnapshot("BTC-C", "BTC", "CALL", 100_000, expiry, .05, .052, 2, 2, now,
                            implied_volatility=.55, delta=.5, gamma=.0001, settlement_currency="BTC"),
        OptionQuoteSnapshot("BTC-P", "BTC", "PUT", 100_000, expiry, .05, .052, 2, 2, now,
                            implied_volatility=.56, delta=-.5, gamma=.0001, settlement_currency="BTC"),
        OptionQuoteSnapshot("BTC-other", "BTC", "PUT", 99_000, expiry, .01, .012, 2, 2, now,
                            implied_volatility=.55, settlement_currency="BTC"),
    ]
    prediction = MarketPrediction(
        as_of_utc=now.isoformat(), expected_return=0.001,
        return_quantiles={.1: -.004, .5: .0, .9: .005}, predicted_volatility=.002,
        bullish_probability=.3, bearish_probability=.2, sideways_probability=.5,
        extreme_movement_probability=.03,
    )

    result = rank_dynamic_straddles(
        quotes, 100_000, prediction, now=now,
        risk=RiskConfig(cvar_penalty_lambda=1.0, minimum_bid_size=1.0),
    )

    assert result.candidates_seen == 1
    assert len(result.evaluated_candidates) == 1
    assert result.selected is not None
    assert result.selected.pair.call.strike == result.selected.pair.put.strike == 100_000
    assert result.selected.cvar95 >= 0
    assert result.selected.analysis.cvar95 == result.selected.cvar95
    assert result.execution_mode.startswith("research_only")


def test_dynamic_ranker_can_score_expiry_aligned_historical_scenarios():
    now = datetime(2026, 11, 15, 12, 0, tzinfo=timezone.utc)
    expiry = (now + timedelta(days=7)).isoformat().replace("+00:00", "Z")
    quotes = [
        OptionQuoteSnapshot("BTC-C", "BTC", "CALL", 100_000, expiry, .03, .031, 4, 4, now,
                            implied_volatility=.55, delta=.5, gamma=.0001, settlement_currency="BTC"),
        OptionQuoteSnapshot("BTC-P", "BTC", "PUT", 100_000, expiry, .03, .031, 4, 4, now,
                            implied_volatility=.55, delta=-.5, gamma=.0001, settlement_currency="BTC"),
    ]
    prediction = MarketPrediction(as_of_utc=now.isoformat(), expected_return=.001,
                                   return_quantiles={.1: -.004, .5: 0, .9: .004},
                                   predicted_volatility=.001)
    terminal = np.linspace(99_500, 100_500, 256)
    scenarios = SimpleNamespace(expiry_utc=expiry, reference_price=100_000,
                                terminal_prices=terminal)
    result = rank_dynamic_straddles(quotes, 100_000, prediction, now=now,
                                    expiry_scenarios={expiry: scenarios})
    assert result.decision == "SELECT_PAIR"
    assert result.selected is not None
    assert result.selected.scenario_source == "historical moving-block expiry scenarios"
    assert result.selected.probability_of_profit is not None
    assert result.selected.moneyness_fraction == 0


def test_dynamic_ranker_rejects_mismatched_expiry_scenarios():
    now = datetime(2026, 11, 15, 12, 0, tzinfo=timezone.utc)
    expiry = (now + timedelta(days=7)).isoformat().replace("+00:00", "Z")
    wrong_expiry = (now + timedelta(days=8)).isoformat().replace("+00:00", "Z")
    quotes = [
        OptionQuoteSnapshot("BTC-C", "BTC", "CALL", 100_000, expiry, .03, .031, 4, 4, now,
                            implied_volatility=.55, settlement_currency="BTC"),
        OptionQuoteSnapshot("BTC-P", "BTC", "PUT", 100_000, expiry, .03, .031, 4, 4, now,
                            implied_volatility=.55, settlement_currency="BTC"),
    ]
    prediction = MarketPrediction(as_of_utc=now.isoformat(), expected_return=0.0,
                                  predicted_volatility=.001)
    scenarios = SimpleNamespace(expiry_utc=wrong_expiry, reference_price=100_000,
                                terminal_prices=np.full(100, 100_000.0))
    result = rank_dynamic_straddles(quotes, 100_000, prediction, now=now,
                                    expiry_scenarios={expiry: scenarios})
    assert result.decision == "NO_TRADE"
    assert "does not match" in result.evaluated_candidates[0].rejection_reasons[0]
