import pandas as pd

from backtesting.straddle_backtester import run_mark_proxy_comparison
from options.historical_options_loader import load_historical_option_trades


def test_backtester_labels_mark_data_and_does_not_claim_executable_metrics():
    start = pd.Timestamp("2025-02-20T12:00:00Z")
    end = start + pd.Timedelta(hours=1)
    predictions = pd.DataFrame([{
        "prediction_time_utc": start,
        "hybrid_expected_return": 0.0,
        "hybrid_volatility": .001,
        "hybrid_return_q10": -.004,
        "hybrid_return_q50": 0.0,
        "hybrid_return_q90": .004,
        "extreme_movement_probability": .05,
        "bearish_probability": .3,
        "sideways_probability": .4,
        "bullish_probability": .3,
    }])
    tape = pd.DataFrame([
        {"instrument_name": instrument, "timestamp_utc": timestamp,
         "timestamp": int(timestamp.timestamp()*1000), "price": price,
         "amount": 1.0, "mark_price": price, "iv": 55.0}
        for timestamp, prices in ((start-pd.Timedelta(seconds=30), (.02,.02)),
                                  (end-pd.Timedelta(seconds=30), (.018,.019)))
        for instrument, price in zip(("BTC-28FEB25-100000-C", "BTC-28FEB25-100000-P"), prices)
    ])
    candles = pd.DataFrame([{"close_time_utc": start, "close": 100_000.0}])

    trades, report = run_mark_proxy_comparison(predictions, tape, candles)
    assert len(trades) == 1
    assert report["price_evidence"] == "historical executed trade print and mark price observations"
    assert report["confirmed_exchange_fills_for_strategy"] is False
    assert report["comparisons"]["nearest_atm_short_straddle"]["status"] == "mark_price_proxy_only"
    assert report["comparisons"]["nearest_atm_short_straddle"]["sharpe_ratio"] is None
    assert report["comparisons"]["historical_scenario_dynamic_strike"]["status"] == "not_evaluated"


def test_option_loader_does_not_misclassify_partial_quote_schema():
    tape = pd.DataFrame({"instrument_name": ["BTC-28FEB25-100000-C"],
                         "timestamp_utc": [pd.Timestamp("2025-02-01", tz="UTC")],
                         "price": [.02], "amount": [1.0], "mark_price": [.021], "iv": [55.0],
                         "bid": [.019]})
    loaded = load_historical_option_trades(tape)
    assert not loaded.has_historical_bid_ask
