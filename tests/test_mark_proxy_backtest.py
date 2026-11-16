import pandas as pd

from backtesting.mark_proxy import run_mark_price_proxy_backtest


def test_mark_proxy_replays_only_past_marks_and_reports_exit_coverage():
    start = pd.Timestamp("2025-02-20T12:00:00Z")
    end = start + pd.Timedelta(hours=1)
    predictions = pd.DataFrame([{
        "prediction_time_utc": start,
        "hybrid_expected_return": 0.0,
        "hybrid_volatility": 0.001,
        "hybrid_return_q10": -0.004,
        "hybrid_return_q50": 0.0,
        "hybrid_return_q90": 0.004,
        "extreme_movement_probability": 0.05,
        "bearish_probability": 0.3,
        "sideways_probability": 0.4,
        "bullish_probability": 0.3,
    }])
    option_trades = pd.DataFrame([
        {"instrument_name": instrument, "timestamp": int((start - pd.Timedelta(seconds=30)).timestamp() * 1000),
         "mark_price": entry, "iv": 55.0}
        for instrument, entry in (("BTC-28FEB25-100000-C", .02), ("BTC-28FEB25-100000-P", .02))
    ] + [
        {"instrument_name": instrument, "timestamp": int((end - pd.Timedelta(seconds=30)).timestamp() * 1000),
         "mark_price": exit_price, "iv": 55.0}
        for instrument, exit_price in (("BTC-28FEB25-100000-C", .018), ("BTC-28FEB25-100000-P", .019))
    ])
    btc = pd.DataFrame([{"close_time_utc": start, "close": 100_000.0}])

    output, summary = run_mark_price_proxy_backtest(predictions, option_trades, btc)

    assert output.loc[0, "status"] == "selected"
    assert output.loc[0, "dynamic_outcome_status"] == "mark_exit_available"
    assert output.loc[0, "dynamic_realized_mark_pnl_btc"] > 0
    assert summary["selected_rows"] == 1
    assert summary["dynamic_outcomes"] == 1
    assert "not executable" in summary["entry_exit_basis"]
