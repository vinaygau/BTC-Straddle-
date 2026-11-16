from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd

from config.risk_config import RiskConfig
from models.candlestick_analysis import analyze_candlesticks
from models.interfaces import MarketPrediction
from options.contracts import OptionQuoteSnapshot
from strategies.dynamic_straddle import rank_dynamic_straddles


NOW = datetime(2026, 11, 16, 12, 0, tzinfo=timezone.utc)


def _candles(*, breakout: bool) -> pd.DataFrame:
    times = pd.date_range(end=pd.Timestamp(NOW), periods=21, freq="5min")
    rows = []
    for stamp in times[:-1]:
        rows.append({"close_time_utc": stamp, "open": 100.0, "high": 101.0,
                     "low": 99.0, "close": 100.0, "volume": 10.0})
    final = {"close_time_utc": times[-1], "open": 100.0, "high": 103.0 if breakout else 101.0,
             "low": 99.8 if breakout else 99.0, "close": 102.8 if breakout else 100.2,
             "volume": 20.0 if breakout else 10.0}
    rows.append(final)
    return pd.DataFrame(rows)


def test_live_candle_analysis_uses_closed_candles_and_flags_breakout_risk():
    frame = _candles(breakout=True)
    future = frame.iloc[-1].copy()
    future["close_time_utc"] = pd.Timestamp(NOW) + pd.Timedelta(minutes=5)
    future["close"] = 150.0
    history = pd.concat([frame, future.to_frame().T], ignore_index=True)

    result = analyze_candlesticks({"5m": history}, as_of_utc=NOW.isoformat(),
                                  config={"lookback": 20, "veto_risk_score": 2})

    five_minute = next(row for row in result.timeframes if row.timeframe == "5m")
    assert result.risk_veto
    assert "range_breakout_up" in five_minute.patterns
    assert five_minute.last_close == 102.8
    assert "insufficient_completed_candles" not in five_minute.patterns


def test_candlestick_veto_rejects_straddle_candidates_even_with_positive_scenario():
    expiry = (NOW + timedelta(days=2)).isoformat().replace("+00:00", "Z")
    quotes = [
        OptionQuoteSnapshot("BTC-C", "BTC", "CALL", 100_000, expiry, .04, .041, 2, 2, NOW,
                            implied_volatility=.5, settlement_currency="BTC"),
        OptionQuoteSnapshot("BTC-P", "BTC", "PUT", 100_000, expiry, .04, .041, 2, 2, NOW,
                            implied_volatility=.5, settlement_currency="BTC"),
    ]
    prediction = MarketPrediction(as_of_utc=NOW.isoformat(), expected_return=.001,
                                  predicted_volatility=.001)
    scenario = SimpleNamespace(expiry_utc=expiry, reference_price=100_000,
                               terminal_prices=np.full(500, 100_000.0))

    result = rank_dynamic_straddles(
        quotes, 100_000, prediction, now=NOW,
        risk=RiskConfig(max_candlestick_risk_score=2),
        expiry_scenarios={expiry: scenario}, candlestick_risk_score=2,
    )

    assert result.decision == "NO_TRADE"
    assert "candlestick risk score" in result.evaluated_candidates[0].rejection_reasons[-1]
