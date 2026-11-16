import numpy as np
import pandas as pd
import pytest

from backtesting.walk_forward import expanding_walk_forward_folds
from models.dataset import chronological_splits
from models.historical_scenario_generator import generate_historical_scenarios
from models.interfaces import MarketPrediction
from models.market_regime_detector import SIMILARITY_FEATURES, detect_market_regime
from models.training_config import ModelTrainingConfig
from options.historical_options_loader import load_historical_option_trades


def _feature_history(rows=240):
    times = pd.date_range("2025-01-01", periods=rows, freq="5min", tz="UTC")
    frame = pd.DataFrame({"prediction_time_utc": times, "5m_close": 100_000 * np.exp(np.arange(rows) * .00001)})
    frame["5m_realized_volatility_6"] = .001
    frame["5m_log_return_1"] = .0001
    for name in SIMILARITY_FEATURES:
        if name == "5m_realized_volatility_12": frame[name] = .001
        elif name == "5m_momentum_return_12": frame[name] = .0005
        elif name == "5m_momentum_return_24": frame[name] = .0008
        elif name == "5m_body_to_range": frame[name] = .5
        elif name == "5m_volume_zscore_24": frame[name] = 0.0
        elif name == "5m_taker_buy_base_ratio": frame[name] = .5
    frame["5m_log_return_1"] = .0001
    return frame


def _candle_history(rows=1800):
    times = pd.date_range("2025-01-01", periods=rows, freq="5min", tz="UTC")
    rng = np.random.default_rng(321)
    returns = rng.normal(0, .001, rows)
    close = 100_000 * np.exp(np.cumsum(returns))
    return pd.DataFrame({"close_time_utc": times, "close": close})


def test_regime_matching_is_asof_causal_and_returns_only_matured_outcomes():
    frame = _feature_history()
    as_of = frame.prediction_time_utc.iloc[-1]
    earlier = frame.iloc[:180].copy()
    result_a = detect_market_regime(frame, as_of_utc=earlier.prediction_time_utc.iloc[-1], k=100, min_comparables=10)
    result_b = detect_market_regime(pd.concat([frame, _feature_history(260).iloc[240:]], ignore_index=True),
                                    as_of_utc=as_of, k=100, min_comparables=10)
    assert result_a.current_regime == "high_volatility_sideways"
    assert result_a.as_of_utc.endswith("Z")
    assert all(pd.Timestamp(t) < pd.Timestamp(result_a.as_of_utc) for t in result_a.matching_timestamps_utc)
    assert result_a.comparable_count > 0
    assert result_b.current_regime == result_a.current_regime


def test_historical_scenario_generator_is_reproducible_expiry_aligned_and_conditions_mean():
    history = _candle_history()
    as_of = history.close_time_utc.iloc[-1]
    expiry = as_of + pd.Timedelta(minutes=65)
    prediction = MarketPrediction(as_of_utc=as_of.isoformat(), expected_return=.001,
                                  return_quantiles={.1: -.004, .5: .001, .9: .006},
                                  predicted_volatility=.001)
    args = dict(reference_price=float(history.close.iloc[-1]), as_of_utc=as_of, expiry_utc=expiry,
                prediction=prediction, historical_candles=history, simulations=256,
                block_length=12, random_seed=7)
    first = generate_historical_scenarios(**args)
    same = generate_historical_scenarios(**args)
    other = generate_historical_scenarios(**{**args, "random_seed": 8})
    assert first.price_paths.shape == (256, 14)
    assert first.path_timestamps_utc[-1] == expiry.isoformat().replace("+00:00", "Z")
    np.testing.assert_array_equal(first.price_paths, same.price_paths)
    assert not np.array_equal(first.terminal_prices, other.terminal_prices)
    assert np.mean(np.log(first.price_paths[:, 12] / first.price_paths[:, 0])) == pytest.approx(np.log1p(.001), abs=1e-7)


def test_scenario_generator_rejects_invalid_expiry_and_insufficient_history():
    history = _candle_history(8)
    as_of = history.close_time_utc.iloc[-1]
    prediction = MarketPrediction(as_of_utc=as_of.isoformat(), expected_return=0.0)
    with pytest.raises(ValueError, match="after as_of"):
        generate_historical_scenarios(reference_price=100_000, as_of_utc=as_of, expiry_utc=as_of,
                                      prediction=prediction, historical_candles=history)
    with pytest.raises(ValueError, match="Insufficient historical"):
        generate_historical_scenarios(reference_price=100_000, as_of_utc=as_of,
                                      expiry_utc=as_of + pd.Timedelta(hours=1), prediction=prediction,
                                      historical_candles=history)


def test_scenario_generator_enforces_memory_bound():
    history = _candle_history(180)
    as_of = history.close_time_utc.iloc[-1]
    prediction = MarketPrediction(as_of_utc=as_of.isoformat(), expected_return=0.0,
                                  return_quantiles={.1: -.002, .5: 0.0, .9: .002})
    with pytest.raises(ValueError, match="path cells"):
        generate_historical_scenarios(reference_price=100_000, as_of_utc=as_of,
                                      expiry_utc=as_of + pd.Timedelta(hours=1),
                                      prediction=prediction, historical_candles=history,
                                      simulations=100, max_path_cells=1000)


def test_historical_options_loader_marks_tape_as_non_executable():
    rows = pd.DataFrame({"instrument_name": ["BTC-28FEB25-100000-C"],
                         "timestamp_utc": [pd.Timestamp("2025-02-01", tz="UTC")],
                         "price": [.02], "amount": [1.0], "mark_price": [.021], "iv": [55.0]})
    loaded = load_historical_option_trades(rows)
    assert loaded.row_count == 1
    assert loaded.source_kind == "deribit_executed_trade_tape_with_mark_price_proxy"
    assert not loaded.has_historical_bid_ask
    assert not loaded.executable_fill_evidence


def test_walk_forward_folds_are_chronological_and_purged():
    times = pd.date_range("2024-01-01", periods=150, freq="h", tz="UTC")
    folds = expanding_walk_forward_folds(times, initial_train_size=60, validation_size=20,
                                         test_size=20, step_size=20, purge_rows=3)
    assert len(folds) == 3
    for fold in folds:
        assert fold.train_indices[-1] + 3 < fold.validation_indices[0]
        assert fold.validation_indices[-1] + 3 < fold.test_indices[0]
        assert fold.train_indices[-1] < fold.validation_indices[0] < fold.test_indices[0]


def test_walk_forward_fold_boundaries_reproduce_existing_chronological_splitter():
    times = pd.date_range("2024-01-01", periods=150, freq="h", tz="UTC")
    fold = expanding_walk_forward_folds(times, initial_train_size=60, validation_size=20,
                                        test_size=20, step_size=20, purge_rows=3)[0]
    n = int(fold.test_indices[-1]) + 1
    train_boundary = int(fold.train_indices[-1]) + 1 + 3
    validation_boundary = int(fold.validation_indices[-1]) + 1 + 3
    config = ModelTrainingConfig(train_boundary_index=train_boundary,
                                 validation_boundary_index=validation_boundary,
                                 horizon_steps=3)
    frame = pd.DataFrame({"prediction_time_utc": times[:n]})
    split = chronological_splits(frame, (), config)
    np.testing.assert_array_equal(split.train, fold.train_indices)
    np.testing.assert_array_equal(split.validation, fold.validation_indices)
    np.testing.assert_array_equal(split.test, fold.test_indices)


def test_walk_forward_rejects_unsorted_or_too_short_timestamps():
    times = pd.date_range("2024-01-01", periods=15, freq="h", tz="UTC")
    with pytest.raises(ValueError, match="chronological"):
        expanding_walk_forward_folds(times[::-1], initial_train_size=5, validation_size=3, test_size=3)
    with pytest.raises(ValueError, match="Not enough"):
        expanding_walk_forward_folds(times, initial_train_size=10, validation_size=3, test_size=3)
