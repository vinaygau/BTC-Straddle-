"""Run a reproducible historical-bootstrap BTC scenario experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from config.data_config import DataPipelineConfig
from models.expiry_forecaster import forecast_to_expiry
from models.interfaces import MarketPrediction
from models.market_regime_detector import REGIME_FEATURES, detect_market_regime


def main():
    config = DataPipelineConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expiry", required=True, help="Timezone-aware UTC option expiry, e.g. 2026-11-27T08:00:00Z")
    parser.add_argument("--artifact-dir", type=Path, default=config.processed_dir / "model_artifacts")
    parser.add_argument("--features", type=Path, default=config.processed_dir / "btc_multitimeframe_features.parquet")
    parser.add_argument("--candles", type=Path, default=config.processed_dir / "btc_5m.parquet")
    parser.add_argument("--simulations", type=int, default=1000)
    parser.add_argument("--block-length", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sensitivity-seeds", type=int, nargs="*", default=[],
                        help="Optional additional seeds for scenario distribution sensitivity")
    parser.add_argument("--output", type=Path,
                        default=config.reports_dir / "historical_scenario_runs" /
                        f"{pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%SZ')}.json")
    args = parser.parse_args()
    prediction_path = args.artifact_dir / "heldout_test_predictions.parquet"
    for path in (prediction_path, args.features, args.candles):
        if not path.is_file():
            parser.error(f"Required input not found: {path}")

    predictions = pd.read_parquet(prediction_path).sort_values("prediction_time_utc", kind="stable")
    row = predictions.iloc[-1]
    as_of = pd.Timestamp(row.prediction_time_utc)
    as_of = as_of.tz_localize("UTC") if as_of.tzinfo is None else as_of.tz_convert("UTC")
    prediction = MarketPrediction(
        as_of_utc=as_of.isoformat().replace("+00:00", "Z"),
        expected_return=float(row.hybrid_expected_return),
        return_quantiles={.1: float(row.hybrid_return_q10), .5: float(row.hybrid_return_q50),
                          .9: float(row.hybrid_return_q90)},
        predicted_volatility=float(row.hybrid_volatility),
        bearish_probability=float(row.bearish_probability),
        sideways_probability=float(row.sideways_probability),
        bullish_probability=float(row.bullish_probability),
        extreme_movement_probability=float(row.extreme_movement_probability),
        model_available=True,
        input_validation_status="validated from saved held-out prediction row",
    )
    feature_columns = ["prediction_time_utc", "5m_close", *REGIME_FEATURES]
    features = pd.read_parquet(args.features, columns=list(dict.fromkeys(feature_columns)))
    regime = detect_market_regime(features, as_of_utc=as_of)
    candles = pd.read_parquet(args.candles, columns=["close_time_utc", "close"])
    candles.close_time_utc = pd.to_datetime(candles.close_time_utc, utc=True, errors="coerce")
    known = candles.loc[candles.close_time_utc <= as_of].sort_values("close_time_utc", kind="stable")
    if known.empty:
        parser.error("No BTC reference candle is available at the saved prediction timestamp")
    spot = float(known.close.iloc[-1])
    result = forecast_to_expiry(
        reference_price=spot,
        as_of_utc=as_of,
        expiry_utc=args.expiry,
        prediction=prediction,
        historical_candles=candles,
        regime=regime,
        simulations=args.simulations,
        block_length=args.block_length,
        random_seed=args.seed,
        price_levels=(spot * .99, spot * 1.01),
    )
    report = {
        "scope": "simulated expiry price distribution; not a historical price observation or profit result",
        "prediction_timestamp_utc": result.as_of_utc,
        "expiry_utc": result.expiry_utc,
        "reference_price_usd": result.reference_price,
        "market_regime": result.regime,
        "historical_comparable_sample_count": result.historical_sample_count,
        "scenario_count": result.simulation_count,
        "block_length_5m_bars": result.block_length,
        "seed": result.random_seed,
        "conditioning_method": result.conditioning_method,
        "expiry_price_quantiles_usd": {str(k): v for k, v in result.terminal_price_quantiles.items()},
        "expiry_mean_usd": result.terminal_mean,
        "expiry_median_usd": result.terminal_median,
        "level_crossing_probabilities": {str(k): v for k, v in result.level_crossing_probabilities.items()},
        "extreme_move_threshold": result.extreme_move_threshold,
        "extreme_move_probability": result.extreme_move_probability,
        "seed_sensitivity": [],
        "warnings": list(result.warnings),
        "evidence_types": {"reference_price": "historical BTC candle at prediction time",
                           "forecast": "saved hybrid model prediction",
                           "expiry_prices": "moving-block simulated scenarios"},
    }
    for seed in dict.fromkeys(args.sensitivity_seeds):
        sensitivity = forecast_to_expiry(
            reference_price=spot,
            as_of_utc=as_of,
            expiry_utc=args.expiry,
            prediction=prediction,
            historical_candles=candles,
            regime=regime,
            simulations=args.simulations,
            block_length=args.block_length,
            random_seed=seed,
            price_levels=(spot * .99, spot * 1.01),
        )
        report["seed_sensitivity"].append({
            "seed": seed,
            "expiry_mean_usd": sensitivity.terminal_mean,
            "expiry_median_usd": sensitivity.terminal_median,
            "q05_usd": sensitivity.terminal_price_quantiles[.05],
            "q95_usd": sensitivity.terminal_price_quantiles[.95],
            "extreme_move_probability": sensitivity.extreme_move_probability,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
