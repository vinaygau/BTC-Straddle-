"""Offline inference adapter that blends the trained LSTM and XGBoost branches."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .interfaces import MarketPrediction, MarketPredictionProvider
from .lstm_inference import LSTMMarketPredictionProvider


class HybridMarketPredictionProvider(MarketPredictionProvider):
    """Load a hybrid artifact set and produce a point-in-time BTC prediction.

    This provider is inference-only. It does not choose options or connect to
    execution code. The supplied candles must end at a fully closed 5m candle.
    """

    def __init__(self, artifact_dir: str | Path, *, device: str = "cpu") -> None:
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise RuntimeError("Hybrid inference requires the CPU XGBoost package.") from exc

        self._lstm = LSTMMarketPredictionProvider(artifact_dir, device=device)
        directory = Path(artifact_dir)
        report_path = directory / "training_report.json"
        if not report_path.is_file():
            raise FileNotFoundError(f"Missing hybrid training report in {directory}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("model_mode") != "hybrid":
            raise ValueError("Artifacts are not from a completed hybrid training run.")
        weights = report.get("validation_blend_weights") or {}
        self._xgb_weight = float(weights.get("xgboost", np.nan))
        self._lstm_weight = float(weights.get("lstm", np.nan))
        if (not np.isfinite([self._xgb_weight, self._lstm_weight]).all()
                or self._xgb_weight < 0 or self._lstm_weight < 0
                or not np.isclose(self._xgb_weight + self._lstm_weight, 1.0, atol=1e-6)):
            raise ValueError("Training report contains invalid hybrid blend weights.")
        offsets = report.get("return_quantile_residual_offsets")
        if not isinstance(offsets, list) or len(offsets) != 3 or not np.isfinite(offsets).all():
            raise ValueError("Training report is missing return-quantile calibration offsets.")
        self._quantile_offsets = np.asarray(offsets, dtype=np.float64)

        self._boosters = {}
        for name in ("return", "volatility", "direction", "extreme"):
            path = directory / f"xgboost_{name}.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing XGBoost {name} model: {path}")
            booster = xgb.Booster()
            booster.load_model(path)
            self._boosters[name] = booster
        self._xgb = xgb

    def predict(self, completed_candles_by_timeframe: dict[str, object], *, as_of_utc: str) -> MarketPrediction:
        as_of, sequence, latest = self._lstm.prepare_input(
            completed_candles_by_timeframe, as_of_utc=as_of_utc
        )
        base = completed_candles_by_timeframe["5m"].copy()
        base["timestamp_utc"] = pd.to_datetime(base["timestamp_utc"], utc=True, errors="coerce")
        if "close_time_utc" in base:
            available = pd.to_datetime(base["close_time_utc"], utc=True, errors="coerce")
        else:
            available = base["timestamp_utc"] + pd.Timedelta(minutes=5) - pd.Timedelta(microseconds=1)
        base = base.loc[available <= as_of].sort_values("timestamp_utc", kind="stable")
        reference_price = float(base.close.iloc[-1])
        lstm_prediction = self._lstm.predict_prepared(as_of, sequence)
        matrix = self._xgb.DMatrix(latest.reshape(1, -1))

        xgb_return = float(self._boosters["return"].predict(matrix)[0])
        xgb_volatility = float(self._boosters["volatility"].predict(matrix)[0])
        direction = np.asarray(self._boosters["direction"].predict(matrix), dtype=np.float64).reshape(-1)
        extreme = float(np.asarray(self._boosters["extreme"].predict(matrix)).reshape(-1)[0])
        if direction.shape != (3,) or not np.isfinite(direction).all():
            raise ValueError("XGBoost returned invalid direction probabilities.")
        direction = np.clip(direction, 0.0, 1.0)
        direction_sum = direction.sum()
        if direction_sum <= 0:
            raise ValueError("XGBoost direction probabilities sum to zero.")
        direction /= direction_sum
        if not np.isfinite([xgb_return, xgb_volatility, extreme]).all():
            raise ValueError("XGBoost returned non-finite predictions.")

        expected_return = self._xgb_weight * xgb_return + self._lstm_weight * float(lstm_prediction.expected_return)
        volatility = max(
            0.0,
            self._xgb_weight * xgb_volatility
            + self._lstm_weight * float(lstm_prediction.predicted_volatility),
        )
        probabilities = (
            self._xgb_weight * direction
            + self._lstm_weight * np.array([
                lstm_prediction.bearish_probability,
                lstm_prediction.sideways_probability,
                lstm_prediction.bullish_probability,
            ], dtype=np.float64)
        )
        probabilities = np.clip(probabilities, 0.0, 1.0)
        probabilities /= probabilities.sum()
        extreme_probability = float(np.clip(
            self._xgb_weight * extreme
            + self._lstm_weight * float(lstm_prediction.extreme_movement_probability),
            0.0, 1.0,
        ))
        quantiles = expected_return + self._quantile_offsets

        return MarketPrediction(
            as_of_utc=as_of.isoformat().replace("+00:00", "Z"),
            reference_price=reference_price,
            expected_return=float(expected_return),
            return_quantiles={
                0.1: float(quantiles[0]), 0.5: float(quantiles[1]), 0.9: float(quantiles[2]),
            },
            predicted_volatility=float(volatility),
            bearish_probability=float(probabilities[0]),
            sideways_probability=float(probabilities[1]),
            bullish_probability=float(probabilities[2]),
            extreme_movement_probability=extreme_probability,
            model_available=True,
            input_validation_status="validated: completed, aligned multi-timeframe inputs",
        )
