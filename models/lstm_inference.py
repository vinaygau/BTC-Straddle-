"""Offline inference adapter for a trained multitask LSTM checkpoint."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from config.data_config import SUPPORTED_TIMEFRAMES
from data.feature_engineering import build_multitimeframe_features, candle_availability_times
from .interfaces import MarketPrediction, MarketPredictionProvider
from .train import _build_torch_network


class LSTMMarketPredictionProvider(MarketPredictionProvider):
    """Load an offline LSTM artifact and predict from completed candle history."""

    def __init__(self, artifact_dir: str | Path, *, device: str = "cpu") -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("LSTM inference requires PyTorch.") from exc

        self._torch = torch
        self._device = torch.device(device)
        if self._device.type != "cpu" and not torch.cuda.is_available():
            raise RuntimeError(f"Requested inference device {self._device} is unavailable.")

        artifact_dir = Path(artifact_dir)
        scaler_path = artifact_dir / "feature_scaler.npz"
        checkpoint_path = artifact_dir / "lstm_multitask.pt"
        if not scaler_path.is_file() or not checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing LSTM model or scaler in {artifact_dir}")

        with np.load(scaler_path, allow_pickle=False) as scaler:
            self._mean = scaler["mean"].astype(np.float64)
            self._scale = scaler["scale"].astype(np.float64)
            self._feature_columns = tuple(str(name) for name in scaler["feature_columns"])
        checkpoint = torch.load(checkpoint_path, map_location=self._device, weights_only=True)
        checkpoint_features = tuple(checkpoint["feature_columns"])
        self._sequence_length = int(checkpoint["sequence_length"])
        input_size = int(checkpoint["input_size"])
        self._return_mean = float(checkpoint["return_mean"])
        self._return_scale = float(checkpoint["return_scale"])
        self._volatility_mean = float(checkpoint["volatility_mean"])
        self._volatility_scale = float(checkpoint["volatility_scale"])
        if checkpoint_features != self._feature_columns:
            raise ValueError("LSTM checkpoint and scaler feature columns do not match.")
        if len(self._feature_columns) != input_size or self._mean.shape != self._scale.shape:
            raise ValueError("LSTM checkpoint and scaler dimensions do not match.")
        if self._mean.shape != (input_size,) or not np.isfinite(self._mean).all():
            raise ValueError("LSTM feature scaler has invalid means.")
        if not np.isfinite(self._scale).all() or np.any(self._scale <= 0):
            raise ValueError("LSTM feature scaler has invalid scales.")
        target_parameters = (
            self._return_mean, self._return_scale,
            self._volatility_mean, self._volatility_scale,
        )
        if not np.isfinite(target_parameters).all() or min(self._return_scale, self._volatility_scale) <= 0:
            raise ValueError("LSTM target scaling parameters are invalid.")
        if self._sequence_length <= 0:
            raise ValueError("LSTM sequence length must be positive.")

        self._network = _build_torch_network(torch, input_size).to(self._device)
        self._network.load_state_dict(checkpoint["state_dict"])
        self._network.eval()

    def predict(
        self,
        completed_candles_by_timeframe: dict[str, object],
        *,
        as_of_utc: str,
    ) -> MarketPrediction:
        as_of, sequence, _latest = self.prepare_input(
            completed_candles_by_timeframe, as_of_utc=as_of_utc
        )
        prediction = self.predict_prepared(as_of, sequence)
        base = completed_candles_by_timeframe["5m"].copy()
        base["timestamp_utc"] = pd.to_datetime(base["timestamp_utc"], utc=True, errors="coerce")
        available = candle_availability_times(base, "5m")
        complete = base.loc[available <= as_of].sort_values("timestamp_utc", kind="stable")
        reference_price = float(pd.to_numeric(complete["close"], errors="coerce").iloc[-1])
        return replace(prediction, reference_price=reference_price, model_available=True,
                       input_validation_status="validated: completed, aligned multi-timeframe inputs")

    def prepare_input(
        self,
        completed_candles_by_timeframe: dict[str, object],
        *,
        as_of_utc: str,
    ) -> tuple[pd.Timestamp, np.ndarray, np.ndarray]:
        """Build the shared, standardized LSTM/XGBoost input at one as-of time."""
        as_of = pd.Timestamp(as_of_utc)
        if as_of.tzinfo is None:
            raise ValueError("as_of_utc must include a timezone.")
        as_of = as_of.tz_convert("UTC")

        unknown_timeframes = set(completed_candles_by_timeframe) - set(SUPPORTED_TIMEFRAMES)
        if unknown_timeframes:
            raise ValueError(f"Unsupported candle timeframes: {sorted(unknown_timeframes)}")
        candles: dict[str, pd.DataFrame] = {}
        for timeframe, source in completed_candles_by_timeframe.items():
            if not isinstance(source, pd.DataFrame):
                raise TypeError(f"{timeframe} candles must be a pandas DataFrame.")
            if "timestamp_utc" not in source:
                raise ValueError(f"{timeframe} candles require timestamp_utc.")
            frame = source.copy()
            frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce")
            if frame["timestamp_utc"].isna().any():
                raise ValueError(f"{timeframe} candles contain invalid timestamps.")
            if "close_time_utc" in frame:
                frame["close_time_utc"] = pd.to_datetime(frame["close_time_utc"], utc=True, errors="coerce")
            available_at = candle_availability_times(frame, timeframe)
            candles[timeframe] = frame.loc[available_at <= as_of].reset_index(drop=True)
        if "5m" not in candles or candles["5m"].empty:
            raise ValueError("No completed 5m candles are available at as_of_utc.")

        features = build_multitimeframe_features(candles, timeframes=SUPPORTED_TIMEFRAMES)
        if features.empty or features["prediction_time_utc"].iloc[-1] != as_of:
            raise ValueError("as_of_utc must equal the latest completed 5m candle close.")
        if len(features) < self._sequence_length:
            raise ValueError("Insufficient completed 5m history for the trained LSTM sequence.")

        missing_features = set(self._feature_columns).difference(features.columns)
        if missing_features:
            raise ValueError(f"Missing trained features: {sorted(missing_features)}")
        sequence = features.loc[:, self._feature_columns].tail(self._sequence_length).to_numpy(dtype=np.float64)
        sequence_times = pd.to_datetime(
            features["prediction_time_utc"].tail(self._sequence_length), utc=True
        )
        if not np.isfinite(sequence).all():
            raise ValueError("Latest feature sequence contains missing or non-finite values.")
        if not sequence_times.diff().dropna().eq(pd.Timedelta(minutes=5)).all():
            raise ValueError("Latest feature sequence has a missing 5m candle.")

        normalized = np.clip((sequence - self._mean) / self._scale, -10.0, 10.0).astype(np.float32)
        return as_of, normalized, normalized[-1].copy()

    def predict_prepared(self, as_of: pd.Timestamp, normalized_sequence: np.ndarray) -> MarketPrediction:
        """Predict from a validated standardized sequence returned by prepare_input."""
        if as_of.tzinfo is None:
            raise ValueError("Prepared prediction timestamp must include a timezone.")
        expected_shape = (self._sequence_length, len(self._feature_columns))
        if normalized_sequence.shape != expected_shape or not np.isfinite(normalized_sequence).all():
            raise ValueError(f"Prepared sequence must be finite with shape {expected_shape}.")
        normalized_sequence = normalized_sequence.astype(np.float32, copy=False)
        values = self._torch.from_numpy(normalized_sequence[None, ...]).to(self._device)
        with self._torch.no_grad():
            quantiles, volatility, direction, extreme = self._network(values)
            return_quantiles = (
                quantiles[0].cpu().numpy() * self._return_scale + self._return_mean
            )
            direction_probabilities = self._torch.softmax(direction, dim=1)[0].cpu().numpy()
            extreme_probability = float(self._torch.sigmoid(extreme[0]).cpu())
            predicted_volatility = max(
                0.0,
                float(volatility[0].cpu()) * self._volatility_scale + self._volatility_mean,
            )

        return MarketPrediction(
            as_of_utc=as_of.isoformat().replace("+00:00", "Z"),
            expected_return=float(return_quantiles[1]),
            return_quantiles={
                0.1: float(return_quantiles[0]),
                0.5: float(return_quantiles[1]),
                0.9: float(return_quantiles[2]),
            },
            predicted_volatility=predicted_volatility,
            bearish_probability=float(direction_probabilities[0]),
            sideways_probability=float(direction_probabilities[1]),
            bullish_probability=float(direction_probabilities[2]),
            extreme_movement_probability=extreme_probability,
            model_available=True,
            input_validation_status="validated: completed, aligned multi-timeframe inputs",
        )
