"""Contracts for offline and live-read-only multi-timeframe prediction providers."""
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class MarketPrediction:
    as_of_utc: str
    reference_price: float | None = None
    expected_return: float | None = None
    return_quantiles: dict[float, float] | None = None
    predicted_volatility: float | None = None
    bullish_probability: float | None = None
    bearish_probability: float | None = None
    sideways_probability: float | None = None
    extreme_movement_probability: float | None = None
    confidence: float | None = None
    model_available: bool = True
    input_validation_status: str = "not_reported"


class MarketPredictionProvider(Protocol):
    def predict(self, completed_candles_by_timeframe: dict[str, object], *, as_of_utc: str) -> MarketPrediction:
        """Predict from completed candles available at the supplied UTC timestamp."""

