"""Causal BTC market-regime labels and nearest historical analogues.

The detector uses only features observed at or before ``as_of_utc``. Historical
forward outcomes are included only after their complete one-hour horizon has
elapsed by that as-of time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


SIMILARITY_FEATURES = (
    "5m_realized_volatility_12",
    "5m_momentum_return_12",
    "5m_momentum_return_24",
    "5m_body_to_range",
    "5m_volume_zscore_24",
    "5m_taker_buy_base_ratio",
)
REGIME_FEATURES = SIMILARITY_FEATURES + ("5m_realized_volatility_6", "5m_log_return_1")


@dataclass(frozen=True)
class MarketRegimeResult:
    as_of_utc: str
    current_regime: str
    current_features: Mapping[str, float]
    matching_timestamps_utc: tuple[str, ...]
    similarity_scores: tuple[float, ...]
    comparable_outcomes: tuple[float, ...]
    outcome_summary: Mapping[str, float | int | None]
    historical_volatility_summary: Mapping[str, float | int | None]
    warnings: tuple[str, ...] = ()

    @property
    def comparable_count(self) -> int:
        return len(self.matching_timestamps_utc)


def _regime_label(row: pd.Series, volatility_cut: tuple[float, float]) -> str:
    rv = float(row["5m_realized_volatility_12"])
    momentum = float(row["5m_momentum_return_12"])
    rv_low, rv_high = volatility_cut
    recent_move = abs(float(row.get("5m_log_return_1", 0.0)))
    rv_short = float(row.get("5m_realized_volatility_6", rv))
    if recent_move >= max(4.0 * rv_short, 0.01):
        return "extreme_movement"
    if rv >= rv_high and abs(momentum) >= 0.0075:
        return "high_volatility_breakout"
    if momentum >= 0.0025:
        return "bullish_trend"
    if momentum <= -0.0025:
        return "bearish_trend"
    return "high_volatility_sideways" if rv >= rv_high else "low_volatility_sideways"


def detect_market_regime(
    feature_history: pd.DataFrame,
    *,
    as_of_utc: str | pd.Timestamp | None = None,
    k: int = 250,
    min_comparables: int = 25,
    horizon_steps: int = 12,
) -> MarketRegimeResult:
    """Label current conditions and return causally comparable past states.

    ``feature_history`` is the aligned multi-timeframe feature table. It may
    include ``target_return``; otherwise the outcome is computed from
    ``5m_close`` and retained only where its horizon matured before as-of.
    Similarity is inverse robust-standardized Euclidean distance among rows
    with the same regime. No matching outcome is fabricated when support is
    below ``min_comparables``.
    """
    required = {"prediction_time_utc", "5m_close", *REGIME_FEATURES}
    missing = required.difference(feature_history.columns)
    if missing:
        raise ValueError(f"Feature history lacks regime inputs: {sorted(missing)}")
    if k <= 0 or min_comparables <= 0 or horizon_steps <= 0:
        raise ValueError("k, min_comparables, and horizon_steps must be positive")

    frame = feature_history.copy()
    frame["prediction_time_utc"] = pd.to_datetime(frame["prediction_time_utc"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["prediction_time_utc"]).sort_values("prediction_time_utc", kind="stable")
    if frame.empty:
        raise ValueError("No timestamped feature rows are available")
    as_of = pd.Timestamp(as_of_utc) if as_of_utc is not None else frame["prediction_time_utc"].iloc[-1]
    if as_of.tzinfo is None:
        raise ValueError("as_of_utc must be timezone-aware")
    as_of = as_of.tz_convert("UTC")
    frame = frame.loc[frame.prediction_time_utc <= as_of].reset_index(drop=True)
    if frame.empty or frame.prediction_time_utc.iloc[-1] != as_of:
        raise ValueError("as_of_utc must match a completed feature timestamp in the history")

    regime_inputs = frame.loc[:, REGIME_FEATURES].apply(pd.to_numeric, errors="coerce")
    complete = np.isfinite(regime_inputs.to_numpy(dtype=float)).all(axis=1)
    usable = frame.loc[complete].copy().reset_index(drop=True)
    regime_inputs = regime_inputs.loc[complete].reset_index(drop=True)
    matrix = regime_inputs.loc[:, SIMILARITY_FEATURES]
    if usable.empty or usable.prediction_time_utc.iloc[-1] != as_of:
        raise ValueError("Current regime features are incomplete or non-finite")

    rv = matrix["5m_realized_volatility_12"].to_numpy(dtype=float)
    rv_low, rv_high = np.quantile(rv, [1 / 3, 2 / 3])
    momentum = matrix["5m_momentum_return_12"].to_numpy(dtype=float)
    short_rv = regime_inputs["5m_realized_volatility_6"].to_numpy(dtype=float)
    latest_bar_move = regime_inputs["5m_log_return_1"].to_numpy(dtype=float)
    labels = np.select(
        [np.abs(latest_bar_move) >= np.maximum(4.0 * short_rv, .01),
         (rv >= rv_high) & (np.abs(momentum) >= .0075),
         momentum >= .0025,
         momentum <= -.0025,
         rv >= rv_high],
        ["extreme_movement", "high_volatility_breakout", "bullish_trend", "bearish_trend",
         "high_volatility_sideways"],
        default="low_volatility_sideways",
    )
    current_label = str(labels[-1])
    current = matrix.iloc[-1]
    current_vector = current.to_numpy(dtype=float)

    # Robust scales are learned only from the historical candidate pool.
    candidate_mask = labels == current_label
    candidate_indices = np.flatnonzero(candidate_mask)
    warnings: list[str] = []
    if len(candidate_indices):
        candidate_matrix = matrix.iloc[candidate_indices].to_numpy(dtype=float)
        med = np.median(candidate_matrix, axis=0)
        iqr = np.quantile(candidate_matrix, .75, axis=0) - np.quantile(candidate_matrix, .25, axis=0)
        scale = np.where(iqr > 1e-12, iqr, np.std(candidate_matrix, axis=0))
        scale = np.where(scale > 1e-12, scale, 1.0)
        distances = np.sqrt(np.mean(((candidate_matrix - current_vector) / scale) ** 2, axis=1))
        order = np.argsort(distances, kind="stable")
        order = order[candidate_indices[order] != len(usable) - 1]
        order = order[:k]
        chosen = candidate_indices[order]
        chosen_distances = distances[order]
    else:
        chosen = np.asarray([], dtype=int)
        chosen_distances = np.asarray([], dtype=float)

    timestamps = pd.to_datetime(usable.prediction_time_utc, utc=True)
    if "target_return" in usable:
        outcomes = pd.to_numeric(usable.target_return, errors="coerce").to_numpy(dtype=float)
        matured = (timestamps + pd.Timedelta(minutes=5 * horizon_steps)) <= as_of
    else:
        # Join by exact horizon timestamp; never treat the next surviving row
        # as one hour ahead if an intervening feature row was dropped.
        source = frame.loc[:, ["prediction_time_utc", "5m_close"]].copy()
        source["5m_close"] = pd.to_numeric(source["5m_close"], errors="coerce")
        close_by_time = source.drop_duplicates("prediction_time_utc", keep="last").set_index("prediction_time_utc")["5m_close"]
        target_times = timestamps + pd.Timedelta(minutes=5 * horizon_steps)
        future_close = close_by_time.reindex(target_times).to_numpy(dtype=float)
        current_close = pd.to_numeric(usable["5m_close"], errors="coerce").to_numpy(dtype=float)
        outcomes = future_close / current_close - 1.0
        matured = np.asarray((timestamps + pd.Timedelta(minutes=5 * horizon_steps)) <= as_of)
    valid_outcome = matured[chosen] & np.isfinite(outcomes[chosen]) if len(chosen) else np.asarray([], dtype=bool)
    outcome_values = outcomes[chosen][valid_outcome] if len(chosen) else np.asarray([], dtype=float)
    chosen = chosen[valid_outcome]
    chosen_distances = chosen_distances[valid_outcome]
    similarity = np.exp(-chosen_distances / max(float(np.median(chosen_distances)) if len(chosen_distances) else 1.0, 1e-9))
    if len(chosen) < min_comparables:
        warnings.append(f"Only {len(chosen)} matured same-regime analogues; {min_comparables} required for outcome summaries.")
        outcome_summary: dict[str, float | int | None] = {"count": int(len(chosen)), "mean": None, "median": None,
                                                               "q10": None, "q90": None, "std": None}
        outcomes_for_result: tuple[float, ...] = ()
        historical_volatility_summary: dict[str, float | int | None] = {
            "count": int(len(chosen)), "current_5m_sigma": float(current["5m_realized_volatility_12"]),
            "matched_median_5m_sigma": None, "matched_q10_5m_sigma": None, "matched_q90_5m_sigma": None,
        }
    else:
        outcome_summary = {
            "count": int(len(outcome_values)),
            "mean": float(np.mean(outcome_values)),
            "median": float(np.median(outcome_values)),
            "q10": float(np.quantile(outcome_values, .1)),
            "q90": float(np.quantile(outcome_values, .9)),
            "std": float(np.std(outcome_values, ddof=0)),
        }
        outcomes_for_result = tuple(float(x) for x in outcome_values)
        matched_sigma = matrix.loc[chosen, "5m_realized_volatility_12"].to_numpy(dtype=float)
        historical_volatility_summary = {
            "count": int(len(matched_sigma)),
            "current_5m_sigma": float(current["5m_realized_volatility_12"]),
            "matched_median_5m_sigma": float(np.median(matched_sigma)),
            "matched_q10_5m_sigma": float(np.quantile(matched_sigma, .1)),
            "matched_q90_5m_sigma": float(np.quantile(matched_sigma, .9)),
        }

    return MarketRegimeResult(
        as_of_utc=as_of.isoformat().replace("+00:00", "Z"),
        current_regime=current_label,
        current_features={str(k): float(v) for k, v in current.items()},
        matching_timestamps_utc=tuple(stamp.isoformat().replace("+00:00", "Z")
                                      for stamp in timestamps.iloc[chosen]),
        similarity_scores=tuple(float(x) for x in similarity),
        comparable_outcomes=outcomes_for_result,
        outcome_summary=outcome_summary,
        historical_volatility_summary=historical_volatility_summary,
        warnings=tuple(warnings),
    )
