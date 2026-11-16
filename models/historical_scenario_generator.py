"""Reproducible moving-block BTC scenarios conditioned on the hybrid forecast."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .market_regime_detector import MarketRegimeResult


@dataclass(frozen=True)
class HistoricalScenarioSet:
    as_of_utc: str
    expiry_utc: str
    reference_price: float
    price_paths: np.ndarray
    path_timestamps_utc: tuple[str, ...]
    terminal_prices: np.ndarray
    terminal_price_quantiles: dict[float, float]
    terminal_returns: np.ndarray
    terminal_mean: float
    terminal_median: float
    level_crossing_probabilities: dict[float, float]
    extreme_move_threshold: float
    extreme_move_probability: float
    simulation_count: int
    block_length: int
    random_seed: int
    historical_sample_count: int
    regime: str | None
    conditioning_method: str
    warnings: tuple[str, ...] = ()


def _timestamp(value: Any, name: str) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return result.tz_convert("UTC")


def _forecast_value(prediction: Any, name: str) -> Any:
    if isinstance(prediction, dict):
        return prediction.get(name)
    return getattr(prediction, name, None)


def _contiguous_blocks(timestamps: pd.Series, block_length: int) -> list[np.ndarray]:
    if len(timestamps) < block_length:
        return []
    delta_ok = timestamps.diff().eq(pd.Timedelta(minutes=5)).to_numpy()
    starts = np.flatnonzero(delta_ok)
    segments: list[np.ndarray] = []
    if len(starts) == 0:
        return []
    run_start = starts[0] - 1
    for i in range(1, len(starts)):
        if starts[i] != starts[i - 1] + 1:
            end = starts[i - 1] + 1
            if end - run_start >= block_length:
                segments.append(np.arange(run_start, end, dtype=int))
            run_start = starts[i] - 1
    end = starts[-1] + 1
    if end - run_start >= block_length:
        segments.append(np.arange(run_start, end, dtype=int))
    return segments


def generate_historical_scenarios(
    *,
    reference_price: float,
    as_of_utc: str | pd.Timestamp,
    expiry_utc: str | pd.Timestamp,
    prediction: Any,
    historical_candles: pd.DataFrame,
    regime: MarketRegimeResult | None = None,
    simulations: int = 1000,
    block_length: int = 12,
    random_seed: int = 42,
    price_levels: Sequence[float] = (),
    extreme_move_threshold: float = 0.01,
    max_simulations: int = 10000,
    max_path_cells: int = 20_000_000,
) -> HistoricalScenarioSet:
    """Generate expiry-aligned price paths from contiguous historical blocks.

    The first forecast hour is calibrated as a distribution: sampled historical
    one-hour blocks are centered on the model's expected return and their
    cross-path dispersion is scaled to the model's quantile width (or predicted
    per-5m volatility fallback). Historical increments are not blindly added
    to the model forecast. The remaining life is a moving-block bootstrap from
    past contiguous returns. This is a scenario distribution, not a price
    prediction or guaranteed option fill.
    """
    as_of = _timestamp(as_of_utc, "as_of_utc")
    expiry = _timestamp(expiry_utc, "expiry_utc")
    if expiry <= as_of:
        raise ValueError("expiry_utc must be after as_of_utc")
    if not np.isfinite(reference_price) or reference_price <= 0:
        raise ValueError("reference_price must be finite and positive")
    if not (1 <= simulations <= max_simulations):
        raise ValueError(f"simulations must be between 1 and {max_simulations}")
    if block_length <= 0:
        raise ValueError("block_length must be positive")
    if not np.isfinite(extreme_move_threshold) or not 0 < extreme_move_threshold < 1:
        raise ValueError("extreme_move_threshold must be finite and between 0 and 1")
    normalized_levels = tuple(float(level) for level in price_levels)
    if any(not np.isfinite(level) or level <= 0 for level in normalized_levels):
        raise ValueError("price_levels must contain finite positive prices")
    expected = _forecast_value(prediction, "expected_return")
    if expected is None or not np.isfinite(float(expected)) or float(expected) <= -1:
        raise ValueError("prediction must contain a finite expected return greater than -100%")
    quantiles = _forecast_value(prediction, "return_quantiles") or {}
    q10, q90 = quantiles.get(.1), quantiles.get(.9)
    sigma_per_bar = _forecast_value(prediction, "predicted_volatility")

    time_col = "close_time_utc" if "close_time_utc" in historical_candles else "timestamp_utc"
    if time_col not in historical_candles or "close" not in historical_candles:
        raise ValueError("historical_candles requires close and a UTC timestamp column")
    candles = historical_candles.loc[:, [time_col, "close"]].copy()
    candles[time_col] = pd.to_datetime(candles[time_col], utc=True, errors="coerce")
    candles["close"] = pd.to_numeric(candles.close, errors="coerce")
    candles = candles.dropna().sort_values(time_col, kind="stable").drop_duplicates(time_col, keep="last")
    candles = candles.loc[(candles[time_col] <= as_of) & (candles.close > 0)].reset_index(drop=True)
    if len(candles) < max(block_length + 1, 13):
        raise ValueError("Insufficient historical candle data for contiguous bootstrap blocks")
    candles["log_return"] = np.log(candles.close).diff()
    candles = candles.dropna(subset=["log_return"]).reset_index(drop=True)

    minutes = (expiry - as_of).total_seconds() / 60.0
    step_count = int(np.ceil(minutes / 5.0))
    if simulations * (step_count + 1) > max_path_cells:
        raise ValueError(
            f"Requested scenario matrix needs {simulations * (step_count + 1):,} path cells; "
            f"the configured limit is {max_path_cells:,}. Lower simulations or select a nearer expiry."
        )
    step_fractions = np.ones(step_count, dtype=np.float64)
    if minutes % 5.0:
        step_fractions[-1] = (minutes % 5.0) / 5.0
    timestamps = [as_of + pd.Timedelta(minutes=5 * i) for i in range(step_count)] + [expiry]

    return_times = candles[time_col].reset_index(drop=True)
    returns = candles.log_return.to_numpy(dtype=np.float64)
    segments = _contiguous_blocks(return_times, block_length)
    if not segments:
        raise ValueError("No contiguous historical return block meets the requested block length")
    valid_start_parts = [seg[:len(seg) - block_length + 1] for seg in segments]
    valid_starts = np.concatenate(valid_start_parts).astype(int, copy=False)
    valid_start_set = set(valid_starts.tolist())

    matched_starts: list[int] = []
    if regime is not None:
        time_to_index = {stamp.value: i for i, stamp in enumerate(return_times)}
        for stamp in regime.matching_timestamps_utc:
            ts = pd.Timestamp(stamp)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            idx = time_to_index.get(ts.tz_convert("UTC").value)
            if idx is not None and idx + 1 in valid_start_set:
                matched_starts.append(idx + 1)
    warnings: list[str] = list(regime.warnings if regime else ())
    if not matched_starts:
        warnings.append("No comparable-regime block starts were available; sampling all past contiguous blocks.")
        matched_starts = valid_starts.tolist()
    if regime is not None and regime.comparable_count < 25:
        warnings.append("Comparable history is below 25 observations; scenario distribution may be unstable.")

    rng = np.random.default_rng(random_seed)
    paths = np.empty((simulations, step_count + 1), dtype=np.float32)
    sampled = np.empty((simulations, step_count), dtype=np.float64)
    for cursor in range(0, step_count, block_length):
        n = min(block_length, step_count - cursor)
        pool = matched_starts if cursor == 0 else valid_starts
        if len(pool) == 0:
            raise ValueError("No eligible historical block starts remain")
        starts = rng.choice(np.asarray(pool, dtype=int), size=simulations)
        sampled[:, cursor:cursor + n] = returns[starts[:, None] + np.arange(n)[None, :]]

    sampled *= step_fractions[None, :]
    conditioning = "historical moving-block bootstrap"
    conditioned_steps = min(12, step_count)
    if conditioned_steps:
        block = sampled[:, :conditioned_steps]
        totals = block.sum(axis=1)
        horizon_fraction = conditioned_steps / 12.0
        target_mean = float(np.log1p(float(expected))) * horizon_fraction
        if q10 is not None and q90 is not None and np.isfinite([q10, q90]).all() and float(q90) > float(q10):
            target_sigma = (float(q90) - float(q10)) / 2.563 * np.sqrt(horizon_fraction)
            conditioning += "; 1h dispersion from model q10–q90"
        elif sigma_per_bar is not None and np.isfinite(float(sigma_per_bar)) and float(sigma_per_bar) > 0:
            target_sigma = float(sigma_per_bar) * np.sqrt(conditioned_steps)
            conditioning += "; 1h dispersion from predicted per-5m volatility"
        else:
            target_sigma = float(np.std(totals, ddof=0))
            warnings.append("Model return dispersion is unavailable; historical first-hour dispersion retained.")
        observed_sigma = float(np.std(totals, ddof=0))
        centered = totals - float(np.mean(totals))
        if observed_sigma > 1e-12:
            calibrated_totals = target_mean + centered * (target_sigma / observed_sigma)
        else:
            calibrated_totals = np.full(simulations, target_mean)
            warnings.append("Sampled first-hour blocks had zero cross-path dispersion; model forecast fixed the first-hour total.")
        block += ((calibrated_totals - totals) / conditioned_steps)[:, None]
        sampled[:, :conditioned_steps] = block
        conditioning += "; model mean conditions first hour without additive forecast duplication"

    if not np.isfinite(sampled).all():
        raise ValueError("Historical bootstrap produced non-finite return increments")
    paths[:, 0] = float(reference_price)
    paths[:, 1:] = float(reference_price) * np.exp(np.cumsum(sampled, axis=1)).astype(np.float32)
    terminal = paths[:, -1].astype(np.float64)
    terminal_returns = terminal / float(reference_price) - 1.0
    quantile_values = np.quantile(terminal, [.01, .05, .10, .25, .50, .75, .90, .95, .99])
    terminal_quantiles = {q: float(v) for q, v in zip((.01,.05,.10,.25,.5,.75,.9,.95,.99), quantile_values)}
    return HistoricalScenarioSet(
        as_of_utc=as_of.isoformat().replace("+00:00", "Z"),
        expiry_utc=expiry.isoformat().replace("+00:00", "Z"),
        reference_price=float(reference_price),
        price_paths=paths,
        path_timestamps_utc=tuple(t.isoformat().replace("+00:00", "Z") for t in timestamps),
        terminal_prices=terminal,
        terminal_price_quantiles=terminal_quantiles,
        terminal_returns=terminal_returns,
        terminal_mean=float(np.mean(terminal)),
        terminal_median=float(np.median(terminal)),
        level_crossing_probabilities={
            level: float(np.mean(np.any(paths >= level, axis=1) if level >= reference_price
                                  else np.any(paths <= level, axis=1)))
            for level in normalized_levels
        },
        extreme_move_threshold=float(extreme_move_threshold),
        extreme_move_probability=float(np.mean(
            np.max(np.abs(paths.astype(np.float64) / float(reference_price) - 1.0), axis=1)
            >= extreme_move_threshold
        )),
        simulation_count=simulations,
        block_length=block_length,
        random_seed=random_seed,
        historical_sample_count=len(matched_starts),
        regime=regime.current_regime if regime else None,
        conditioning_method=conditioning,
        warnings=tuple(dict.fromkeys(warnings)),
    )
