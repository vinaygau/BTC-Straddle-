"""Expiry-aligned forecast wrapper around the hybrid-conditioned path engine."""
from __future__ import annotations

from typing import Any

from .historical_scenario_generator import HistoricalScenarioSet, generate_historical_scenarios


def forecast_to_expiry(*, minimum_simulations: int = 1000, **scenario_inputs: Any) -> HistoricalScenarioSet:
    """Generate and validate an expiry forecast from actual UTC contract expiry.

    The forecast is a distribution of simulated terminal prices, not a point
    guarantee. The selector must use the returned scenario set for the exact
    option expiry and must not fall back to a one-hour-only distribution.
    """
    simulations = int(scenario_inputs.get("simulations", 1000))
    if simulations < minimum_simulations:
        raise ValueError(f"Expiry forecasting requires at least {minimum_simulations} scenarios")
    result = generate_historical_scenarios(**scenario_inputs)
    if result.expiry_utc <= result.as_of_utc:
        raise ValueError("Expiry scenario must end after its as-of timestamp")
    if result.price_paths.shape[0] != result.simulation_count:
        raise ValueError("Expiry scenario path count does not match its declared simulation count")
    if result.price_paths.shape[1] != len(result.path_timestamps_utc):
        raise ValueError("Expiry scenario timestamps do not match path horizon")
    if result.path_timestamps_utc[-1] != result.expiry_utc:
        raise ValueError("Expiry scenario does not terminate exactly at the option expiry")
    if not result.terminal_mean > 0 or not 0 <= result.extreme_move_probability <= 1:
        raise ValueError("Expiry scenario summary statistics are invalid")
    if any(not 0 <= probability <= 1 for probability in result.level_crossing_probabilities.values()):
        raise ValueError("Expiry level-crossing probabilities must be between zero and one")
    return result
