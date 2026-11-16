"""Central configurable filters and penalties for BTC short-straddle research."""
from dataclasses import dataclass


@dataclass(frozen=True)
class RiskConfig:
    max_abs_delta: float | None = None
    max_abs_gamma: float | None = None
    max_cvar95: float | None = None
    max_scenario_loss: float | None = None
    cvar_confidence: float = 0.95
    max_margin_requirement: float | None = None
    max_margin_fraction: float | None = None
    account_equity: float | None = None
    max_abs_expected_return: float | None = None
    max_extreme_movement_probability: float | None = None
    minimum_bid_size: float | None = None
    maximum_spread_fraction: float | None = None
    cvar_penalty_lambda: float = 0.0
    fee_rate: float = 0.0
    slippage_rate: float = 0.0
    minimum_expected_net_pnl: float = 0.0
    minimum_score: float = 0.0
    require_bid_size: bool = True
    require_expiry_scenarios: bool = False
    minimum_scenario_count: int = 1000
    minimum_historical_samples: int = 25
    max_candlestick_risk_score: int | None = None
    max_strike_distance_fraction: float | None = 0.02

