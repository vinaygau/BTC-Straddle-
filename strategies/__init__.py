"""Strategy package for Phase 1 BTC ATM straddle logic."""

from .atm_selection import OptionContract, nearest_atm_strike, select_atm_pair
from .risk_management import RiskThresholds, compute_risk_thresholds, should_stop_loss, should_take_profit
from .short_straddle import ShortStraddle, StraddleLeg, create_short_straddle

__all__ = [
    "OptionContract",
    "nearest_atm_strike",
    "select_atm_pair",
    "StraddleLeg",
    "ShortStraddle",
    "create_short_straddle",
    "RiskThresholds",
    "compute_risk_thresholds",
    "should_stop_loss",
    "should_take_profit",
]
