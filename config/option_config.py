"""Future option-pair validation settings; no execution is configured here."""
from dataclasses import dataclass


@dataclass(frozen=True)
class OptionMatchingConfig:
    underlying: str = "BTC"
    require_same_strike: bool = True
    require_same_expiry: bool = True
    require_active_contracts: bool = True
    require_fresh_quotes: bool = True
    max_quote_age_seconds: int = 30
    require_matched_quantities: bool = True

