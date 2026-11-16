"""Validated settings for the paper-only position simulator."""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any


@lru_cache(maxsize=1)
def load_paper_trading_config() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[1] / "configs" / "paper_trading.yaml"
    try:
        config = json.loads(path.read_text(encoding="utf-8"))["paper_trading"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(f"Invalid paper-trading configuration: {path}") from exc
    freshness = float(config["quote_validation"]["maximum_quote_age_seconds"])
    fee = float(config["accounting"]["fee_rate_per_side"])
    slip = float(config["accounting"]["slippage_rate_per_side"])
    cap = float(config["accounting"]["fee_cap_percent_of_premium"])
    stop = float(config["risk"]["stop_loss_percentage"])
    take = float(config["risk"]["take_profit_percentage"])
    if (not all(math.isfinite(v) for v in (freshness, fee, slip, cap, stop, take))
            or freshness <= 0 or min(fee, slip, cap, stop, take) < 0 or cap > 100):
        raise ValueError("Paper-trading numeric settings must be finite and non-negative; freshness must be positive.")
    if config["risk"]["stop_loss_method"] != "percentage_of_entry_credit":
        raise ValueError("Unsupported paper-trading stop-loss method.")
    if not config["execution"]["require_both_legs"]:
        raise ValueError("A paper short straddle must require matched Call and Put fills.")
    if config["accounting"].get("fee_model") != "deribit_option_standard_tier_estimate":
        raise ValueError("Unsupported paper fee model.")
    return config
