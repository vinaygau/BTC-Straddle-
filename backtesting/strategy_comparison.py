"""Fair, same-date comparisons for four short-straddle selection variants."""
from __future__ import annotations

from typing import Any

import pandas as pd

from .performance_metrics import calculate_trade_ledger_metrics
from config.strategy_settings import load_strategy_settings

STRATEGIES = (
    "nearest_atm_short_straddle",
    "existing_dynamic_pair_ranking",
    "historical_scenario_dynamic_strike",
    "hybrid_forecast_plus_historical_scenario",
)


def compare_strategy_ledgers(
    ledgers: dict[str, pd.DataFrame],
    *,
    starting_equity_btc: float,
    cost_assumptions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare realized ledgers only when strategy dates and costs match exactly.

    Missing strategy outputs remain ``not_evaluated``; mark-price proxy inputs
    must not be passed as executable trade ledgers.
    """
    backtest_config = load_strategy_settings()["backtest"]
    cost_assumptions = cost_assumptions or {
        "fee_rate": backtest_config["default_fee_rate"],
        "slippage_rate": backtest_config["default_slippage_rate"],
    }
    report: dict[str, Any] = {
        "status": "not_evaluated",
        "price_evidence": "requires causal out-of-fold decisions and verified or explicitly labeled fills",
        "cost_assumptions": cost_assumptions,
        "comparisons": {},
        "warnings": [],
    }
    available: dict[str, pd.DataFrame] = {}
    for name in STRATEGIES:
        frame = ledgers.get(name)
        if frame is None or frame.empty:
            report["comparisons"][name] = {"status": "not_evaluated", "reason": "no causal realized trade ledger supplied"}
            continue
        required = {"entry_time_utc", "exit_time_utc", "net_pnl_btc", "decision_time_utc"}
        if not required.issubset(frame.columns):
            report["comparisons"][name] = {"status": "not_evaluated", "reason": f"missing columns: {sorted(required - set(frame.columns))}"}
            continue
        if frame.attrs.get("price_evidence") != "verified_fills":
            report["comparisons"][name] = {"status": "not_evaluated", "reason": "ledger is not explicitly labeled as verified fills"}
            continue
        available[name] = frame.copy()
    if len(available) != len(STRATEGIES):
        report["warnings"].append("All four strategies need verified fills before a complete same-date comparison is available.")
        return report

    date_sets = {
        name: tuple(pd.to_datetime(frame.decision_time_utc, utc=True).sort_values().astype(str))
        for name, frame in available.items()
    }
    if len(set(date_sets.values())) != 1:
        report["warnings"].append("Strategy evaluation dates differ; no comparative ranking was calculated.")
        return report
    for name, frame in available.items():
        try:
            metrics = calculate_trade_ledger_metrics(frame, starting_equity_btc=starting_equity_btc)
        except ValueError as exc:
            report["comparisons"][name] = {"status": "invalid_ledger", "reason": str(exc)}
            report["warnings"].append("At least one strategy ledger is invalid or contains overlapping positions.")
            return report
        report["comparisons"][name] = {"status": "verified_fills", **metrics}
    report["status"] = "complete"
    return report
