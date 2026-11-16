"""Honest strategy comparison wrapper for available Deribit trade/mark history.

This adapter runs the repository's mark-price proxy replay. It deliberately
does not relabel trade marks as executable quotes or confirmed fills.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from backtesting.mark_proxy import run_mark_price_proxy_backtest
from options.historical_options_loader import HistoricalOptionsData, load_historical_option_trades


def _proxy_metrics(trades: pd.DataFrame, prefix: str, selected_rows: int) -> dict[str, Any]:
    column = f"{prefix}_realized_mark_pnl_btc"
    values = pd.to_numeric(trades.get(column, pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
    if not len(values):
        return {
            "status": "no_observed_exit_marks",
            "outcome_count": 0,
            "outcome_coverage": 0.0,
            "win_rate": None,
            "mean_mark_pnl_btc": None,
            "median_mark_pnl_btc": None,
            "cvar95_mark_loss_btc": None,
            "profit_factor": None,
            "max_drawdown": None,
            "sharpe_ratio": None,
            "sortino_ratio": None,
            "return_on_margin": None,
        }
    positive = values[values > 0].sum()
    negative = -values[values < 0].sum()
    losses = np.maximum(-values, 0.0)
    tail = max(1, int(np.ceil(len(losses) * .05)))
    # The legacy replay is overlapping one-hour snapshots, not a serial account
    # equity curve; path-dependent metrics are deliberately not calculated.
    return {
        "status": "mark_price_proxy_only",
        "outcome_count": int(len(values)),
        "outcome_coverage": float(len(values) / max(1, selected_rows)),
        "win_rate": float(np.mean(values > 0)),
        "mean_mark_pnl_btc": float(np.mean(values)),
        "median_mark_pnl_btc": float(np.median(values)),
        "cvar95_mark_loss_btc": float(np.sort(losses)[-tail:].mean()),
        "profit_factor": float(positive / negative) if negative > 0 else None,
        "max_drawdown": None,
        "sharpe_ratio": None,
        "sortino_ratio": None,
        "return_on_margin": None,
    }


def run_mark_proxy_comparison(
    predictions: pd.DataFrame,
    historical_options: HistoricalOptionsData | pd.DataFrame | str,
    btc_candles: pd.DataFrame,
    *,
    max_mark_age_seconds: int = 300,
    cvar_lambda: float = 1.0,
    quantity: float = 1.0,
    fee_rate: float = 0.0,
    slippage_rate: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compare ATM and current dynamic selection where marks exist.

    Historical-scenario selector columns may be supplied in a future causal
    walk-forward export. Until then those two requested variants are reported
    as unavailable, never as modelled or realized performance.
    """
    tape = (historical_options if isinstance(historical_options, HistoricalOptionsData)
            else load_historical_option_trades(historical_options))
    trades, legacy_summary = run_mark_price_proxy_backtest(
        predictions,
        tape.trades,
        btc_candles,
        max_mark_age_seconds=max_mark_age_seconds,
        cvar_lambda=cvar_lambda,
        quantity=quantity,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
    )
    selected_rows = int(legacy_summary.get("selected_rows", 0))
    comparisons: dict[str, Any] = {
        "nearest_atm_short_straddle": _proxy_metrics(trades, "atm", selected_rows),
        "existing_dynamic_pair_ranking": _proxy_metrics(trades, "dynamic", selected_rows),
        "historical_scenario_dynamic_strike": {
            "status": "not_evaluated",
            "reason": "The supplied historical decision rows do not contain a causally generated historical-scenario selector output.",
        },
        "hybrid_forecast_plus_historical_scenario": {
            "status": "not_evaluated",
            "reason": "No overlapping, causally generated hybrid-plus-scenario decisions and historical option quote history are available.",
        },
    }
    report = {
        "scope": "historical option trade/mark-price proxy comparison; not an executable-fill backtest",
        "price_evidence": "historical executed trade print and mark price observations",
        "confirmed_exchange_fills_for_strategy": False,
        "has_historical_bid_ask": tape.has_historical_bid_ask,
        "has_displayed_order_book_size": tape.has_displayed_size,
        "has_margin_observations": False,
        "option_history_first_utc": tape.first_timestamp_utc,
        "option_history_last_utc": tape.last_timestamp_utc,
        "prediction_rows": len(predictions),
        "selected_rows": selected_rows,
        "no_trade_or_no_pair_rows": max(0, len(trades) - selected_rows),
        "fee_rate": fee_rate,
        "slippage_rate": slippage_rate,
        "margin_requirement": None,
        "comparisons": comparisons,
        "proxy_replay_details": legacy_summary,
        "warnings": list(tape.warnings) + [
            "Default fees and slippage are zero unless explicitly supplied; run sensitivity cases before interpreting proxy values.",
            "Decision rows overlap in time, so path-dependent drawdown and risk-adjusted account ratios are not calculated.",
        ],
    }
    return trades, report
