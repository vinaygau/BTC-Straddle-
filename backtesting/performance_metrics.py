"""Path metrics for a non-overlapping, realized trade ledger."""
from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_trade_ledger_metrics(
    trades: pd.DataFrame,
    *,
    pnl_column: str = "net_pnl_btc",
    starting_equity_btc: float,
    periods_per_year: float | None = None,
) -> dict[str, float | int | None]:
    """Calculate account-path metrics only for ordered, non-overlapping trades."""
    if starting_equity_btc <= 0 or (periods_per_year is not None and periods_per_year <= 0):
        raise ValueError("starting equity must be positive and periods_per_year, when supplied, must be positive")
    if pnl_column not in trades:
        raise ValueError(f"trade ledger is missing {pnl_column}")
    if not {"entry_time_utc", "exit_time_utc"}.issubset(trades.columns):
        raise ValueError("trade ledger must include entry_time_utc and exit_time_utc")
    frame = trades.copy()
    frame["entry_time_utc"] = pd.to_datetime(frame.entry_time_utc, utc=True, errors="coerce")
    frame["exit_time_utc"] = pd.to_datetime(frame.exit_time_utc, utc=True, errors="coerce")
    frame[pnl_column] = pd.to_numeric(frame[pnl_column], errors="coerce")
    if frame[["entry_time_utc", "exit_time_utc", pnl_column]].isna().any().any():
        raise ValueError("trade ledger contains invalid timestamps or P&L")
    frame = frame.sort_values(["entry_time_utc", "exit_time_utc"], kind="stable")
    if (frame.exit_time_utc <= frame.entry_time_utc).any():
        raise ValueError("each trade must exit after entry")
    if len(frame) > 1 and (frame.entry_time_utc.iloc[1:].to_numpy() < frame.exit_time_utc.iloc[:-1].to_numpy()).any():
        raise ValueError("overlapping trades cannot be treated as a single account equity curve")
    pnl = frame[pnl_column].to_numpy(dtype=float)
    equity = starting_equity_btc + np.cumsum(pnl)
    if np.any(equity <= 0):
        raise ValueError("trade ledger depletes starting equity; return ratios are undefined")
    returns = pnl / np.concatenate(([starting_equity_btc], equity[:-1]))
    running_peak = np.maximum.accumulate(np.concatenate(([starting_equity_btc], equity)))[1:]
    drawdowns = equity / running_peak - 1.0
    downside = returns[returns < 0]
    losses = np.maximum(-pnl, 0.0)
    tail_count = max(1, int(np.ceil(len(losses) * 0.05)))
    std = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    downside_std = float(np.sqrt(np.mean(np.minimum(returns, 0.0) ** 2))) if len(returns) else 0.0
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = float(-pnl[pnl < 0].sum())
    return {
        "trade_count": int(len(frame)),
        "net_pnl_btc": float(pnl.sum()),
        "ending_equity_btc": float(equity[-1]),
        "win_rate": float(np.mean(pnl > 0)),
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else None,
        "cvar95_loss_btc": float(np.sort(losses)[-tail_count:].mean()),
        "max_drawdown_fraction": float(drawdowns.min()),
        "sharpe_ratio": float(np.mean(returns) / std * np.sqrt(periods_per_year)) if std > 0 and periods_per_year else None,
        "sortino_ratio": float(np.mean(returns) / downside_std * np.sqrt(periods_per_year)) if downside_std > 0 and periods_per_year else None,
        "mean_return_per_trade": float(np.mean(returns)) if len(returns) else None,
    }
