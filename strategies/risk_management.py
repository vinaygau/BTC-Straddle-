"""Level 2 risk-management helpers for the short straddle."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskThresholds:
    """Thresholds for stop-loss and take-profit on combined premium."""

    entry_premium: float
    stop_loss: float
    take_profit: float
    stop_loss_pct: float = 0.01
    take_profit_pct: float = 0.02

    @property
    def is_valid(self) -> bool:
        return self.entry_premium > 0 and self.stop_loss > 0 and self.take_profit > 0


@dataclass(frozen=True)
class RiskEvaluation:
    """Full result of evaluating the combined buyback premium against thresholds."""

    current_buyback_premium: float
    stop_loss_threshold: float
    take_profit_threshold: float
    reason: str | None = None
    triggered: bool = False

    @property
    def is_valid(self) -> bool:
        return self.current_buyback_premium >= 0


def compute_risk_thresholds(
    entry_premium: float,
    *,
    stop_loss_pct: float = 0.01,
    take_profit_pct: float = 0.02,
) -> RiskThresholds:
    """Compute the trigger values for a short straddle using combined premium."""
    if entry_premium <= 0:
        raise ValueError("entry_premium must be greater than zero.")
    if stop_loss_pct <= 0 or take_profit_pct <= 0:
        raise ValueError("risk thresholds must be positive percentages.")

    stop_loss = entry_premium * (1 + stop_loss_pct)
    take_profit = entry_premium * (1 - take_profit_pct)
    return RiskThresholds(
        entry_premium=entry_premium,
        stop_loss=stop_loss,
        take_profit=take_profit,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
    )


def evaluate_buyback_risk(
    current_buyback_premium: float,
    thresholds: RiskThresholds,
) -> RiskEvaluation:
    """Evaluate combined premium against the stop-loss and take-profit rules."""
    if current_buyback_premium < 0:
        raise ValueError("current_buyback_premium must be non-negative.")

    if current_buyback_premium >= thresholds.stop_loss:
        return RiskEvaluation(
            current_buyback_premium=current_buyback_premium,
            stop_loss_threshold=thresholds.stop_loss,
            take_profit_threshold=thresholds.take_profit,
            reason="stop_loss",
            triggered=True,
        )

    if current_buyback_premium <= thresholds.take_profit:
        return RiskEvaluation(
            current_buyback_premium=current_buyback_premium,
            stop_loss_threshold=thresholds.stop_loss,
            take_profit_threshold=thresholds.take_profit,
            reason="take_profit",
            triggered=True,
        )

    return RiskEvaluation(
        current_buyback_premium=current_buyback_premium,
        stop_loss_threshold=thresholds.stop_loss,
        take_profit_threshold=thresholds.take_profit,
        reason=None,
        triggered=False,
    )


def should_stop_loss(current_buyback_premium: float, thresholds: RiskThresholds) -> bool:
    """Return True when the current buyback premium has increased enough."""
    return current_buyback_premium >= thresholds.stop_loss


def should_take_profit(current_buyback_premium: float, thresholds: RiskThresholds) -> bool:
    """Return True when the current buyback premium has decreased enough."""
    return current_buyback_premium <= thresholds.take_profit
