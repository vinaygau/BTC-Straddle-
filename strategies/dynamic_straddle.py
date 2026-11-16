"""Research-only, scenario-based ranking for complete BTC short straddles."""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
import math

import numpy as np

from config.risk_config import RiskConfig
from models.interfaces import MarketPrediction
from options.combined_straddle_analyzer import CombinedStraddleAnalysis, analyze_short_straddle
from options.contracts import MatchedOptionPair, OptionQuoteSnapshot
from options.option_pair_matcher import match_pairs


@dataclass(frozen=True)
class DynamicCandidate:
    pair: MatchedOptionPair
    analysis: CombinedStraddleAnalysis
    expected_net_pnl: float
    cvar95: float
    score: float
    status: str
    rejection_reasons: tuple[str, ...]
    margin_requirement: float | None = None
    liquidity_status: str = "unknown"
    scenario_source: str = "model-conditioned one-hour distribution"
    probability_of_profit: float | None = None
    moneyness_fraction: float | None = None


@dataclass(frozen=True)
class DynamicSelection:
    as_of_utc: str
    candidates_seen: int
    evaluated_candidates: tuple[DynamicCandidate, ...]
    eligible_candidates: tuple[DynamicCandidate, ...]
    selected: DynamicCandidate | None
    execution_mode: str = "research_only; no orders created"

    @property
    def decision(self) -> str:
        return "SELECT_PAIR" if self.selected is not None else "NO_TRADE"


def candidate_risk_summary(
    candidate: DynamicCandidate,
    *,
    candlestick_risk_score: int | None = None,
    candlestick_veto_threshold: int | None = None,
) -> dict[str, float | int | str | None]:
    """Return interpretable, scenario-based risk measures for one candidate.

    The qualitative band is a transparent UI heuristic, not a calibrated
    probability of loss or a guarantee about losses outside the scenarios.
    """
    profit_probability = candidate.probability_of_profit
    loss_probability = None if profit_probability is None else max(0.0, 1.0 - profit_probability)
    pnl_samples = candidate.analysis.pnl_samples
    worst_simulated_loss = max((max(0.0, -float(value)) for value in pnl_samples), default=None)
    credit = float(candidate.analysis.entry_credit)
    cvar = float(candidate.cvar95)
    candle_veto = (
        candlestick_risk_score is not None
        and candlestick_veto_threshold is not None
        and candlestick_risk_score >= candlestick_veto_threshold
    )

    if (candidate.status != "eligible" or candle_veto
            or (loss_probability is not None and loss_probability >= 0.50)
            or (worst_simulated_loss is not None and credit > 0 and worst_simulated_loss >= 2.0 * credit)):
        level = "ELEVATED"
    elif (loss_probability is None or loss_probability >= 0.25
          or (credit > 0 and cvar >= 0.50 * credit)
          or candidate.liquidity_status != "sufficient_top_of_book_size"):
        level = "MODERATE"
    else:
        level = "LOWER IN SAMPLED SCENARIOS"

    reasons = []
    if loss_probability is not None:
        reasons.append(f"scenario probability of loss {loss_probability:.1%}")
    if worst_simulated_loss is not None:
        reasons.append(f"worst sampled loss {worst_simulated_loss:.6f} BTC per unit")
    if candle_veto:
        reasons.append("candlestick risk veto active")
    if candidate.liquidity_status != "sufficient_top_of_book_size":
        reasons.append("displayed bid liquidity is insufficient or unknown")
    if candidate.margin_requirement is None:
        reasons.append("account-specific margin is unknown")
    return {
        "level": level,
        "probability_of_loss": loss_probability,
        "worst_simulated_loss": worst_simulated_loss,
        "cvar95": cvar,
        "entry_credit": credit,
        "scenario_count": len(pnl_samples),
        "margin_requirement": candidate.margin_requirement,
        "reasons": tuple(reasons),
    }


def _normal_cdf(values: np.ndarray) -> np.ndarray:
    return np.fromiter((0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0))) for value in values),
                       dtype=np.float64, count=len(values))


def _iv_fraction(value: float | None) -> float | None:
    if value is None or not math.isfinite(float(value)) or value <= 0:
        return None
    # The live adapter normalizes Deribit IV to fractions; tolerate raw percent input too.
    result = float(value) / 100.0 if value > 3.0 else float(value)
    return result if 0.001 <= result <= 5.0 else None


def _option_btc_value(spot: np.ndarray, strike: float, years: float, iv: float, option_type: str) -> np.ndarray:
    """Black–Scholes USD value converted to BTC, for a Deribit BTC-settled option proxy."""
    intrinsic = np.maximum(spot - strike, 0.0) if option_type == "CALL" else np.maximum(strike - spot, 0.0)
    if years <= 0 or iv <= 0:
        return intrinsic / spot
    root_time = math.sqrt(years)
    d1 = (np.log(spot / strike) + 0.5 * iv * iv * years) / (iv * root_time)
    d2 = d1 - iv * root_time
    if option_type == "CALL":
        usd_value = spot * _normal_cdf(d1) - strike * _normal_cdf(d2)
    else:
        usd_value = strike * _normal_cdf(-d2) - spot * _normal_cdf(-d1)
    return np.maximum(usd_value, 0.0) / spot


def _forecast_scenarios(prediction: MarketPrediction, horizon_steps: int) -> tuple[np.ndarray, np.ndarray]:
    expected = prediction.expected_return
    if expected is None or not math.isfinite(float(expected)):
        raise ValueError("A finite expected_return is required for dynamic ranking.")
    sigma = prediction.predicted_volatility
    if sigma is None or not math.isfinite(float(sigma)) or sigma <= 0:
        quantiles = prediction.return_quantiles or {}
        low, high = quantiles.get(0.1), quantiles.get(0.9)
        if low is None or high is None or high <= low:
            raise ValueError("Forecast needs positive volatility or ordered 10th/90th return quantiles.")
        sigma = (float(high) - float(low)) / 2.563
    total_sigma = max(float(sigma), 1e-8) * math.sqrt(horizon_steps)
    z = np.linspace(-3.0, 3.0, 121)
    weights = np.exp(-0.5 * z * z)
    weights /= weights.sum()
    returns = np.clip(float(expected) + z * total_sigma, -0.95, 5.0)
    return returns, weights


def _candidate_pnl_scenarios(
    pair: MatchedOptionPair,
    spot: float,
    prediction: MarketPrediction,
    as_of: datetime,
    horizon_steps: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    call_iv, put_iv = _iv_fraction(pair.call.implied_volatility), _iv_fraction(pair.put.implied_volatility)
    if call_iv is None or put_iv is None:
        raise ValueError("Both option legs need valid implied volatility.")
    expiry = datetime.fromisoformat(pair.call.expiry.replace("Z", "+00:00"))
    if expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=timezone.utc)
    seconds_to_expiry = (expiry.astimezone(timezone.utc) - as_of.astimezone(timezone.utc)).total_seconds()
    if seconds_to_expiry <= 0:
        raise ValueError("Option expiry is not in the future.")

    returns, weights = _forecast_scenarios(prediction, horizon_steps)
    future_spot = spot * (1.0 + returns)
    horizon_seconds = horizon_steps * 300
    remaining_years = max(0.0, seconds_to_expiry - horizon_seconds) / (365.25 * 24 * 3600)
    call_exit = _option_btc_value(future_spot, pair.call.strike, remaining_years, call_iv, "CALL")
    put_exit = _option_btc_value(future_spot, pair.put.strike, remaining_years, put_iv, "PUT")
    entry = (pair.call.bid * pair.call.multiplier + pair.put.bid * pair.put.multiplier) * pair.quantity
    exit_cost = (call_exit * pair.call.multiplier + put_exit * pair.put.multiplier) * pair.quantity
    pnl = entry - exit_cost
    return pnl, weights, float(np.dot(weights, pnl))


def _expiry_pnl_scenarios(pair: MatchedOptionPair, scenario_set, reference_price: float):
    """Calculate inverse BTC-settled expiry payoff on simulated terminal prices."""
    try:
        scenario_expiry = datetime.fromisoformat(str(scenario_set.expiry_utc).replace("Z", "+00:00"))
        pair_expiry = datetime.fromisoformat(pair.call.expiry.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise ValueError("Expiry scenarios must provide a valid expiry_utc") from exc
    if scenario_expiry.tzinfo is None or pair_expiry.tzinfo is None:
        raise ValueError("Scenario and option expiry timestamps must be timezone-aware")
    if scenario_expiry.astimezone(timezone.utc) != pair_expiry.astimezone(timezone.utc):
        raise ValueError("Expiry scenario timestamp does not match the option pair expiry")
    spot = float(scenario_set.reference_price)
    if not math.isfinite(spot) or spot <= 0 or abs(spot / reference_price - 1.0) > .005:
        raise ValueError("Expiry scenario reference price is missing or stale relative to the live index")
    terminal = np.asarray(scenario_set.terminal_prices, dtype=np.float64)
    if terminal.ndim != 1 or len(terminal) < 20 or not np.isfinite(terminal).all() or np.any(terminal <= 0):
        raise ValueError("Expiry scenario set needs at least 20 finite positive terminal prices")
    call_payoff = np.maximum(terminal - pair.call.strike, 0.0) / terminal
    put_payoff = np.maximum(pair.put.strike - terminal, 0.0) / terminal
    exit_cost = (call_payoff * pair.call.multiplier + put_payoff * pair.put.multiplier) * pair.quantity
    entry_credit = (pair.call.bid * pair.call.multiplier + pair.put.bid * pair.put.multiplier) * pair.quantity
    raw_pnl = entry_credit - exit_cost
    weights = np.full(len(raw_pnl), 1.0 / len(raw_pnl), dtype=np.float64)
    expected = float(np.mean(raw_pnl))
    return raw_pnl, weights, expected, entry_credit, float(np.mean(exit_cost)), spot


def _weighted_cvar(losses: np.ndarray, weights: np.ndarray, confidence: float = 0.95) -> float:
    if not 0.0 < confidence < 1.0:
        raise ValueError("CVaR confidence must be between zero and one")
    order = np.argsort(losses)[::-1]
    remaining = 1.0 - confidence
    tail_loss = 0.0
    for index in order:
        weight = min(float(weights[index]), remaining)
        tail_loss += weight * float(losses[index])
        remaining -= weight
        if remaining <= 1e-12:
            break
    return tail_loss / (1.0 - confidence)


def _weighted_cvar95(losses: np.ndarray, weights: np.ndarray) -> float:
    """Compatibility helper for the existing mark-proxy backtester."""
    return _weighted_cvar(losses, weights, 0.95)


def rank_dynamic_straddles(
    quotes: list[OptionQuoteSnapshot] | tuple[OptionQuoteSnapshot, ...],
    reference_price: float,
    prediction: MarketPrediction,
    *,
    now: datetime | None = None,
    quantity: float = 1.0,
    horizon_steps: int = 12,
    risk: RiskConfig | None = None,
    expiry_scenarios: dict[str, object] | None = None,
    candlestick_risk_score: int | None = None,
) -> DynamicSelection:
    """Rank available same-strike/same-expiry pairs; never creates orders.

    Entry credit uses the live bid for both short legs. Future buyback values are
    Black–Scholes scenario marks using each leg's current IV, so the score is a
    research estimate, not an executable quote or a fill guarantee.
    """
    risk = risk or RiskConfig()
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("now must include a timezone.")
    if reference_price <= 0 or quantity <= 0 or horizon_steps <= 0:
        raise ValueError("reference_price, quantity, and horizon_steps must be positive.")
    as_of = now.astimezone(timezone.utc)
    as_of_iso = as_of.isoformat().replace("+00:00", "Z")
    sized_quotes = [replace(quote, quantity=quantity) for quote in quotes]
    matched = match_pairs(sized_quotes, now=as_of, max_age_seconds=30)
    results: list[DynamicCandidate] = []

    for pair in matched:
        reasons: list[str] = []
        scenario_set = (expiry_scenarios or {}).get(pair.call.expiry)
        if risk.require_expiry_scenarios and scenario_set is None:
            results.append(DynamicCandidate(
                pair, analyze_short_straddle(pair, reference_price=reference_price, now=as_of),
                0.0, 0.0, -math.inf, "rejected",
                ("expiry-matched historical scenarios are required",),
                margin_requirement=None,
            ))
            continue
        if risk.require_expiry_scenarios:
            scenario_count = getattr(scenario_set, "simulation_count", 0)
            historical_samples = getattr(scenario_set, "historical_sample_count", 0)
            if scenario_count < risk.minimum_scenario_count or historical_samples < risk.minimum_historical_samples:
                results.append(DynamicCandidate(
                    pair, analyze_short_straddle(pair, reference_price=reference_price, now=as_of),
                    0.0, 0.0, -math.inf, "rejected",
                    (f"scenario reliability thresholds not met (scenarios {scenario_count}/{risk.minimum_scenario_count}; "
                     f"historical samples {historical_samples}/{risk.minimum_historical_samples})",),
                ))
                continue
        try:
            if scenario_set is not None:
                pnl_scenarios, scenario_weights, scenario_mean, entry_credit, mean_exit_cost, _ = _expiry_pnl_scenarios(pair, scenario_set, reference_price)
            else:
                pnl_scenarios, scenario_weights, scenario_mean = _candidate_pnl_scenarios(
                    pair, reference_price, prediction, as_of, horizon_steps
                )
                entry_credit = (pair.call.bid * pair.call.multiplier + pair.put.bid * pair.put.multiplier) * quantity
                mean_exit_cost = entry_credit - scenario_mean
        except ValueError as exc:
            results.append(DynamicCandidate(pair, analyze_short_straddle(pair, reference_price=reference_price, now=as_of), 0.0,
                                            0.0, -math.inf, "rejected", (str(exc),)))
            continue

        scenario_pnl = pnl_scenarios
        losses = np.maximum(-scenario_pnl, 0.0)
        cvar95 = _weighted_cvar(losses, scenario_weights, risk.cvar_confidence)
        fee_estimate = (entry_credit + max(0.0, mean_exit_cost)) * risk.fee_rate
        slippage_estimate = (entry_credit + max(0.0, mean_exit_cost)) * risk.slippage_rate
        expected_net_pnl = scenario_mean * (quantity / pair.quantity) - fee_estimate - slippage_estimate
        cvar95 += fee_estimate + slippage_estimate
        score = expected_net_pnl - risk.cvar_penalty_lambda * cvar95
        analysis = analyze_short_straddle(
            pair,
            expected_buyback_cost=mean_exit_cost,
            fee_rate=risk.fee_rate,
            slippage_rate=risk.slippage_rate,
            reference_price=reference_price,
            now=as_of,
        )
        analysis = replace(analysis, expected_pnl=float(expected_net_pnl),
                           cvar95=float(cvar95), pnl_samples=tuple(float(item) for item in scenario_pnl))

        moneyness_fraction = float(pair.call.strike / reference_price - 1.0)
        if (risk.max_strike_distance_fraction is not None
                and abs(moneyness_fraction) > risk.max_strike_distance_fraction):
            reasons.append("strike is outside the configured near-ATM analysis band")
        if (risk.max_candlestick_risk_score is not None and candlestick_risk_score is not None
                and candlestick_risk_score >= risk.max_candlestick_risk_score):
            reasons.append(
                f"live candlestick risk score {candlestick_risk_score} meets/exceeds "
                f"veto threshold {risk.max_candlestick_risk_score}"
            )

        if risk.max_abs_delta is not None and (analysis.delta is None or abs(analysis.delta) > risk.max_abs_delta):
            reasons.append("combined delta exceeds limit or is unavailable")
        if risk.max_abs_gamma is not None and (analysis.gamma is None or abs(analysis.gamma) > risk.max_abs_gamma):
            reasons.append("combined gamma exceeds limit or is unavailable")
        if risk.max_cvar95 is not None and cvar95 > risk.max_cvar95:
            reasons.append("CVaR95 exceeds configured limit")
        worst_scenario_loss = float(np.max(losses)) if len(losses) else 0.0
        if risk.max_scenario_loss is not None and worst_scenario_loss > risk.max_scenario_loss:
            reasons.append("worst simulated scenario loss exceeds configured maximum")
        required_size = max(float(quantity), float(risk.minimum_bid_size or 0.0))
        if risk.require_bid_size and (pair.call.bid_size is None or pair.put.bid_size is None):
            reasons.append("displayed bid size is unavailable for one or both legs")
        elif pair.call.bid_size is not None and pair.put.bid_size is not None:
            if min(pair.call.bid_size, pair.put.bid_size) < required_size:
                reasons.append(f"displayed bid size is below requested quantity ({required_size:g} contracts)")
        if risk.maximum_spread_fraction is not None:
            for leg in (pair.call, pair.put):
                mid = (leg.bid + leg.ask) / 2.0
                if mid <= 0 or (leg.ask - leg.bid) / mid > risk.maximum_spread_fraction:
                    reasons.append("one or both option spreads exceed configured limit")
                    break
        if risk.max_abs_expected_return is not None and abs(float(prediction.expected_return or 0.0)) > risk.max_abs_expected_return:
            reasons.append("forecast directional move exceeds configured limit")
        if expected_net_pnl <= risk.minimum_expected_net_pnl:
            reasons.append("expected net P&L does not exceed the configured minimum")
        if score <= risk.minimum_score:
            reasons.append("risk-adjusted score does not exceed the configured minimum")
        if risk.max_extreme_movement_probability is not None:
            extreme_probability = prediction.extreme_movement_probability
            if extreme_probability is None or extreme_probability > risk.max_extreme_movement_probability:
                reasons.append("extreme-movement probability is unavailable or exceeds configured limit")
        margin_estimate = analysis.initial_margin_estimate
        if risk.max_margin_requirement is not None:
            if margin_estimate is None:
                reasons.append("standard margin estimate is unavailable for the configured absolute margin limit")
            elif margin_estimate > risk.max_margin_requirement:
                reasons.append("standard margin estimate exceeds configured absolute margin limit")
        if risk.max_margin_fraction is not None:
            if margin_estimate is None or risk.account_equity is None or risk.account_equity <= 0:
                reasons.append("margin estimate or positive account equity is unavailable for configured margin fraction")
            elif margin_estimate / risk.account_equity > risk.max_margin_fraction:
                reasons.append("standard margin estimate exceeds configured margin fraction")

        results.append(DynamicCandidate(
            pair=pair,
            analysis=analysis,
            expected_net_pnl=float(expected_net_pnl),
            cvar95=float(cvar95),
            score=float(score),
            status="eligible" if not reasons else "rejected",
            rejection_reasons=tuple(reasons),
            margin_requirement=analysis.initial_margin_estimate,
            liquidity_status=("sufficient_top_of_book_size" if pair.call.bid_size is not None and pair.put.bid_size is not None and min(pair.call.bid_size, pair.put.bid_size) >= required_size
                              else "insufficient_or_unknown_top_of_book_size"),
            scenario_source=("historical moving-block expiry scenarios" if scenario_set is not None
                             else "model-conditioned one-hour distribution"),
            probability_of_profit=float(np.dot(scenario_weights, (scenario_pnl > 0).astype(float))),
            moneyness_fraction=moneyness_fraction,
        ))

    results.sort(key=lambda item: (-item.score, item.pair.call.expiry, item.pair.call.strike))
    eligible = tuple(item for item in results if item.status == "eligible")
    return DynamicSelection(as_of_iso, len(matched), tuple(results), eligible, eligible[0] if eligible else None)
