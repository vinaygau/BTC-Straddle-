"""Position-level short-straddle math; unknown risk inputs stay explicitly unset."""
from dataclasses import dataclass
from .contracts import MatchedOptionPair
from .option_validation import validate_pair


@dataclass(frozen=True)
class CombinedStraddleAnalysis:
    quantity: float
    entry_credit: float
    settlement_currency: str
    delta: float | None
    gamma: float | None
    theta: float | None
    vega: float | None
    expected_buyback_cost: float | None = None
    expected_pnl: float | None = None
    max_margin_requirement: float | None = None
    initial_margin_estimate: float | None = None
    maintenance_margin_estimate: float | None = None
    margin_model: str | None = None
    cvar95: float | None = None
    pnl_samples: tuple[float, ...] = ()
    fees_estimate: float | None = None
    slippage_estimate: float | None = None


def estimate_deribit_standard_inverse_margin(pair: MatchedOptionPair, reference_price: float):
    """Estimate BTC inverse-option margin under Deribit's standard margin formulas.

    This is not an account-level or portfolio-margin calculation. Mark prices
    must come from the exchange; no bid/ask midpoint substitution is made.
    """
    if not isinstance(reference_price, (int, float)) or reference_price <= 0:
        return None, None
    initial_total = 0.0
    maintenance_total = 0.0
    for leg in (pair.call, pair.put):
        mark = leg.mark_price
        if mark is None or mark < 0 or leg.multiplier <= 0:
            return None, None
        itm_fraction = (leg.strike - reference_price) / reference_price if leg.option_type.upper() == "CALL" else (reference_price - leg.strike) / reference_price
        out_of_money = max(itm_fraction, 0.0)
        if leg.option_type.upper() == "CALL":
            initial_rate = max(0.15 - out_of_money, 0.10) + mark
            maintenance_rate = 0.075 + mark
        elif leg.option_type.upper() == "PUT":
            initial_rate = max(max(0.15 - out_of_money, 0.10) + mark,
                               max(0.075, 0.075 * mark) + mark)
            maintenance_rate = max(0.075, 0.075 * mark) + mark
        else:
            return None, None
        contracts = pair.quantity * leg.multiplier
        initial_total += initial_rate * contracts
        maintenance_total += maintenance_rate * contracts
    return initial_total, maintenance_total


def analyze_short_straddle(pair: MatchedOptionPair, *, expected_buyback_cost: float | None = None,
                           pnl_samples=None, fee_rate: float = 0.0, slippage_rate: float = 0.0,
                           reference_price: float | None = None,
                           now=None, max_age_seconds=30) -> CombinedStraddleAnalysis:
    call, put, qty = pair.call, pair.put, pair.quantity
    validate_pair(call, put, now=now, max_age_seconds=max_age_seconds)
    if qty != call.quantity: raise ValueError("Pair quantity must match both option legs")
    credit = (call.bid * call.multiplier + put.bid * put.multiplier) * qty
    if fee_rate < 0 or slippage_rate < 0: raise ValueError("Fee and slippage rates cannot be negative")
    def greek(name):
        c, p = getattr(call, name), getattr(put, name)
        return None if c is None or p is None else -(c * call.multiplier + p * put.multiplier) * qty
    samples = tuple(float(x) for x in (pnl_samples or ()))
    cvar = None
    if samples:
        losses = sorted(max(0.0, -x) for x in samples)
        tail_count = max(1, (len(losses) + 19) // 20)
        cvar = sum(losses[-tail_count:]) / tail_count
    fees = None if expected_buyback_cost is None else (credit + float(expected_buyback_cost)) * fee_rate
    slippage = None if expected_buyback_cost is None else (credit + float(expected_buyback_cost)) * slippage_rate
    pnl = None if expected_buyback_cost is None else credit - float(expected_buyback_cost) - fees - slippage
    initial_margin, maintenance_margin = (estimate_deribit_standard_inverse_margin(pair, reference_price)
                                          if reference_price is not None else (None, None))
    return CombinedStraddleAnalysis(
        quantity=qty, entry_credit=credit, settlement_currency=call.settlement_currency.upper(),
        delta=greek("delta"), gamma=greek("gamma"),
        theta=greek("theta"), vega=greek("vega"), expected_buyback_cost=expected_buyback_cost,
        expected_pnl=pnl, max_margin_requirement=initial_margin,
        initial_margin_estimate=initial_margin, maintenance_margin_estimate=maintenance_margin,
        margin_model="deribit_standard_inverse_estimate" if initial_margin is not None else None,
        cvar95=cvar, pnl_samples=samples,
        fees_estimate=fees, slippage_estimate=slippage,
    )

