"""Evaluate matching live Call–Put candidates as combined short straddles."""
from dataclasses import dataclass, replace

from .call_analysis import analyze_call
from .combined_straddle_analyzer import CombinedStraddleAnalysis, analyze_short_straddle
from .contracts import LegAnalysis, MatchedOptionPair, OptionQuoteSnapshot
from .option_pair_matcher import match_pairs
from .put_analysis import analyze_put


@dataclass(frozen=True)
class StraddleCandidateAnalysis:
    pair: MatchedOptionPair
    call: LegAnalysis
    put: LegAnalysis
    combined: CombinedStraddleAnalysis
    liquidity_status: str


@dataclass(frozen=True)
class OptionChainAnalysis:
    reference_price: float
    candidate_count: int
    candidates: tuple[StraddleCandidateAnalysis, ...]


def analyze_option_chain(quotes: list[OptionQuoteSnapshot] | tuple[OptionQuoteSnapshot, ...],
                         reference_price: float, *, quantity: float = 1.0,
                         fee_rate: float = 0.0, slippage_rate: float = 0.0,
                         now=None, max_age_seconds: int = 30) -> OptionChainAnalysis:
    if reference_price <= 0 or quantity <= 0:
        raise ValueError("Reference price and analysis quantity must be positive")
    sized_quotes = [replace(quote, quantity=quantity) for quote in quotes]
    matched = match_pairs(sized_quotes, now=now, max_age_seconds=max_age_seconds)
    candidates = []
    for pair in matched:
        bid_sizes = (pair.call.bid_size, pair.put.bid_size)
        if any(size is None for size in bid_sizes):
            liquidity_status = "unknown"
        elif any(size < quantity for size in bid_sizes):
            liquidity_status = "insufficient_top_of_book_size"
        else:
            liquidity_status = "sufficient_top_of_book_size"
        candidates.append(StraddleCandidateAnalysis(
            pair=pair,
            call=analyze_call(pair.call, now=now, max_age_seconds=max_age_seconds),
            put=analyze_put(pair.put, now=now, max_age_seconds=max_age_seconds),
            combined=analyze_short_straddle(pair, fee_rate=fee_rate, slippage_rate=slippage_rate,
                                             now=now, max_age_seconds=max_age_seconds),
            liquidity_status=liquidity_status,
        ))
    candidates.sort(key=lambda item: (item.pair.call.expiry, abs(item.pair.call.strike - reference_price), item.pair.call.strike))
    return OptionChainAnalysis(float(reference_price), len(candidates), tuple(candidates))
