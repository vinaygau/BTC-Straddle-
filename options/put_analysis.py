"""Put-leg quote metrics, without recommending or submitting an order."""
from .contracts import LegAnalysis, OptionQuoteSnapshot
from .option_validation import validate_option_quote


def analyze_put(quote: OptionQuoteSnapshot, *, now=None, max_age_seconds=30) -> LegAnalysis:
    validate_option_quote(quote, now=now, max_age_seconds=max_age_seconds)
    if quote.option_type.upper() != "PUT": raise ValueError("Put analysis requires a Put")
    spread = quote.ask - quote.bid
    mid = (quote.ask + quote.bid) / 2
    return LegAnalysis(quote.instrument_id, "PUT", spread, spread / mid if mid else None, quote.bid,
                       quote.bid_size, quote.ask_size)

