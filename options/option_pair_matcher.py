"""Match fresh, active Call/Put quotes into same-strike, same-expiry pairs."""
from .contracts import MatchedOptionPair, OptionQuoteSnapshot
from .option_validation import validate_option_quote


def match_pairs(quotes, *, now=None, max_age_seconds=30):
    calls, puts = [], []
    for quote in quotes:
        try: validate_option_quote(quote, now=now, max_age_seconds=max_age_seconds)
        except ValueError: continue
        (calls if quote.option_type.upper() == "CALL" else puts).append(quote)
    pairs = []
    for call in calls:
        for put in puts:
            if call.underlying.upper() != put.underlying.upper() or call.strike != put.strike or call.expiry != put.expiry or call.quantity != put.quantity: continue
            pairs.append(MatchedOptionPair(call, put, call.quantity))
    return pairs

