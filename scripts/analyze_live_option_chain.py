"""Read and analyze a current public Deribit BTC Call–Put chain."""
import argparse
import json
from datetime import timezone

from options.market_analysis import analyze_option_chain
from options.option_chain_loader import DeribitPublicOptionChainLoader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", choices=("production", "testnet"), default="production")
    parser.add_argument("--quantity", type=float, default=1.0)
    parser.add_argument("--max-quote-age-seconds", type=int, default=30)
    parser.add_argument("--fee-rate", type=float, default=0.0)
    parser.add_argument("--slippage-rate", type=float, default=0.0)
    args = parser.parse_args()
    snapshot = DeribitPublicOptionChainLoader(environment=args.environment).load_snapshot()
    analysis = analyze_option_chain(snapshot.quotes, snapshot.reference_price, quantity=args.quantity,
                                    fee_rate=args.fee_rate, slippage_rate=args.slippage_rate,
                                    now=snapshot.fetched_at, max_age_seconds=args.max_quote_age_seconds)
    output = {
        "environment": snapshot.environment, "underlying": snapshot.underlying,
        "reference_price": snapshot.reference_price, "fetched_at": snapshot.fetched_at.isoformat(),
        "loaded_quotes": len(snapshot.quotes), "matched_straddles": analysis.candidate_count,
        "candidates": [{
            "expiry": item.pair.call.expiry, "strike": item.pair.call.strike,
            "quantity": item.pair.quantity, "call_instrument": item.pair.call.instrument_id,
            "put_instrument": item.pair.put.instrument_id,
            "call_bid": item.pair.call.bid, "put_bid": item.pair.put.bid,
            "call_bid_size": item.pair.call.bid_size, "put_bid_size": item.pair.put.bid_size,
            "liquidity_status": item.liquidity_status,
            "entry_credit": item.combined.entry_credit,
            "delta": item.combined.delta, "gamma": item.combined.gamma,
            "theta": item.combined.theta, "vega": item.combined.vega,
            "call_spread": item.call.spread, "put_spread": item.put.spread,
        } for item in analysis.candidates],
        "entry_credit_note": "Top-of-book estimate; full quantity is not guaranteed unless both displayed bid sizes cover it.",
        "execution": "none; read-only market analysis",
    }
    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
