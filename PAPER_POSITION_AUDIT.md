# Paper Position Module Audit

## Existing implementation

- `execution/paper_trader.py`: `simulate_entry` caps a paired entry to the smaller Call/Put bid size; `simulate_exit` caps a paired buyback to the smaller ask size. `PaperPosition` stores a combined entry premium and one open amount.
- `streamlit_app.py`: `stream_quote` normalizes Deribit WebSocket quotes; `render_live_panel` gates entry on five-second quote freshness, computes combined ask, unrealized/realized quote-unit P&L, premium-price stop/take-profit signals, simulates close, and appends an in-memory event list.
- `data/deribit_stream.py`: receives bid/ask, displayed sizes, timestamps, and instrument state from Deribit ticker updates.
- `data/deribit_live.py` and `strategies/atm_selection.py`: exchange instrument metadata includes contract size, quote currency, settlement currency, expiry, strike, and option type. `OptionContract` currently retains contract size and settlement currency but drops quote currency.
- `data/market_data.py`: `OptionQuote` currently omits contract multiplier and currency metadata; `has_fresh_quote` supplies the existing freshness check.
- `strategies/risk_management.py`: current stop and take-profit thresholds use combined premium changes (1% stop and 2% take-profit), rather than position net P&L.

## Gaps found

- P&L is combined and expressed as quote-units times amount; per-leg prices and costs are not stored. Fees, conversion units, and multiplier semantics are not carried into the paper position.
- Matched top-of-book sizing already prevents one-leg-only entries and closes, but requested-versus-filled quantities and partial-close lifecycle state are not explicit.
- Entry checks do not fully validate both sides of the book, quote timestamps, crossed markets, future/expired instruments, and BTC settlement metadata in the execution engine.
- Position valuation is not persisted. The paper event log is limited to the current browser session.
- A stop signal is generated from a buyback-price threshold. The close remains manual, but the signal is not based on net position P&L.
- Expired options have no settlement-data path. The application must mark settlement as pending and avoid treating a stale ask as an executable close.

## Implemented changes and remaining limitation

The existing module now stores per-leg entry prices, matched quantities, multiplier/currency metadata, fee/slippage assumptions, and realized gross/net P&L. Freshness, two-sided book validity, positive sizes, same-strike/expiry, supported BTC settlement, and non-expiry are checked before entry, valuation, and close. Entry/close are matched to both legs' top-of-book amounts; close has a review/confirm step and can partially close. Net P&L-based stop/take-profit status is configurable and signal-only. SQLite stores the active position and append-only lifecycle events; the Streamlit dashboard loads the open position/history after restart.

BTC inverse P&L is only enabled for the Deribit metadata represented by this app: BTC premiums, BTC settlement, and multiplier 1. The amount field is the exchange BTC amount and already contains the multiplier unit, so it is not multiplied twice. Fee estimation uses the configured Standard-tier Deribit options reference; the account's actual tier and execution classification are unknown. Slippage defaults to zero beyond the displayed bid/ask.

Expiry settlement is not implemented because the application does not yet retrieve and validate the actual Deribit delivery price/settlement record. Expired positions are retained as `EXPIRED_AWAITING_SETTLEMENT` and cannot be closed against a stale ask. Pre-upgrade session-only positions missing per-leg prices are preserved as `LEGACY_UNVERIFIED`; the app blocks P&L and close until archived or reviewed. No real orders are created.
