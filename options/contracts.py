"""Typed contracts shared by call, put and combined-position analysis."""
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class OptionQuoteSnapshot:
    instrument_id: str
    underlying: str
    option_type: str
    strike: float
    expiry: str
    bid: float
    ask: float
    bid_size: float | None
    ask_size: float | None
    quote_time: datetime
    active: bool = True
    quantity: float = 1.0
    multiplier: float = 1.0
    implied_volatility: float | None = None
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None
    settlement_currency: str | None = None
    mark_price: float | None = None


@dataclass(frozen=True)
class MatchedOptionPair:
    call: OptionQuoteSnapshot
    put: OptionQuoteSnapshot
    quantity: float


@dataclass(frozen=True)
class LegAnalysis:
    instrument_id: str
    option_type: str
    spread: float
    spread_fraction: float | None
    estimated_sell_price: float
    bid_liquidity: float | None
    ask_liquidity: float | None

