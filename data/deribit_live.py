"""Read-only BTC option snapshots from Deribit's public APIs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from data.market_data import OptionQuote, filter_btc_option_instruments
from strategies.atm_selection import OptionContract, select_atm_pair


PUBLIC_API_ROOT = "https://www.deribit.com/api/v2"
TESTNET_API_ROOT = "https://test.deribit.com/api/v2"
REQUEST_TIMEOUT_SECONDS = 10


@dataclass(frozen=True)
class LiveOptionMarket:
    """A selected option contract and its latest public ticker quote."""

    contract: OptionContract
    quote: OptionQuote
    state: str


@dataclass(frozen=True)
class LiveMarketSnapshot:
    """One manually refreshed production snapshot; it does not place orders."""

    reference_price: float
    call: LiveOptionMarket
    put: LiveOptionMarket
    fetched_at: str


@dataclass(frozen=True)
class LiveMarketCatalog:
    """Open BTC options plus one point-in-time book summary for the option chain."""

    reference_price: float
    calls: tuple[OptionContract, ...]
    puts: tuple[OptionContract, ...]
    states: dict[str, str]
    summaries: dict[str, dict[str, Any]]
    fetched_at: str
    environment: str = "production"


def _rpc(
    method: str,
    params: dict[str, Any],
    *,
    api_root: str = PUBLIC_API_ROOT,
) -> Any:
    """Call one public Deribit JSON-RPC method without API credentials."""
    if api_root not in {PUBLIC_API_ROOT, TESTNET_API_ROOT}:
        raise ValueError("Only the configured Deribit production and TestNet public APIs are supported.")
    body = json.dumps(
        {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    ).encode("utf-8")
    request = Request(
        f"{api_root}/{method}",
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "BTCOptionsStraddleDemo/1.0"},
        method="POST",
    )

    with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        payload = json.loads(response.read().decode("utf-8"))

    error = payload.get("error")
    if error:
        message = error.get("message", "Unknown Deribit API error")
        code = error.get("code", "unknown")
        raise RuntimeError(f"Deribit {method} failed ({code}): {message}")
    if "result" not in payload:
        raise RuntimeError(f"Deribit {method} returned no result.")
    return payload["result"]


def _iso_from_milliseconds(value: Any) -> str | None:
    if value is None:
        return None
    try:
        timestamp = datetime.fromtimestamp(float(value) / 1000.0, tz=timezone.utc)
    except (TypeError, ValueError, OverflowError, OSError):
        return None
    return timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _instrument_contracts(raw_instruments: list[dict[str, Any]]) -> tuple[list[OptionContract], list[OptionContract], dict[str, str]]:
    """Normalize Deribit instrument records for the existing ATM selector."""
    ordered = sorted(
        raw_instruments,
        key=lambda item: int(item.get("expiration_timestamp") or 0),
    )
    normalized: list[dict[str, Any]] = []
    states: dict[str, str] = {}
    raw_by_instrument = {
        str(item.get("instrument_name") or ""): item for item in ordered
    }

    for item in ordered:
        instrument_id = str(item.get("instrument_name") or "").strip()
        expiry = _iso_from_milliseconds(item.get("expiration_timestamp"))
        if not instrument_id or not expiry:
            continue

        state = str(item.get("state") or "unknown").lower()
        states[instrument_id] = state
        is_open = state == "open"
        normalized.append(
            {
                "instrument_id": instrument_id,
                "underlying": item.get("base_currency"),
                "option_type": item.get("option_type"),
                "strike": item.get("strike"),
                "expiry": expiry,
                "is_active": is_open and bool(item.get("is_active", True)),
                "is_tradeable": is_open,
                "multiplier": item.get("contract_size"),
                "tick_size": item.get("tick_size"),
                "min_quantity": item.get("min_trade_amount"),
                "tick_size_steps": item.get("tick_size_steps") or [],
                "lot_size": item.get("lot_size"),
                "quote_currency": item.get("quote_currency"),
                "settlement_currency": item.get("settlement_currency"),
            }
        )

    metadata = filter_btc_option_instruments(normalized, require_tradeable=True)
    calls = [
        OptionContract(
            instrument_id=item.instrument_id,
            option_type=item.option_type,
            strike=item.strike,
            expiry=item.expiry,
            is_active=item.is_active,
            min_trade_amount=item.min_quantity,
            tick_size=item.tick_size,
            tick_size_steps=tuple(
                (float(step["above_price"]), float(step["tick_size"]))
                for step in raw_by_instrument.get(item.instrument_id, {}).get("tick_size_steps", [])
                if isinstance(step, dict)
                and step.get("above_price") is not None
                and step.get("tick_size") is not None
            ),
            multiplier=item.multiplier,
            settlement_currency=item.settlement_currency,
            quote_currency=item.quote_currency,
        )
        for item in metadata
        if item.option_type == "CALL"
    ]
    puts = [
        OptionContract(
            instrument_id=item.instrument_id,
            option_type=item.option_type,
            strike=item.strike,
            expiry=item.expiry,
            is_active=item.is_active,
            min_trade_amount=item.min_quantity,
            tick_size=item.tick_size,
            tick_size_steps=tuple(
                (float(step["above_price"]), float(step["tick_size"]))
                for step in raw_by_instrument.get(item.instrument_id, {}).get("tick_size_steps", [])
                if isinstance(step, dict)
                and step.get("above_price") is not None
                and step.get("tick_size") is not None
            ),
            multiplier=item.multiplier,
            settlement_currency=item.settlement_currency,
        )
        for item in metadata
        if item.option_type == "PUT"
    ]
    return calls, puts, states


def _fetch_option_market(contract: OptionContract, state: str) -> LiveOptionMarket:
    ticker = _rpc(
        "public/ticker",
        {"instrument_name": contract.instrument_id},
    )
    bid = _optional_float(ticker.get("best_bid_price"))
    ask = _optional_float(ticker.get("best_ask_price"))
    quote = OptionQuote(
        instrument_id=contract.instrument_id,
        option_type=contract.option_type,
        strike=contract.strike,
        expiry=contract.expiry,
        bid=bid,
        ask=ask,
        bid_size=_optional_float(ticker.get("best_bid_amount")),
        ask_size=_optional_float(ticker.get("best_ask_amount")),
        timestamp=_iso_from_milliseconds(ticker.get("timestamp")),
        is_active=state == "open",
    )
    quoted_contract = OptionContract(
        instrument_id=contract.instrument_id,
        option_type=contract.option_type,
        strike=contract.strike,
        expiry=contract.expiry,
        is_active=contract.is_active,
        bid=bid,
        ask=ask,
    )
    return LiveOptionMarket(contract=quoted_contract, quote=quote, state=state)


def fetch_live_btc_option_snapshot() -> LiveMarketSnapshot:
    """Fetch the BTC index, nearest-ATM pair, and their latest public quotes."""
    index_result = _rpc("public/get_index_price", {"index_name": "btc_usd"})
    reference_price = _optional_float(index_result.get("index_price"))
    if reference_price is None or reference_price <= 0:
        raise RuntimeError("Deribit returned an invalid BTC index price.")

    instruments = _rpc(
        "public/get_instruments",
        {"currency": "BTC", "kind": "option", "expired": False},
    )
    calls, puts, states = _instrument_contracts(instruments)
    if not calls or not puts:
        raise RuntimeError("Deribit returned no open BTC option Call/Put instruments.")

    try:
        call_contract, put_contract = select_atm_pair(reference_price, calls, puts)
    except ValueError as exc:
        raise RuntimeError(f"Could not select a matching live BTC option pair: {exc}") from exc

    call_market = _fetch_option_market(call_contract, states.get(call_contract.instrument_id, "unknown"))
    put_market = _fetch_option_market(put_contract, states.get(put_contract.instrument_id, "unknown"))
    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return LiveMarketSnapshot(
        reference_price=reference_price,
        call=call_market,
        put=put_market,
        fetched_at=fetched_at,
    )


def fetch_live_btc_option_catalog() -> LiveMarketCatalog:
    """Fetch the BTC index, open option chain, and one summary quote per option."""
    return _fetch_btc_option_catalog(PUBLIC_API_ROOT, environment="production")


def fetch_testnet_btc_option_catalog() -> LiveMarketCatalog:
    """Fetch the BTC option chain from Deribit TestNet public market data."""
    return _fetch_btc_option_catalog(TESTNET_API_ROOT, environment="testnet")


def fetch_btc_index_price() -> float:
    """Fetch Deribit's current public BTC USD index independently of candle bars."""
    result = _rpc("public/get_index_price", {"index_name": "btc_usd"})
    index_price = _optional_float(result.get("index_price"))
    if index_price is None or index_price <= 0:
        raise RuntimeError("Deribit returned an invalid BTC USD index price.")
    return index_price


def fetch_btc_perpetual_candles(resolution: str = "60", candle_count: int = 96) -> list[dict[str, Any]]:
    """Fetch actual BTC perpetual OHLC candles from Deribit's public production API."""
    minutes_by_resolution = {
        "1": 1,
        "5": 5,
        "15": 15,
        "60": 60,
        "120": 120,
        "360": 360,
        "1D": 1440,
    }
    if resolution not in minutes_by_resolution:
        raise ValueError("Unsupported BTC candle interval.")
    if isinstance(candle_count, bool) or not 2 <= candle_count <= 240:
        raise ValueError("BTC candle count must be between 2 and 240.")

    now = datetime.now(timezone.utc)
    end_timestamp = int(now.timestamp() * 1000)
    start_timestamp = int(
        (now - timedelta(minutes=minutes_by_resolution[resolution] * candle_count)).timestamp() * 1000
    )
    candle_params = urlencode(
        {
            "instrument_name": "BTC-PERPETUAL",
            "start_timestamp": start_timestamp,
            "end_timestamp": end_timestamp,
            "resolution": resolution,
        }
    )
    request = Request(
        f"{PUBLIC_API_ROOT}/public/get_tradingview_chart_data?{candle_params}",
        headers={"User-Agent": "BTCOptionsStraddleDemo/1.0"},
        method="GET",
    )
    with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        payload = json.loads(response.read().decode("utf-8"))
    error = payload.get("error")
    if error:
        raise RuntimeError(
            f"Deribit candle request failed ({error.get('code', 'unknown')}): "
            f"{error.get('message', 'Unknown Deribit API error')}"
        )
    result = payload.get("result")
    if not isinstance(result, dict) or result.get("status") == "no_data":
        raise RuntimeError("Deribit did not return BTC candle data for this interval.")

    ticks = result.get("ticks")
    opens = result.get("open")
    highs = result.get("high")
    lows = result.get("low")
    closes = result.get("close")
    volumes = result.get("volume")
    arrays = (ticks, opens, highs, lows, closes)
    if any(not isinstance(values, list) for values in arrays):
        raise RuntimeError("Deribit returned an incomplete BTC candle response.")
    if len({len(values) for values in arrays}) != 1 or not ticks:
        raise RuntimeError("Deribit returned mismatched or empty BTC candle data.")
    if not isinstance(volumes, list) or len(volumes) != len(ticks):
        volumes = [None] * len(ticks)

    candles: list[dict[str, Any]] = []
    for tick, open_price, high_price, low_price, close_price, volume in zip(
        ticks, opens, highs, lows, closes, volumes,
    ):
        values = tuple(_optional_float(value) for value in (tick, open_price, high_price, low_price, close_price))
        if any(value is None for value in values):
            continue
        timestamp_ms, open_value, high_value, low_value, close_value = values
        if timestamp_ms is None or min(open_value, high_value, low_value, close_value) <= 0:
            continue
        if low_value > min(open_value, close_value) or high_value < max(open_value, close_value):
            continue
        timestamp = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)
        candles.append(
            {
                "Time": timestamp.strftime("%Y-%m-%d %H:%M"),
                "Open": open_value,
                "Close": close_value,
                "Low": low_value,
                "High": high_value,
                "Volume": _optional_float(volume),
            }
        )
    if not candles:
        raise RuntimeError("Deribit returned no valid BTC candles for this interval.")
    return candles


def _fetch_btc_option_catalog(api_root: str, *, environment: str) -> LiveMarketCatalog:
    index_result = _rpc(
        "public/get_index_price", {"index_name": "btc_usd"}, api_root=api_root
    )
    reference_price = _optional_float(index_result.get("index_price"))
    if reference_price is None or reference_price <= 0:
        raise RuntimeError("Deribit returned an invalid BTC index price.")

    instruments = _rpc(
        "public/get_instruments",
        {"currency": "BTC", "kind": "option", "expired": False},
        api_root=api_root,
    )
    calls, puts, states = _instrument_contracts(instruments)
    if not calls or not puts:
        raise RuntimeError("Deribit returned no open BTC option Call/Put instruments.")

    summaries_raw = _rpc(
        "public/get_book_summary_by_currency",
        {"currency": "BTC", "kind": "option"},
        api_root=api_root,
    )
    summaries: dict[str, dict[str, Any]] = {}
    for item in summaries_raw:
        instrument_id = str(item.get("instrument_name") or "")
        if instrument_id:
            raw_greeks = item.get("greeks") if isinstance(item.get("greeks"), dict) else {}
            def field(name: str):
                return _optional_float(item.get(name, raw_greeks.get(name)))

            mark_iv = field("mark_iv")
            bid_iv = field("bid_iv")
            ask_iv = field("ask_iv")
            summaries[instrument_id] = {
                "bid": _optional_float(item.get("bid_price")),
                "ask": _optional_float(item.get("ask_price")),
                "mark_price": _optional_float(item.get("mark_price")),
                "timestamp": _iso_from_milliseconds(item.get("creation_timestamp")),
                "bid_size": _optional_float(item.get("bid_amount")),
                "ask_size": _optional_float(item.get("ask_amount")),
                "implied_volatility": (mark_iv if mark_iv is not None else bid_iv if bid_iv is not None else ask_iv) / 100.0
                    if any(value is not None for value in (mark_iv, bid_iv, ask_iv)) else None,
                "delta": field("delta"),
                "gamma": field("gamma"),
                "theta": field("theta"),
                "vega": field("vega"),
            }

    return LiveMarketCatalog(
        reference_price=reference_price,
        calls=tuple(calls),
        puts=tuple(puts),
        states=states,
        summaries=summaries,
        fetched_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        environment=environment,
    )
