"""Live Deribit market data, AI research, and TestNet strategy dashboard."""

from __future__ import annotations

import csv
import hashlib
import io
import os
import time
from dataclasses import asdict
from importlib import reload
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
import streamlit as st

import config.config as project_config
import data.deribit_live as deribit_live
import data.deribit_stream as deribit_stream
import execution.deribit_testnet as deribit_testnet
import execution.deribit_nautilus as deribit_nautilus
import strategies.atm_selection as atm_selection

# A long-running Streamlit process can retain earlier versions of project
# modules in sys.modules. Reload their dependencies first, then the adapters,
# so option-rule fields and new TestNet functions resolve together.
atm_selection = reload(atm_selection)
project_config = reload(project_config)
deribit_live = reload(deribit_live)
deribit_stream = reload(deribit_stream)
deribit_testnet = reload(deribit_testnet)
deribit_nautilus = reload(deribit_nautilus)
OptionContract = atm_selection.OptionContract
Settings = project_config.Settings
get_deribit_credentials = project_config.get_deribit_credentials
LiveMarketCatalog = deribit_live.LiveMarketCatalog
DeribitPublicFeed = deribit_stream.DeribitPublicFeed
PUBLIC_WEBSOCKET_URL = deribit_stream.PUBLIC_WEBSOCKET_URL
TESTNET_WEBSOCKET_URL = deribit_stream.TESTNET_WEBSOCKET_URL
DeribitTestNetClient = deribit_testnet.DeribitTestNetClient
DeribitTestNetError = deribit_testnet.DeribitTestNetError
DeribitNautilusService = deribit_nautilus.DeribitNautilusService
from data.market_data import OptionQuote, has_fresh_quote
from data.binance_live import fetch_btcusdt_candles
from data.feature_engineering import build_multitimeframe_features
from chat.ollama_client import OllamaChatError, list_local_models, stream_chat_response
from config.data_config import DataPipelineConfig
from config.risk_config import RiskConfig
from config.strategy_settings import load_strategy_settings
from models.hybrid_inference import HybridMarketPredictionProvider
from models.expiry_forecaster import forecast_to_expiry
from models.interfaces import MarketPrediction
from models.market_regime_detector import SIMILARITY_FEATURES, REGIME_FEATURES, detect_market_regime
from models.candlestick_analysis import CandlestickAnalysis, analyze_candlesticks
from options.contracts import OptionQuoteSnapshot
from runtime import StrategyRuntime
from strategies.dynamic_straddle import DynamicSelection, rank_dynamic_straddles

st.set_page_config(
    page_title="BTC analytics terminal",
    page_icon=":material/currency_bitcoin:",
    layout="wide",
)


def _testnet_credentials() -> tuple[str, str]:
    """Read Streamlit server secrets first, then the existing private .env settings."""
    try:
        client_id = str(st.secrets.get("DERIBIT_TESTNET_CLIENT_ID", "")).strip()
        client_secret = str(st.secrets.get("DERIBIT_TESTNET_CLIENT_SECRET", "")).strip()
        if client_id and client_secret:
            return client_id, client_secret
    except Exception:
        # Local installs may not have a .streamlit/secrets.toml file.
        pass
    return get_deribit_credentials()


def _testnet_order_submission_enabled(settings: Settings) -> bool:
    """Allow a server-side Streamlit secret to opt in, with environment fallback."""
    try:
        value = st.secrets.get("DERIBIT_TESTNET_ORDER_SUBMISSION_ENABLED")
        if value is not None:
            return str(value).strip().lower() in {"1", "true", "yes", "on"}
    except Exception:
        pass
    return settings.deribit_testnet_order_submission_enabled


def _select_market_data_source(source: str) -> None:
    """Persist the sidebar workspace choice before the app reruns."""
    st.session_state["market_data_source"] = source


def _create_testnet_strategy_service(
    client: DeribitTestNetClient,
    call_contract: OptionContract,
    put_contract: OptionContract,
    amount_min: float,
    *,
    order_submission_enabled: bool,
) -> DeribitNautilusService:
    """Create a per-account, per-pair monitor without persisting credentials."""
    account_key = hashlib.sha256(client.client_id.encode("utf-8")).hexdigest()[:16]
    pair_key = hashlib.sha256(
        f"{call_contract.instrument_id}|{put_contract.instrument_id}".encode("utf-8")
    ).hexdigest()[:16]
    state_path = Path(__file__).resolve().parent / ".runtime" / "deribit" / f"{account_key}_{pair_key}.json"
    runtime = StrategyRuntime(settings=Settings.from_env(), live_trading=False)
    return runtime.prepare_live_node(
        call_contract,
        put_contract,
        amount_min=amount_min,
        client_id=client.client_id,
        client_secret=client._client_secret,
        state_path=state_path,
        order_submission_enabled=order_submission_enabled,
    )


@st.cache_data(ttl="1s", max_entries=12, refresh_mode="background")
def _cached_btc_candles(
    resolution: str,
    candle_count: int,
) -> list[dict[str, Any]]:
    """Cache the public Deribit candle request briefly between auto-refreshes."""
    return deribit_live.fetch_btc_perpetual_candles(resolution, candle_count)


@st.cache_data(ttl="1s", max_entries=2, refresh_mode="background")
def _cached_btc_index_price() -> tuple[float, str]:
    """Short-cache the live BTC index separately from the slower candle bars."""
    return (
        deribit_live.fetch_btc_index_price(),
        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    )


@st.fragment(run_every="1s")
def render_btc_candlestick_chart() -> None:
    """Show actual Deribit BTC perpetual candles, independent of strategy mode."""
    with st.container(border=True):
        st.subheader("BTC perpetual", icon=":material/candlestick_chart:")
        st.caption(
            "Actual BTC-PERPETUAL USD candles from Deribit’s public production market data. "
            "The app polls for updated index and candle data every second; displayed values change when Deribit publishes newer data. "
            "Read-only; this chart does not connect an account or place orders."
        )

        controls = st.columns([1, 1], vertical_alignment="bottom")
        interval_labels = {
            "1 minute": "1",
            "5 minutes": "5",
            "15 minutes": "15",
            "1 hour": "60",
            "2 hours": "120",
            "6 hours": "360",
            "1 day": "1D",
        }
        with controls[0]:
            interval_label = st.selectbox(
                "Candle interval",
                list(interval_labels),
                index=0,
                key="btc_candle_interval",
            )
        with controls[1]:
            candle_count = st.selectbox(
                "Candles to show",
                [48, 72, 96, 144, 192, 240],
                index=2,
                key="btc_candle_count",
            )
        resolution = interval_labels[interval_label]
        try:
            candles = _cached_btc_candles(resolution, candle_count)
        except Exception as exc:
            st.error(
                f"Could not load BTC candles from Deribit: {exc}",
                title="BTC chart unavailable",
            )
            return

        candle_frame = pd.DataFrame(candles)
        if candle_frame.empty:
            st.warning("Deribit returned no candles for the selected interval.")
            return

        first_open = float(candle_frame.iloc[0]["Open"])
        latest_close = float(candle_frame.iloc[-1]["Close"])
        price_change = latest_close - first_open
        price_change_pct = price_change / first_open * 100 if first_open else 0.0
        live_index = None
        index_as_of = None
        index_error = None
        try:
            live_index, index_as_of = _cached_btc_index_price()
        except Exception as exc:
            index_error = str(exc)
        metric_columns = st.columns([1, 1, 1, 1])
        with metric_columns[0]:
            st.metric("Live BTC index", "Unavailable" if live_index is None else f"${live_index:,.2f}",
                      delta=None if live_index is None else f"{live_index-latest_close:+,.2f} vs candle close")
        with metric_columns[1]:
            st.metric("Latest candle close", f"${latest_close:,.2f}")
        with metric_columns[2]:
            st.metric(
                f"Change across {len(candle_frame)} candles",
                f"{price_change_pct:+.2f}%",
                delta=f"${price_change:+,.2f}",
            )
        with metric_columns[3]:
            st.caption(
                f"Bar size: {interval_label} · BTC-PERPETUAL · USD · "
                f"index quote as of {index_as_of or 'unavailable'}"
            )
        if index_error:
            st.warning(f"Live index unavailable; candle bars may still display: {index_error}")

        chart_spec = {
            "animation": False,
            "grid": {"left": 72, "right": 24, "top": 24, "bottom": 84},
            "dataset": {"source": candle_frame[["Time", "Open", "Close", "Low", "High"]]},
            "tooltip": {"trigger": "axis", "axisPointer": {"type": "cross"}},
            "xAxis": {
                "type": "category",
                "boundaryGap": False,
                "axisLine": {"onZero": False},
                "splitLine": {"show": False},
                "min": "dataMin",
                "max": "dataMax",
            },
            "yAxis": {
                "scale": True,
                "axisLabel": {"formatter": "${value}"},
                "splitArea": {"show": True},
            },
            "dataZoom": [
                {"type": "inside", "start": 65, "end": 100},
                {"type": "slider", "start": 65, "end": 100, "bottom": 20},
            ],
            "series": [
                {
                    "name": "BTC-PERPETUAL",
                    "type": "candlestick",
                    "encode": {"x": "Time", "y": ["Open", "Close", "Low", "High"]},
                    "itemStyle": {
                        "color": "#18C784",
                        "color0": "#EA5966",
                        "borderColor": "#18C784",
                        "borderColor0": "#EA5966",
                    },
                }
            ],
        }
        st.echarts_chart(
            chart_spec,
            height=470,
            alt="Actual Deribit BTC perpetual USD price candlesticks with open, high, low, and close values.",
            key="btc_candlestick_chart",
        )
        st.caption(
            f"Latest source candle (UTC): {candles[-1]['Time']} · poll cycle: 1 second. "
            f"Candle interval: {interval_label}; the same candle may remain until Deribit updates it."
        )


def stream_quote(contract: OptionContract, raw: dict[str, Any] | None) -> OptionQuote:
    """Normalize one WebSocket ticker notification into the project's quote model."""
    raw = raw or {}
    return OptionQuote(
        instrument_id=contract.instrument_id,
        option_type=contract.option_type,
        strike=contract.strike,
        expiry=contract.expiry,
        bid=raw.get("bid"),
        ask=raw.get("ask"),
        bid_size=raw.get("bid_size"),
        ask_size=raw.get("ask_size"),
        timestamp=raw.get("timestamp"),
        is_active=bool(contract.is_active and raw.get("state", "open") == "open"),
        multiplier=contract.multiplier,
        quote_currency=contract.quote_currency,
        settlement_currency=contract.settlement_currency,
    )


def option_chain_rows(catalog: LiveMarketCatalog, expiry: str) -> list[dict[str, object]]:
    """Build a paired Call/Put chain view from one public summary snapshot."""
    calls = {
        contract.strike: contract
        for contract in catalog.calls
        if contract.expiry == expiry
    }
    puts = {
        contract.strike: contract
        for contract in catalog.puts
        if contract.expiry == expiry
    }
    rows: list[dict[str, object]] = []
    for strike in sorted(calls.keys() & puts.keys()):
        call = calls[strike]
        put = puts[strike]
        call_summary = catalog.summaries.get(call.instrument_id, {})
        put_summary = catalog.summaries.get(put.instrument_id, {})
        rows.append(
            {
                "Strike": strike,
                "Call instrument": call.instrument_id,
                "Call bid": call_summary.get("bid"),
                "Call ask": call_summary.get("ask"),
                "Put instrument": put.instrument_id,
                "Put bid": put_summary.get("bid"),
                "Put ask": put_summary.get("ask"),
            }
        )
    return rows


@st.cache_data(ttl="10s", max_entries=2)
def _cached_ollama_models() -> tuple[list[str], str | None]:
    try:
        return list_local_models(), None
    except OllamaChatError as exc:
        return [], str(exc)


@st.cache_resource
def _hybrid_model_provider() -> HybridMarketPredictionProvider:
    """Load the offline-trained hybrid model once per Streamlit process."""
    config = DataPipelineConfig()
    return HybridMarketPredictionProvider(config.processed_dir / "model_artifacts", device="cpu")


def _validate_live_prediction_freshness(as_of_utc: str) -> float:
    """Reject stale prepared/live candles before showing a current signal."""
    as_of = pd.Timestamp(as_of_utc)
    if as_of.tzinfo is None:
        raise RuntimeError("Hybrid input timestamp is missing its UTC timezone.")
    age_seconds = (pd.Timestamp.now(tz="UTC") - as_of.tz_convert("UTC")).total_seconds()
    max_age = DataPipelineConfig().live_5m_max_age_seconds
    if age_seconds < -60:
        raise RuntimeError("Hybrid input timestamp is more than one minute in the future.")
    if age_seconds > max_age:
        raise RuntimeError(
            f"Latest completed BTC 5m candle is {age_seconds / 60:.1f} minutes old "
            f"(maximum allowed: {max_age / 60:.0f} minutes). Refresh public candle data; no current prediction is available."
        )
    return max(0.0, age_seconds)


@st.cache_data(ttl="60s", max_entries=2)
def _cached_live_hybrid_snapshot() -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    """Fetch complete Binance candles and score them with the offline research model."""
    frames = {timeframe: fetch_btcusdt_candles(timeframe, limit=240)
              for timeframe in ("5m", "15m", "1h", "daily")}
    as_of = pd.to_datetime(frames["5m"].iloc[-1]["close_time_utc"], utc=True).isoformat()
    _validate_live_prediction_freshness(as_of)
    prediction = _hybrid_model_provider().predict(frames, as_of_utc=as_of)
    return asdict(prediction), frames


@st.cache_data(ttl="30s", max_entries=2)
def _cached_live_option_catalog(environment: str) -> LiveMarketCatalog:
    """Refresh the public option chain independently of the manual pair snapshot."""
    if environment == "production":
        return deribit_live.fetch_live_btc_option_catalog()
    if environment == "testnet":
        return deribit_live.fetch_testnet_btc_option_catalog()
    raise ValueError(f"Unsupported Deribit environment: {environment}")


@st.cache_data(ttl=3600, max_entries=1)
def _cached_historical_research_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the actual local BTC history used only for regime analogues and bootstrap sampling."""
    config = DataPipelineConfig()
    feature_path = config.processed_dir / "btc_multitimeframe_features.parquet"
    candle_path = config.processed_dir / "btc_5m.parquet"
    if not feature_path.is_file() or not candle_path.is_file():
        raise FileNotFoundError("Prepared BTC feature history and 5m candle data are required for scenario research.")
    feature_columns = ["prediction_time_utc", "5m_close", *REGIME_FEATURES]
    features = pd.read_parquet(feature_path, columns=list(dict.fromkeys(feature_columns)))
    candles = pd.read_parquet(candle_path, columns=["close_time_utc", "close"])
    return features, candles


def _catalog_analysis_quotes(catalog: LiveMarketCatalog) -> list[OptionQuoteSnapshot]:
    """Adapt exchange-sourced chain summaries without filling missing quote fields."""
    quotes = []
    for contract in (*catalog.calls, *catalog.puts):
        summary = catalog.summaries.get(contract.instrument_id, {})
        try:
            quote_time = datetime.fromisoformat(str(summary["timestamp"]).replace("Z", "+00:00"))
            if quote_time.tzinfo is None:
                quote_time = quote_time.replace(tzinfo=timezone.utc)
            multiplier = float(contract.multiplier)
            bid, ask = float(summary["bid"]), float(summary["ask"])
        except (KeyError, TypeError, ValueError):
            continue
        if multiplier <= 0 or bid <= 0 or ask < bid:
            continue
        quotes.append(OptionQuoteSnapshot(
            instrument_id=contract.instrument_id,
            underlying="BTC",
            option_type=contract.option_type,
            strike=float(contract.strike),
            expiry=contract.expiry,
            bid=bid,
            ask=ask,
            bid_size=summary.get("bid_size"),
            ask_size=summary.get("ask_size"),
            quote_time=quote_time,
            active=bool(contract.is_active),
            quantity=1.0,
            multiplier=multiplier,
            implied_volatility=summary.get("implied_volatility"),
            delta=summary.get("delta"),
            gamma=summary.get("gamma"),
            theta=summary.get("theta"),
            vega=summary.get("vega"),
            settlement_currency=contract.settlement_currency,
            mark_price=summary.get("mark_price"),
        ))
    return quotes


def _rank_live_chain(catalog: LiveMarketCatalog, prediction: MarketPrediction, *,
                     cvar_lambda: float = 1.0, fee_rate: float = 0.0,
                     slippage_rate: float = 0.0, max_spread_fraction: float | None = None,
                     max_abs_delta: float | None = None, max_cvar95: float | None = None,
                     max_scenario_loss: float | None = None,
                     max_margin_requirement: float | None = None,
                     cvar_confidence: float = 0.95,
                     minimum_bid_size: float | None = None,
                     selected_expiry: str | None = None,
                     expiry_scenarios: dict[str, object] | None = None,
                     candlestick_risk_score: int | None = None) -> DynamicSelection:
    strategy_settings = load_strategy_settings()
    optimizer_config = strategy_settings["optimizer"]
    quotes = _catalog_analysis_quotes(catalog)
    if selected_expiry is not None:
        quotes = [quote for quote in quotes if quote.expiry == selected_expiry]
    return rank_dynamic_straddles(
        quotes,
        catalog.reference_price,
        prediction,
        now=datetime.now(timezone.utc),
        risk=RiskConfig(
            cvar_penalty_lambda=cvar_lambda,
            cvar_confidence=cvar_confidence,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            maximum_spread_fraction=max_spread_fraction,
            max_abs_delta=max_abs_delta,
            max_cvar95=max_cvar95,
            max_scenario_loss=max_scenario_loss,
            max_margin_requirement=max_margin_requirement,
            minimum_bid_size=minimum_bid_size,
            require_bid_size=True,
            require_expiry_scenarios=bool(optimizer_config["require_expiry_scenarios"]),
            minimum_scenario_count=int(optimizer_config["minimum_scenario_count"]),
            minimum_historical_samples=int(optimizer_config["minimum_historical_samples"]),
            max_candlestick_risk_score=int(strategy_settings["candlestick"]["veto_risk_score"]),
            max_strike_distance_fraction=float(optimizer_config["max_strike_distance_fraction"]),
        ),
        expiry_scenarios=expiry_scenarios,
        candlestick_risk_score=candlestick_risk_score,
    )


def candidate_risk_summary(
    candidate: Any,
    *,
    candlestick_risk_score: int | None = None,
    candlestick_veto_threshold: int | None = None,
) -> dict[str, Any]:
    """Build the UI risk readout from one candidate's simulated P&L samples."""
    profit_probability = candidate.probability_of_profit
    loss_probability = None if profit_probability is None else max(0.0, 1.0 - profit_probability)
    pnl_samples = candidate.analysis.pnl_samples
    worst_simulated_loss = max((max(0.0, -float(value)) for value in pnl_samples), default=None)
    credit = float(candidate.analysis.entry_credit)
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
          or (credit > 0 and candidate.cvar95 >= 0.50 * credit)
          or candidate.liquidity_status != "sufficient_top_of_book_size"):
        level = "MODERATE"
    else:
        level = "LOWER IN SAMPLED SCENARIOS"

    reasons = []
    if loss_probability is not None:
        reasons.append(f"scenario probability of loss {loss_probability:.1%}")
    if worst_simulated_loss is not None:
        reasons.append(f"worst sampled loss {worst_simulated_loss:.3f} BTC per unit")
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
        "cvar95": float(candidate.cvar95),
        "entry_credit": credit,
        "scenario_count": len(pnl_samples),
        "margin_requirement": candidate.margin_requirement,
        "reasons": tuple(reasons),
    }


def _calculate_selected_pair_risk(
    environment: str,
    expiry: str,
    strike: float,
) -> dict[str, Any]:
    """Run expiry-scenario risk analysis for the currently selected pair."""
    strategy_settings = load_strategy_settings()
    scenario_config = strategy_settings["scenario"]
    optimizer_config = strategy_settings["optimizer"]
    prediction_data, research_frames = _cached_live_hybrid_snapshot()
    prediction = MarketPrediction(**prediction_data)
    _validate_live_prediction_freshness(prediction.as_of_utc)
    if not prediction.model_available or prediction.expected_return is None:
        raise RuntimeError("The hybrid BTC forecast is unavailable; selected-pair scenario risk cannot be calculated.")

    candle_analysis = analyze_candlesticks(
        research_frames,
        as_of_utc=prediction.as_of_utc,
        config=strategy_settings["candlestick"],
    )
    catalog = _cached_live_option_catalog(environment)
    history_features, history_candles = _cached_historical_research_data()
    current_features = build_multitimeframe_features(research_frames)
    if not current_features.empty:
        asof = pd.to_datetime(prediction.as_of_utc, utc=True)
        live_row = current_features.loc[current_features.prediction_time_utc == asof]
        if not live_row.empty:
            history_features = pd.concat(
                [history_features, live_row[history_features.columns]], ignore_index=True
            )
    regime = detect_market_regime(history_features, as_of_utc=prediction.as_of_utc)
    current_price = float(catalog.reference_price)
    simulation_count = max(
        int(scenario_config["minimum_simulations"]),
        int(scenario_config["simulation_count"]),
    )
    scenario = forecast_to_expiry(
        reference_price=current_price,
        as_of_utc=prediction.as_of_utc,
        expiry_utc=expiry,
        prediction=prediction,
        historical_candles=history_candles,
        regime=regime,
        simulations=simulation_count,
        block_length=int(scenario_config["block_length"]),
        random_seed=int(scenario_config["random_seed"]),
        price_levels=(current_price * 0.99, current_price * 1.01),
        extreme_move_threshold=float(scenario_config["extreme_move_threshold"]),
        minimum_simulations=int(scenario_config["minimum_simulations"]),
    )
    selection = _rank_live_chain(
        catalog,
        prediction,
        selected_expiry=expiry,
        expiry_scenarios={expiry: scenario},
        candlestick_risk_score=candle_analysis.risk_score,
    )
    pair_candidate = next(
        (
            item for item in selection.evaluated_candidates
            if item.pair.call.expiry == expiry and item.pair.call.strike == float(strike)
        ),
        None,
    )
    if pair_candidate is None:
        raise RuntimeError("The selected Call/Put pair is absent from the refreshed analysis chain.")
    summary = candidate_risk_summary(
        pair_candidate,
        candlestick_risk_score=candle_analysis.risk_score,
        candlestick_veto_threshold=candle_analysis.veto_threshold,
    )
    thresholds = optimizer_config.get("selection_risk_thresholds", {})
    max_loss_probability = float(thresholds.get("max_probability_of_loss", 0.25))
    max_cvar_credit_ratio = float(thresholds.get("max_cvar_to_entry_credit", 0.5))
    max_worst_credit_ratio = float(thresholds.get("max_worst_sampled_loss_to_entry_credit", 2.0))
    risk_qualified_candidates = []
    for candidate in selection.eligible_candidates:
        candidate_risk = candidate_risk_summary(
            candidate,
            candlestick_risk_score=candle_analysis.risk_score,
            candlestick_veto_threshold=candle_analysis.veto_threshold,
        )
        credit = float(candidate_risk["entry_credit"])
        prob_loss = candidate_risk["probability_of_loss"]
        worst_loss = candidate_risk["worst_simulated_loss"]
        if (
            credit > 0
            and prob_loss is not None
            and prob_loss <= max_loss_probability
            and float(candidate_risk["cvar95"]) <= max_cvar_credit_ratio * credit
            and worst_loss is not None
            and worst_loss <= max_worst_credit_ratio * credit
        ):
            risk_qualified_candidates.append(candidate)

    # If nothing meets the absolute risk budgets, find a useful fallback only
    # when it improves all three sampled-risk measures versus the selected pair
    # and still has positive modeled net P&L and score. It remains a research
    # alternative when quote/liquidity or another hard filter rejected it.
    lower_risk_alternatives = []
    selected_probability_loss = summary["probability_of_loss"]
    selected_cvar = float(summary["cvar95"])
    selected_worst_loss = summary["worst_simulated_loss"]
    configured_atm_band = optimizer_config.get("max_strike_distance_fraction")
    max_near_atm_distance = float("inf") if configured_atm_band is None else float(configured_atm_band)
    for candidate in selection.evaluated_candidates:
        if candle_analysis.risk_score >= candle_analysis.veto_threshold:
            break
        if candidate.pair.call.strike == float(strike):
            continue
        if candidate.expected_net_pnl <= 0 or candidate.score <= 0:
            continue
        if candidate.moneyness_fraction is None or abs(candidate.moneyness_fraction) > max_near_atm_distance:
            continue
        if candidate.status != "eligible" and any(
            "bid size" not in reason.lower() and "bid liquidity" not in reason.lower()
            for reason in candidate.rejection_reasons
        ):
            continue
        candidate_risk = candidate_risk_summary(candidate)
        candidate_probability_loss = candidate_risk["probability_of_loss"]
        candidate_worst_loss = candidate_risk["worst_simulated_loss"]
        if None in (selected_probability_loss, selected_worst_loss, candidate_probability_loss, candidate_worst_loss):
            continue
        risk_dominates = (
            candidate_probability_loss <= selected_probability_loss
            and candidate_risk["cvar95"] <= selected_cvar
            and candidate_worst_loss <= selected_worst_loss
        )
        strictly_safer = (
            candidate_probability_loss < selected_probability_loss
            or candidate_risk["cvar95"] < selected_cvar
            or candidate_worst_loss < selected_worst_loss
        )
        if risk_dominates and strictly_safer:
            lower_risk_alternatives.append(candidate)

    # Risk gates are applied before score ranking: among candidates that pass
    # all existing execution/research filters and these explicit risk budgets,
    # choose the highest expected-net-P&L minus CVaR-penalty score.
    model_candidate = max(risk_qualified_candidates, key=lambda item: item.score, default=None)
    diagnostic_candidate = min(
        selection.evaluated_candidates,
        key=lambda item: (
            1.0 if item.probability_of_profit is None else 1.0 - item.probability_of_profit,
            item.cvar95 / max(float(item.analysis.entry_credit), 1e-12),
            max((max(0.0, -float(value)) for value in item.analysis.pnl_samples), default=float("inf"))
            / max(float(item.analysis.entry_credit), 1e-12),
        ),
        default=None,
    )
    model_recommendation = None
    if model_candidate is not None:
        model_recommendation = {
            "expiry": model_candidate.pair.call.expiry,
            "strike": float(model_candidate.pair.call.strike),
            "call_instrument": model_candidate.pair.call.instrument_id,
            "put_instrument": model_candidate.pair.put.instrument_id,
            "expected_net_pnl": float(model_candidate.expected_net_pnl),
            "cvar95": float(model_candidate.cvar95),
            "score": float(model_candidate.score),
            "risk": candidate_risk_summary(
                model_candidate,
                candlestick_risk_score=candle_analysis.risk_score,
                candlestick_veto_threshold=candle_analysis.veto_threshold,
            ),
            "status": model_candidate.status,
            "rejection_reasons": tuple(model_candidate.rejection_reasons),
            "liquidity_status": model_candidate.liquidity_status,
            "moneyness_fraction": model_candidate.moneyness_fraction,
        }
    diagnostic = None
    if diagnostic_candidate is not None:
        diagnostic_risk = candidate_risk_summary(
            diagnostic_candidate,
            candlestick_risk_score=candle_analysis.risk_score,
            candlestick_veto_threshold=candle_analysis.veto_threshold,
        )
        diagnostic = {
            "strike": float(diagnostic_candidate.pair.call.strike),
            "status": diagnostic_candidate.status,
            "risk": diagnostic_risk,
            "rejection_reasons": tuple(diagnostic_candidate.rejection_reasons),
        }
    lower_risk_alternative = None
    fallback_candidate = max(lower_risk_alternatives, key=lambda item: item.score, default=None)
    if fallback_candidate is not None and model_recommendation is None:
        fallback_risk = candidate_risk_summary(
            fallback_candidate,
            candlestick_risk_score=candle_analysis.risk_score,
            candlestick_veto_threshold=candle_analysis.veto_threshold,
        )
        lower_risk_alternative = {
            "expiry": fallback_candidate.pair.call.expiry,
            "strike": float(fallback_candidate.pair.call.strike),
            "expected_net_pnl": float(fallback_candidate.expected_net_pnl),
            "cvar95": float(fallback_candidate.cvar95),
            "score": float(fallback_candidate.score),
            "risk": fallback_risk,
            "status": fallback_candidate.status,
            "rejection_reasons": tuple(fallback_candidate.rejection_reasons),
            "moneyness_fraction": fallback_candidate.moneyness_fraction,
            "order_ready": fallback_candidate.status == "eligible",
            "loss_probability_improvement": float(selected_probability_loss - fallback_risk["probability_of_loss"]),
            "cvar_improvement": float(selected_cvar - fallback_risk["cvar95"]),
            "worst_loss_improvement": float(selected_worst_loss - fallback_risk["worst_simulated_loss"]),
        }
    expiry_call_strikes = {float(contract.strike) for contract in catalog.calls if contract.expiry == expiry}
    expiry_put_strikes = {float(contract.strike) for contract in catalog.puts if contract.expiry == expiry}
    expiry_strikes = sorted(expiry_call_strikes & expiry_put_strikes)
    atm_strike = min(expiry_strikes, key=lambda value: abs(value - current_price)) if expiry_strikes else None
    return {
        "analysis_version": 2,
        "risk": summary,
        "candidate_status": pair_candidate.status,
        "rejection_reasons": tuple(pair_candidate.rejection_reasons),
        "liquidity_status": pair_candidate.liquidity_status,
        "expected_net_pnl": pair_candidate.expected_net_pnl,
        "score": pair_candidate.score,
        "decision": "SELECT_PAIR" if model_recommendation is not None else "NO_TRADE",
        "eligible_candidate_count": len(selection.eligible_candidates),
        "risk_qualified_candidate_count": len(risk_qualified_candidates),
        "candidates_seen": selection.candidates_seen,
        "model_recommendation": model_recommendation,
        "lower_risk_alternative": lower_risk_alternative,
        "lowest_risk_diagnostic": diagnostic,
        "risk_thresholds": {
            "max_probability_of_loss": max_loss_probability,
            "max_cvar_to_entry_credit": max_cvar_credit_ratio,
            "max_worst_sampled_loss_to_entry_credit": max_worst_credit_ratio,
        },
        "atm_strike": atm_strike,
        "reference_price": current_price,
        "expiry": expiry,
        "strike": float(strike),
        "environment": environment,
        "prediction_as_of_utc": prediction.as_of_utc,
        "catalog_fetched_at": catalog.fetched_at,
        "calculated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "calculated_monotonic": time.monotonic(),
        "candlestick_risk_score": candle_analysis.risk_score,
        "candlestick_veto_threshold": candle_analysis.veto_threshold,
        "candlestick_patterns": [
            {
                "timeframe": item.timeframe,
                "patterns": list(item.patterns),
                "direction_bias": item.direction_bias,
                "risk_score": item.risk_score,
            }
            for item in candle_analysis.timeframes
        ],
        "hybrid_forecast": {
            "horizon": "60 minutes",
            "expected_return": None if prediction.expected_return is None else float(prediction.expected_return),
            "expected_btc_price": (
                None if prediction.expected_return is None
                else float(current_price * (1.0 + prediction.expected_return))
            ),
            "predicted_volatility": None if prediction.predicted_volatility is None else float(prediction.predicted_volatility),
            "bullish_probability": None if prediction.bullish_probability is None else float(prediction.bullish_probability),
            "sideways_probability": None if prediction.sideways_probability is None else float(prediction.sideways_probability),
            "bearish_probability": None if prediction.bearish_probability is None else float(prediction.bearish_probability),
            "extreme_movement_probability": (
                None if prediction.extreme_movement_probability is None
                else float(prediction.extreme_movement_probability)
            ),
            "return_quantiles": prediction.return_quantiles or {},
        },
        "expiry_scenario_summary": {
            "terminal_mean_price": float(scenario.terminal_mean),
            "terminal_median_price": float(scenario.terminal_median),
            "terminal_quantiles": {str(key): float(value) for key, value in scenario.terminal_price_quantiles.items()},
            "extreme_move_probability": float(scenario.extreme_move_probability),
            "regime": scenario.regime,
        },
        "selected_pair_quotes": {
            "call_bid_btc": None if pair_candidate.pair.call.bid is None else float(pair_candidate.pair.call.bid),
            "call_ask_btc": None if pair_candidate.pair.call.ask is None else float(pair_candidate.pair.call.ask),
            "call_bid_size": None if pair_candidate.pair.call.bid_size is None else float(pair_candidate.pair.call.bid_size),
            "put_bid_btc": None if pair_candidate.pair.put.bid is None else float(pair_candidate.pair.put.bid),
            "put_ask_btc": None if pair_candidate.pair.put.ask is None else float(pair_candidate.pair.put.ask),
            "put_bid_size": None if pair_candidate.pair.put.bid_size is None else float(pair_candidate.pair.put.bid_size),
        },
        "expiry_scenario_count": scenario.simulation_count,
    }


def _ollama_risk_explanation(result: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    """Ask the configured local Ollama model to explain, not decide, the risk result."""
    models, models_error = _cached_ollama_models()
    if not models:
        return None, None, models_error or "No local Ollama model is installed."

    preferred_model = st.session_state.get("ollama_model_choice") or os.getenv("OLLAMA_MODEL")
    model = preferred_model if preferred_model in models else models[0]
    risk = result["risk"]
    recommendation = result.get("model_recommendation")
    limits = result["risk_thresholds"]
    credit = float(risk["entry_credit"])
    selected_risk_targets = {
        "max_probability_of_loss": limits["max_probability_of_loss"],
        "probability_of_loss_pass": (
            risk["probability_of_loss"] is not None
            and risk["probability_of_loss"] <= limits["max_probability_of_loss"]
        ),
        "max_cvar_to_entry_credit": limits["max_cvar_to_entry_credit"],
        "cvar_pass": credit > 0 and risk["cvar95"] <= limits["max_cvar_to_entry_credit"] * credit,
        "max_worst_sampled_loss_to_entry_credit": limits["max_worst_sampled_loss_to_entry_credit"],
        "worst_sampled_loss_pass": (
            risk["worst_simulated_loss"] is not None
            and credit > 0
            and risk["worst_simulated_loss"] <= limits["max_worst_sampled_loss_to_entry_credit"] * credit
        ),
    }
    context = {
        "current_btc_index_usd": result["reference_price"],
        "hybrid_forecast_next_hour": result["hybrid_forecast"],
        "completed_candlestick_patterns": result["candlestick_patterns"],
        "expiry_scenario_distribution": result["expiry_scenario_summary"],
        "current_deribit_pair_quotes": result["selected_pair_quotes"],
        "selected_pair": {
            "expiry_utc": result["expiry"],
            "strike_usd": result["strike"],
            "status": result["candidate_status"],
            "rejection_reasons": result["rejection_reasons"],
            "entry_credit_btc_per_unit": risk["entry_credit"],
            "expected_net_pnl_btc_per_unit": result["expected_net_pnl"],
            "scenario_probability_of_loss": risk["probability_of_loss"],
            "cvar95_loss_btc_per_unit": risk["cvar95"],
            "worst_sampled_loss_btc_per_unit": risk["worst_simulated_loss"],
            "displayed_bid_liquidity": result["liquidity_status"],
            "candlestick_risk_score": result["candlestick_risk_score"],
            "candlestick_veto_threshold": result["candlestick_veto_threshold"],
            "risk_target_checks": selected_risk_targets,
        },
        "model_decision": result["decision"],
        "risk_thresholds": result["risk_thresholds"],
        "candidate_counts": {
            "analyzed": result["candidates_seen"],
            "passed_strategy_filters": result["eligible_candidate_count"],
            "passed_risk_thresholds": result["risk_qualified_candidate_count"],
        },
        "recommended_pair": None if recommendation is None else {
            "expiry_utc": recommendation["expiry"],
            "strike_usd": recommendation["strike"],
            "expected_net_pnl_btc_per_unit": recommendation["expected_net_pnl"],
            "scenario_probability_of_loss": recommendation["risk"]["probability_of_loss"],
            "cvar95_loss_btc_per_unit": recommendation["cvar95"],
            "worst_sampled_loss_btc_per_unit": recommendation["risk"]["worst_simulated_loss"],
        },
        "lower_risk_alternative": result.get("lower_risk_alternative"),
        "data_freshness": {
            "forecast_as_of_utc": result["prediction_as_of_utc"],
            "option_chain_fetched_at": result["catalog_fetched_at"],
            "scenario_count": result["expiry_scenario_count"],
        },
    }
    prompt = (
        "Write exactly two concise sentences, at most 45 words total. In sentence one, state the hybrid model's one-hour BTC return and estimated price, "
        "the detected completed-candle pattern/direction context, and selected-pair loss probability plus CVaR95. Do not call candle patterns probabilities. "
        "In sentence two, state the deterministic decision and why: use exact rejection reasons; name the risk-qualified strike and expected net P&L if present; "
        "otherwise name the lower-risk alternative and its blocker, or say none was found. Use only supplied data; do not invent, repeat, or advise placing a trade."
    )
    try:
        answer = "".join(
            stream_chat_response(
                model,
                [{"role": "user", "content": prompt}],
                context,
                timeout_seconds=45,
                num_predict=120,
            )
        ).strip()
    except OllamaChatError as exc:
        return None, model, str(exc)
    if not answer:
        return None, model, "Ollama returned an empty explanation."
    answer_words = answer.split()
    if len(answer_words) > 45:
        answer = " ".join(answer_words[:45]).rstrip(" ,;:") + "…"
    return answer, model, None


def _fallback_risk_explanation(result: dict[str, Any]) -> str:
    """Provide one concise rule-based explanation when the local model is unavailable."""
    def shorten(text: str) -> str:
        words = text.split()
        return text if len(words) <= 45 else " ".join(words[:45]).rstrip(" ,;:") + "…"

    rejection = "; ".join(result.get("rejection_reasons", ()))
    recommendation = result.get("model_recommendation")
    alternative = result.get("lower_risk_alternative")
    forecast = result["hybrid_forecast"]
    return_value = forecast.get("expected_return")
    price_value = forecast.get("expected_btc_price")
    future_text = (
        "next-hour forecast unavailable"
        if return_value is None or price_value is None
        else f"next-hour BTC {return_value:+.2%} to about ${price_value:,.0f}"
    )
    patterns = [
        f"{item['timeframe']} {', '.join(pattern for pattern in item['patterns'] if pattern != 'no_named_pattern')[:70]} ({item['direction_bias']})"
        for item in result.get("candlestick_patterns", [])
        if any(pattern != "no_named_pattern" for pattern in item["patterns"])
    ]
    pattern_text = "; ".join(patterns[:2]) if patterns else "no named candle pattern"
    loss_probability = result["risk"].get("probability_of_loss")
    loss_text = "unavailable" if loss_probability is None else f"{loss_probability:.1%}"
    first_sentence = (
        f"{future_text}; candles: {pattern_text}; selected-pair loss probability {loss_text}, "
        f"CVaR95 {result['risk']['cvar95']:.3f} BTC/unit."
    )
    if recommendation is not None:
        reason = f"Selected strike rejected: {rejection}. " if rejection else "Selected strike passed filters. "
        return shorten(
            f"{first_sentence} {reason}Risk-qualified strike ${recommendation['strike']:,.0f}; "
            f"expected net P&L {recommendation['expected_net_pnl']:+.3f} BTC/unit."
        )
    if alternative is not None:
        blockers = "; ".join(alternative["rejection_reasons"]) or "it misses one or more risk targets"
        reason = f"Selected strike rejected: {rejection}. " if rejection else "No pair met all risk limits. "
        return shorten(f"{first_sentence} {reason}Lower-risk strike ${alternative['strike']:,.0f} is review-only: {blockers}.")
    if rejection:
        return shorten(f"{first_sentence} NO_TRADE: {rejection}; no safer profitable alternative passed the filters.")
    return shorten(f"{first_sentence} NO_TRADE: no near-ATM pair met the strategy filters and risk limits.")


def _selection_table(selection: DynamicSelection, reference_price: float | None = None) -> pd.DataFrame:
    rows = []
    for item in selection.evaluated_candidates:
        risk_summary = candidate_risk_summary(item)
        rows.append({
            "Status": "RECOMMENDED" if selection.selected is item else item.status.upper(),
            "Expiry (UTC)": item.pair.call.expiry[:10],
            "Strike (USD)": item.pair.call.strike,
            "Moneyness": (f"{item.moneyness_fraction:+.2%}" if item.moneyness_fraction is not None else None),
            "Call bid (BTC)": item.pair.call.bid,
            "Put bid (BTC)": item.pair.put.bid,
            "Call bid size": item.pair.call.bid_size,
            "Put bid size": item.pair.put.bid_size,
            "Call ask size": item.pair.call.ask_size,
            "Put ask size": item.pair.put.ask_size,
            "Total credit (BTC)": item.analysis.entry_credit,
            "Settlement": item.analysis.settlement_currency,
            "Expected net P&L (BTC/unit)": item.expected_net_pnl,
            "Probability of profit": item.probability_of_profit,
            "Probability of loss": risk_summary["probability_of_loss"],
            "CVaR95 (BTC/unit)": item.cvar95,
            "Worst sampled loss (BTC/unit)": risk_summary["worst_simulated_loss"],
            "Risk level": risk_summary["level"],
            "Score (BTC/unit)": item.score,
            "Margin (BTC)": item.margin_requirement,
            "Margin model": item.analysis.margin_model,
            "Combined delta": item.analysis.delta,
            "Liquidity": item.liquidity_status,
            "Scenario": item.scenario_source,
            "Decision / reason": "selected" if item.status == "eligible" and selection.selected is item else "; ".join(item.rejection_reasons),
        })
    return pd.DataFrame(rows)


def render_ai_strategy_research(catalog: LiveMarketCatalog, prediction_data: dict[str, Any],
                                research_frames: dict[str, pd.DataFrame] | None = None,
                                candle_analysis: CandlestickAnalysis | None = None) -> DynamicSelection:
    """Show research forecasts and candidate ranks; this view has no execution actions."""
    strategy_settings = load_strategy_settings()
    scenario_config = strategy_settings["scenario"]
    optimizer_config = strategy_settings["optimizer"]
    prediction = MarketPrediction(**prediction_data)
    candle_age_seconds = _validate_live_prediction_freshness(prediction.as_of_utc)
    candle_analysis = candle_analysis or analyze_candlesticks(
        research_frames or {}, as_of_utc=prediction.as_of_utc,
        config=strategy_settings["candlestick"],
    )
    current_price = float(catalog.reference_price)
    st.subheader("Hybrid forecast and straddle ranking", icon=":material/insights:")
    st.caption(
        f"Forecast updated {pd.to_datetime(prediction.as_of_utc, utc=True):%H:%M UTC} · "
        f"completed BTC candles are {candle_age_seconds / 60:.1f} min old · "
        f"BTC index ${current_price:,.2f} · research only"
    )
    if not prediction.model_available or prediction.expected_return is None:
        st.warning("The hybrid model is unavailable; the selector is forced to NO_TRADE.", title="Forecast unavailable")
        return DynamicSelection(prediction.as_of_utc, 0, (), (), None)

    horizon_return = float(prediction.expected_return)
    forecast_price = current_price * (1.0 + horizon_return)
    with st.container(border=True):
        st.subheader("BTC forecast · next 60 minutes", icon=":material/trending_up:")
        headline = st.columns(4)
        headline[0].metric("Expected return", f"{horizon_return:+.2%}", border=True)
        headline[1].metric("Forecast BTC price", f"${forecast_price:,.2f}", border=True)
        headline[2].metric("Predicted 5m volatility", f"{float(prediction.predicted_volatility or 0.0):.2%}", border=True)
        headline[3].metric("Extreme-move probability", f"{float(prediction.extreme_movement_probability or 0.0):.1%}", border=True)
        direction = st.columns(3)
        direction[0].metric("Bullish", f"{float(prediction.bullish_probability or 0.0):.1%}")
        direction[1].metric("Sideways", f"{float(prediction.sideways_probability or 0.0):.1%}")
        direction[2].metric("Bearish", f"{float(prediction.bearish_probability or 0.0):.1%}")
        quantiles = prediction.return_quantiles or {}
        if .1 in quantiles and .9 in quantiles:
            st.caption(
                f"Model return range: q10 {quantiles[.1]:+.2%} · q50 {quantiles.get(.5, float('nan')):+.2%} · "
                f"q90 {quantiles[.9]:+.2%}. Forecast probabilities and quantiles are uncertain estimates."
            )

    with st.container(border=True):
        st.subheader("Candle pattern analysis", icon=":material/candlestick_chart:")
        overall = st.columns([1, 3], vertical_alignment="center")
        overall[0].metric("Candle risk score", f"{candle_analysis.risk_score} / {candle_analysis.veto_threshold}", border=True)
        if candle_analysis.risk_veto:
            overall[1].warning("Candle risk meets the veto; no short straddle will be recommended.", title="NO_TRADE filter")
        else:
            overall[1].success("Candle risk is below the configured veto for this snapshot.", icon=":material/check_circle:")
        candle_cards = st.columns(2)
        for index, item in enumerate(candle_analysis.timeframes):
            with candle_cards[index % 2].container(border=True):
                if "insufficient_completed_candles" in item.patterns:
                    pattern_names = ["Insufficient completed candles"]
                elif item.patterns == ("no_named_pattern",):
                    pattern_names = ["No distinct pattern"]
                else:
                    pattern_names = [name.replace("_", " ").title() for name in item.patterns]
                st.markdown(f"**{item.timeframe}** · {item.direction_bias.title()} context")
                st.write(", ".join(pattern_names))
                stats = []
                if item.range_expansion_ratio is not None:
                    stats.append(f"Range ×{item.range_expansion_ratio:.2f}")
                if item.volume_ratio is not None:
                    stats.append(f"Volume ×{item.volume_ratio:.2f}")
                stats.append(f"Risk {item.risk_score}")
                st.caption(" · ".join(stats))
        if candle_analysis.warnings:
            st.caption("Data notes: " + " · ".join(candle_analysis.warnings))

    quote_expiries = sorted({q.expiry for q in _catalog_analysis_quotes(catalog)
                             if datetime.fromisoformat(q.expiry.replace("Z", "+00:00")) > datetime.now(timezone.utc)})
    if not quote_expiries:
        st.info("No active same-strike Call–Put expiry is available in the current quote snapshot.")
        return DynamicSelection(prediction.as_of_utc, 0, (), (), None)
    with st.container(border=True):
        st.subheader("Scenario inputs", icon=":material/tune:")
        selected_expiry = st.selectbox(
            "Option expiry",
            quote_expiries,
            format_func=lambda value: value[:10] + " UTC",
            key=f"research_expiry_{catalog.environment}",
        )
        simulation_controls = st.columns(2)
        with simulation_controls[0]:
            simulations = st.slider(
                "Historical price paths",
                scenario_config["minimum_simulations"],
                scenario_config["maximum_simulations"],
                scenario_config["simulation_count"],
                100,
                key=f"scenario_count_v2_{catalog.environment}",
            )
        with simulation_controls[1]:
            block_length = st.slider(
                "Bootstrap block length (5m bars)", 6, 48, scenario_config["block_length"], 3,
                key=f"scenario_block_{catalog.environment}",
            )
    regime = None
    expiry_scenarios = None
    try:
        history_features, history_candles = _cached_historical_research_data()
        if research_frames:
            current_features = build_multitimeframe_features(research_frames)
            if not current_features.empty:
                asof = pd.to_datetime(prediction.as_of_utc, utc=True)
                live_row = current_features.loc[current_features.prediction_time_utc == asof]
                if not live_row.empty:
                    history_features = pd.concat([history_features, live_row[history_features.columns]], ignore_index=True)
        history_fingerprint = (
            len(history_features),
            str(history_features.iloc[-1].get("prediction_time_utc")) if not history_features.empty else "empty",
            len(history_candles),
            str(history_candles.iloc[-1].get("close_time_utc")) if not history_candles.empty else "empty",
        )
        regime_key = (catalog.environment, prediction.as_of_utc, history_fingerprint)
        cached_regime = st.session_state.get("research_regime_cache")
        if cached_regime is not None and cached_regime["key"] == regime_key:
            regime = cached_regime["value"]
        else:
            regime = detect_market_regime(history_features, as_of_utc=prediction.as_of_utc)
            st.session_state["research_regime_cache"] = {"key": regime_key, "value": regime}
        scenario_key = (
            catalog.environment, catalog.fetched_at, prediction.as_of_utc, selected_expiry,
            simulations, block_length, round(current_price, 6), history_fingerprint,
            candle_analysis.risk_score, tuple(sorted(scenario_config.items())),
        )
        cached_scenario = st.session_state.get("research_expiry_scenario_cache")
        if cached_scenario is not None and cached_scenario["key"] == scenario_key:
            scenario = cached_scenario["value"]
        else:
            scenario = forecast_to_expiry(
                reference_price=current_price,
                as_of_utc=prediction.as_of_utc,
                expiry_utc=selected_expiry,
                prediction=prediction,
                historical_candles=history_candles,
                regime=regime,
                simulations=simulations,
                block_length=block_length,
                random_seed=scenario_config["random_seed"],
                price_levels=(current_price * 0.99, current_price * 1.01),
                extreme_move_threshold=scenario_config["extreme_move_threshold"],
                minimum_simulations=scenario_config["minimum_simulations"],
            )
            st.session_state["research_expiry_scenario_cache"] = {
                "key": scenario_key, "value": scenario,
            }
        expiry_scenarios = {selected_expiry: scenario}
        with st.container(border=True):
            st.subheader("Expiry scenario outlook", icon=":material/query_stats:")
            st.caption(f"{selected_expiry[:10]} UTC · {scenario.simulation_count:,} historical moving-block paths · {scenario.conditioning_method}")
            scenario_metrics = st.columns(4)
            scenario_metrics[0].metric("Market regime", regime.current_regime.replace("_", " ").title(), border=True)
            scenario_metrics[1].metric("Scenario median BTC", f"${scenario.terminal_price_quantiles[.5]:,.0f}", border=True)
            scenario_metrics[2].metric("Extreme-move probability", f"{scenario.extreme_move_probability:.1%}", border=True)
            scenario_metrics[3].metric("Comparable historical states", f"{regime.comparable_count:,}", border=True)
            vol_summary = regime.historical_volatility_summary
            matched_vol = vol_summary["matched_median_5m_sigma"]
            st.caption(
                f"Current 5m volatility {vol_summary['current_5m_sigma']:.2%} · "
                f"matched-state median {matched_vol:.2%}" if matched_vol is not None
                else f"Current 5m volatility {vol_summary['current_5m_sigma']:.2%} · matched-state median unavailable"
            )
        hist, edges = np.histogram(scenario.terminal_prices, bins=24)
        distribution = pd.DataFrame({"BTC price bin": edges[:-1], "Scenario count": hist})
        st.markdown("#### Simulated expiry-price distribution")
        st.bar_chart(distribution, x="BTC price bin", y="Scenario count")
        with st.expander("Scenario quantiles and level-crossing probabilities"):
            scenario_rows = [{"Percentile": f"{int(q*100)}%", "BTC at expiry": price,
                              "Return from current": price/current_price-1}
                             for q, price in scenario.terminal_price_quantiles.items()]
            crossing_rows = [{"BTC level": level, "Touch / cross probability": probability}
                             for level, probability in scenario.level_crossing_probabilities.items()]
            table_cols = st.columns(2)
            with table_cols[0]:
                st.dataframe(scenario_rows, hide_index=True, alt="Simulated BTC price quantiles at option expiry")
            with table_cols[1]:
                st.dataframe(crossing_rows, hide_index=True, alt="Scenario probability of crossing BTC levels")
        if scenario.warnings:
            with st.expander("Scenario data notes"):
                for warning in scenario.warnings:
                    st.caption(warning)
    except Exception as exc:
        st.error(f"Historical expiry scenarios are unavailable: {exc}", title="Scenario analysis unavailable")
        return DynamicSelection(prediction.as_of_utc, 0, (), (), None)

    with st.expander("Ranking and risk filters", expanded=False):
        controls = st.columns(3)
        with controls[0]:
            cvar_lambda = st.slider("CVaR penalty λ", 0.0, 5.0, float(optimizer_config["cvar_penalty_lambda"]), 0.25, key="research_cvar_lambda")
            fee_rate = st.number_input("Fee rate (per side)", min_value=0.0, max_value=0.05, value=float(optimizer_config["fee_rate_per_side"]),
                                       step=0.0001, format="%.4f", key="research_fee_rate")
        with controls[1]:
            slippage_rate = st.number_input("Slippage rate (per side)", min_value=0.0, max_value=0.05,
                                            value=float(optimizer_config["slippage_rate_per_side"]), step=0.0001, format="%.4f", key="research_slippage_rate")
            max_delta_input = st.number_input("Max absolute combined delta (0 disables)", min_value=0.0,
                                              max_value=2.0, value=0.0, step=0.05, key="research_max_delta")
        with controls[2]:
            max_spread_input = st.number_input("Max leg spread fraction (0 disables)", min_value=0.0,
                                                max_value=2.0, value=0.0, step=0.05, key="research_max_spread")
            minimum_size_input = st.number_input("Minimum displayed bid size (0 means order quantity)", min_value=0.0,
                                                 value=float(optimizer_config["minimum_bid_size_contracts"]), step=0.01, key="research_min_bid_size")
        max_cvar_input = st.number_input("Maximum CVaR95 in BTC per unit (0 disables)", min_value=0.0,
                                         value=0.0, step=0.0001, format="%.4f", key="research_max_cvar")
        max_loss_default = optimizer_config["maximum_loss_btc"] or 0.0
        max_loss_input = st.number_input("Maximum simulated loss in BTC per unit (0 disables)", min_value=0.0,
                                         value=float(max_loss_default), step=0.0001, format="%.4f", key="research_max_loss")
        max_margin_input = st.number_input("Maximum standard margin estimate (BTC; 0 disables)", min_value=0.0,
                                           value=float(optimizer_config["maximum_margin_btc"] or 0.0), step=0.01,
                                           format="%.4f", key="research_max_margin")
    ranking_key = (
        scenario_key, round(cvar_lambda, 6), round(fee_rate, 8), round(slippage_rate, 8),
        max_spread_input, max_delta_input, max_cvar_input, max_loss_input, max_margin_input,
        minimum_size_input, float(optimizer_config["cvar_confidence"]),
    )
    cached_ranking = st.session_state.get("research_ranking_cache")
    if cached_ranking is not None and cached_ranking["key"] == ranking_key:
        selection = cached_ranking["value"]
    else:
        selection = _rank_live_chain(
            catalog,
            prediction,
            cvar_lambda=cvar_lambda,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            max_spread_fraction=max_spread_input or None,
            max_abs_delta=max_delta_input or None,
            max_cvar95=max_cvar_input or None,
            max_scenario_loss=max_loss_input or None,
            max_margin_requirement=max_margin_input or None,
            cvar_confidence=float(optimizer_config["cvar_confidence"]),
            minimum_bid_size=minimum_size_input or None,
            selected_expiry=selected_expiry,
            expiry_scenarios=expiry_scenarios,
            candlestick_risk_score=candle_analysis.risk_score,
        )
        st.session_state["research_ranking_cache"] = {"key": ranking_key, "value": selection}

    with st.container(border=True):
        st.subheader("Dynamic strike recommendation", icon=":material/target:")
        if selection.selected is not None:
            selected = selection.selected
            risk_summary = candidate_risk_summary(
                selected,
                candlestick_risk_score=candle_analysis.risk_score,
                candlestick_veto_threshold=candle_analysis.veto_threshold,
            )
            risk_color = {
                "ELEVATED": "red",
                "MODERATE": "orange",
                "LOWER IN SAMPLED SCENARIOS": "green",
            }.get(str(risk_summary["level"]), "blue")
            st.badge(f"{risk_summary['level']} · eligible", color=risk_color)
            st.markdown(
                f"**${selected.pair.call.strike:,.0f}** Call + Put · expiry {selected.pair.call.expiry[:10]} UTC · "
                f"{(selected.moneyness_fraction or 0.0):+.2%} vs BTC index"
            )
            risk_metrics = st.columns(4)
            risk_metrics[0].metric("Expected net P&L / unit", f"{selected.expected_net_pnl:+.4f} BTC", border=True)
            probability_loss = risk_summary["probability_of_loss"]
            risk_metrics[1].metric("Scenario loss probability", "Unavailable" if probability_loss is None else f"{probability_loss:.1%}", border=True)
            risk_metrics[2].metric("CVaR95 / unit", f"{selected.cvar95:.4f} BTC", border=True)
            worst_loss = risk_summary["worst_simulated_loss"]
            risk_metrics[3].metric("Worst sampled loss / unit", "Unavailable" if worst_loss is None else f"{worst_loss:.4f} BTC", border=True)
            margin = risk_summary["margin_requirement"]
            st.caption(
                f"Risk-adjusted score {selected.score:+.4f} BTC/unit · "
                f"standard margin {'unavailable' if margin is None else f'{float(margin):.4f} BTC/unit'} · "
                f"{risk_summary['scenario_count']:,} expiry paths. Research estimate; no order was created."
            )
            if st.button(
                "Use this pair in the main selector",
                key=f"use_dynamic_pair_{catalog.environment}",
                help="Updates the dashboard pair for review. It does not submit an order.",
            ):
                st.session_state["pending_research_pair"] = {
                    "environment": catalog.environment,
                    "expiry": selected.pair.call.expiry,
                    "strike": selected.pair.call.strike,
                    "model_as_of_utc": prediction.as_of_utc,
                    "risk_level": risk_summary["level"],
                    "probability_of_loss": risk_summary["probability_of_loss"],
                    "worst_sampled_loss_btc_per_unit": risk_summary["worst_simulated_loss"],
                    "cvar95_btc_per_unit": selected.cvar95,
                    "expected_net_pnl_btc_per_unit": selected.expected_net_pnl,
                    "scenario_count": risk_summary["scenario_count"],
                }
                st.rerun()
            nearby = sorted(selection.evaluated_candidates, key=lambda item: abs(item.pair.call.strike - current_price))
            atm = nearby[0] if nearby else None
            with st.expander("Why this strike?", expanded=False):
                st.write(
                    f"It has the highest expected-net-P&L-minus-CVaR score among candidates that passed the configured filters. "
                    f"Score: {selected.score:+.4f} BTC/unit; expected net P&L: {selected.expected_net_pnl:+.4f} BTC/unit."
                )
                if atm is not None and atm is not selected:
                    st.write(
                        f"Nearest-ATM candidate: ${atm.pair.call.strike:,.0f} · score {atm.score:+.4f} BTC/unit · "
                        f"CVaR95 {atm.cvar95:.4f} BTC/unit."
                    )
        else:
            st.warning(
                f"NO_TRADE · {selection.candidates_seen} pairs checked; none passed the current risk and quote filters.",
                title="No eligible strike",
            )

    table = _selection_table(selection, current_price)
    if not table.empty:
        table_columns = [
            "Status", "Strike (USD)", "Moneyness", "Expected net P&L (BTC/unit)",
            "Probability of loss", "CVaR95 (BTC/unit)", "Score (BTC/unit)", "Liquidity", "Decision / reason",
        ]
        with st.expander(f"Candidate comparison · {len(table)} strikes", expanded=False):
            st.dataframe(
                table[table_columns],
                hide_index=True,
                alt="Risk, return, and rejection reasons for same-expiry strike candidates",
            )
    st.caption("Scenario prices are simulated from historical BTC moves; option fills, future quotes, and account-specific margin are not established.")
    return selection


@st.fragment(run_every="60s")
def render_auto_refreshing_ai_research(environment: str) -> None:
    """Refresh live candle, forecast, option-chain, and pattern analysis every minute."""
    try:
        snapshot = st.session_state.get("ai_research_view_snapshot")
        snapshot_is_fresh = (
            snapshot is not None
            and snapshot["environment"] == environment
            and time.monotonic() - snapshot["cached_at_monotonic"] < 60
        )
        if snapshot_is_fresh:
            catalog = snapshot["catalog"]
            prediction_data = snapshot["prediction_data"]
            research_frames = snapshot["research_frames"]
        else:
            with st.spinner("Refreshing live candles, hybrid forecast, and option chain…"):
                catalog = _cached_live_option_catalog(environment)
                prediction_data, research_frames = _cached_live_hybrid_snapshot()
            st.session_state["ai_research_view_snapshot"] = {
                "environment": environment,
                "cached_at_monotonic": time.monotonic(),
                "catalog": catalog,
                "prediction_data": prediction_data,
                "research_frames": research_frames,
            }
        _validate_live_prediction_freshness(prediction_data["as_of_utc"])
        candle_analysis = analyze_candlesticks(
            research_frames,
            as_of_utc=prediction_data["as_of_utc"],
            config=load_strategy_settings()["candlestick"],
        )
        st.caption("Auto-refreshes every 60 seconds while this tab is open. Candle patterns use completed BTCUSDT bars; option summaries are public Deribit snapshots.")
        render_ai_strategy_research(catalog, prediction_data, research_frames, candle_analysis)
    except Exception as exc:
        st.error(f"Live AI research is unavailable: {exc}", title="No current model decision")


def render_live_chat(context: dict[str, Any]) -> None:
    """Render streaming, open-ended chat powered by the local Ollama service."""
    if "strategy_chat_messages" not in st.session_state:
        st.session_state["strategy_chat_messages"] = [
            {
                "role": "assistant",
                "content": (
                    "Hi, ask me about BTC, live option quotes, or the current AI research. "
                    "I’ll use the latest dashboard snapshot with each message."
                ),
            }
        ]
    messages: list[dict[str, str]] = st.session_state["strategy_chat_messages"]

    st.subheader("Ask about BTC", icon=":material/chat:")
    st.caption(
        "Open-ended AI chat powered by a model running locally in Ollama. "
        "Your messages and current market snapshot stay on this PC."
    )
    selected_model: str | None = None
    with st.container(horizontal=True, vertical_alignment="bottom"):
        if st.button("Refresh models", key="refresh_ollama_models"):
            _cached_ollama_models.clear()
        models, models_error = _cached_ollama_models()
        if models:
            configured_model = os.getenv("OLLAMA_MODEL")
            default_model_index = models.index(configured_model) if configured_model in models else 0
            selected_model = st.selectbox(
                "Local AI model",
                models,
                index=default_model_index,
                key="ollama_model_choice",
            )
        if st.button("Clear chat", key="clear_strategy_chat"):
            messages.clear()
            messages.append({
                "role": "assistant",
                "content": "Chat cleared. Ask about BTC or the current dashboard data.",
            })

    if not models:
        st.warning(
            models_error or "Ollama is running, but no local chat model was found.",
            title="Local AI model unavailable",
        )
        with st.expander("Set up local AI chat"):
            st.markdown(
                "Start the Ollama desktop app, download a chat model from its library, "
                "then choose **Refresh local models** here."
            )
            st.code("ollama pull <model-name>", language="powershell")

    # Reserve the transcript's position before the composer. Filling this
    # placeholder after st.chat_input keeps new user and streamed assistant
    # messages above the composer instead of below it on submit.
    transcript = st.container()
    prompt = st.chat_input(
        "Ask about BTC, live option quotes, or AI research",
        key="strategy_chat_input",
        disabled=not bool(selected_model),
    )
    if prompt:
        messages.append({"role": "user", "content": prompt})
    with transcript:
        for item in messages:
            with st.chat_message(item["role"]):
                st.markdown(item["content"])
        if prompt:
            with st.chat_message("assistant"):
                try:
                    with st.spinner(f"Thinking with {selected_model}…"):
                        response = st.write_stream(
                            stream_chat_response(
                                selected_model or "",
                                messages,
                                context,
                            )
                        )
                    messages.append({"role": "assistant", "content": response})
                except OllamaChatError as exc:
                    st.error(str(exc), title="Local AI chat failed")
            if len(messages) > 101:
                messages[:] = [messages[0], *messages[-100:]]


@st.fragment(run_every="1s")
def render_live_quotes_panel() -> None:
    """Render live option quotes and a short-lived quote replay."""
    catalog: LiveMarketCatalog = st.session_state["active_catalog"]
    feed: DeribitPublicFeed = st.session_state["live_feed"]
    call_contract: OptionContract = st.session_state["live_selected_call"]
    put_contract: OptionContract = st.session_state["live_selected_put"]
    feed_state = feed.snapshot()
    feed_quotes = feed_state["quotes"]
    call_quote = stream_quote(call_contract, feed_quotes.get(call_contract.instrument_id))
    put_quote = stream_quote(put_contract, feed_quotes.get(put_contract.instrument_id))
    call_fresh = has_fresh_quote(call_quote, max_age_seconds=5) and call_quote.is_active
    put_fresh = has_fresh_quote(put_quote, max_age_seconds=5) and put_quote.is_active

    if feed_state["error"]:
        st.badge("Reconnecting to Deribit", icon=":material/sync:", color="orange")
        st.caption(str(feed_state["error"]))
    elif feed_state["connected"]:
        st.badge("Live feed connected", icon=":material/wifi:", color="green")
    else:
        st.badge("Connecting to live feed", icon=":material/pending:", color="blue")

    index_price = feed_state["index_price"] or catalog.reference_price
    with st.container(horizontal=True):
        st.metric("BTC index price", f"${index_price:,.2f}", border=True)
        st.metric("Selected strike", f"${call_contract.strike:,.0f}", border=True)
        st.metric("Selected expiry (UTC)", call_contract.expiry[:10], border=True)
        st.metric(
            "Pair quote status",
            "Fresh" if call_fresh and put_fresh else "Waiting / stale",
            border=True,
        )

    live_rows = []
    for label, contract, quote, fresh in (
        ("Call", call_contract, call_quote, call_fresh),
        ("Put", put_contract, put_quote, put_fresh),
    ):
        live_rows.append(
            {
                "Leg": label,
                "Instrument": contract.instrument_id,
                "Bid": quote.bid,
                "Bid size": quote.bid_size,
                "Ask": quote.ask,
                "Ask size": quote.ask_size,
                "Quote time (UTC)": quote.timestamp,
                "Quote status": "Fresh" if fresh else "Waiting / stale",
            }
        )
    with st.container(border=True):
        st.subheader("Selected pair quotes", icon=":material/candlestick_chart:")
        st.dataframe(
            live_rows,
            hide_index=True,
            alt="Live public Deribit top-of-book quotes for the selected option pair",
        )

    with st.container(border=True):
        st.subheader("Selected pair risk", icon=":material/shield:")
        risk_key = (
            catalog.environment,
            call_contract.instrument_id,
            put_contract.instrument_id,
            call_contract.expiry,
            float(call_contract.strike),
        )
        if st.button(
            "Calculate / refresh risk",
            type="primary",
            key=f"selected_pair_risk_{catalog.environment}_{call_contract.expiry[:10]}_{call_contract.strike:g}",
        ):
            with st.spinner("Calculating scenario risk and asking local Ollama to explain the decision…"):
                try:
                    risk_result = _calculate_selected_pair_risk(
                        catalog.environment,
                        call_contract.expiry,
                        call_contract.strike,
                    )
                    explanation, explanation_model, explanation_error = _ollama_risk_explanation(risk_result)
                    risk_result["ollama_explanation"] = explanation
                    risk_result["ollama_model"] = explanation_model
                    risk_result["ollama_explanation_error"] = explanation_error
                    st.session_state["selected_pair_risk_cache"] = {
                        "key": risk_key,
                        "result": risk_result,
                    }
                    st.session_state.pop("selected_pair_risk_error", None)
                except Exception as exc:
                    st.session_state["selected_pair_risk_error"] = {
                        "key": risk_key,
                        "message": str(exc),
                    }

        risk_cache = st.session_state.get("selected_pair_risk_cache")
        if (
            risk_cache is not None
            and risk_cache.get("key") == risk_key
            and risk_cache.get("result", {}).get("analysis_version") == 2
        ):
            result = risk_cache["result"]
            risk = result["risk"]
            result_age = time.monotonic() - float(result["calculated_monotonic"])
            risk_color = {
                "ELEVATED": "red",
                "MODERATE": "orange",
                "LOWER IN SAMPLED SCENARIOS": "green",
            }.get(risk["level"], "blue")
            st.badge(f"{risk['level']} · {result['candidate_status'].upper()}", color=risk_color)
            if result_age > 60:
                st.warning(
                    "This estimate is over one minute old. Refresh it before relying on the risk figures.",
                    title="Risk estimate may be stale",
                )
            risk_metrics = st.columns(4)
            risk_metrics[0].metric(
                "Scenario probability of loss",
                "Unavailable" if risk["probability_of_loss"] is None else f"{risk['probability_of_loss']:.1%}",
                border=True,
            )
            risk_metrics[1].metric("CVaR95 loss / unit", f"{risk['cvar95']:.3f} BTC", border=True)
            risk_metrics[2].metric(
                "Worst sampled loss / unit",
                "Unavailable" if risk["worst_simulated_loss"] is None else f"{risk['worst_simulated_loss']:.3f} BTC",
                border=True,
            )
            risk_metrics[3].metric("Expected net P&L / unit", f"{result['expected_net_pnl']:+.3f} BTC", border=True)
            more_risk = st.columns(3)
            more_risk[0].metric("Entry credit / unit", f"{risk['entry_credit']:.3f} BTC")
            more_risk[1].metric(
                "Standard margin / unit",
                "Unknown" if risk["margin_requirement"] is None else f"{risk['margin_requirement']:.3f} BTC",
            )
            more_risk[2].metric("Expiry scenarios", f"{result['expiry_scenario_count']:,}")
            if result.get("ollama_explanation"):
                st.markdown(result["ollama_explanation"])
            else:
                st.markdown(_fallback_risk_explanation(result))

            recommendation = result.get("model_recommendation")
            fallback = result.get("lower_risk_alternative")
            if recommendation is not None:
                recommended_risk = recommendation["risk"]
                if st.button(
                    "Apply model-selected strike",
                    key=f"apply_risk_model_pair_{catalog.environment}_{call_contract.expiry[:10]}_{call_contract.strike:g}",
                    help="Sets the main dashboard Call/Put selector to the top eligible pair for this expiry. No order is submitted.",
                ):
                    st.session_state["pending_research_pair"] = {
                        "environment": catalog.environment,
                        "expiry": recommendation["expiry"],
                        "strike": recommendation["strike"],
                        "model_as_of_utc": result["prediction_as_of_utc"],
                        "risk_level": recommended_risk["level"],
                        "probability_of_loss": recommended_risk["probability_of_loss"],
                        "worst_sampled_loss_btc_per_unit": recommended_risk["worst_simulated_loss"],
                        "cvar95_btc_per_unit": recommendation["cvar95"],
                        "expected_net_pnl_btc_per_unit": recommendation["expected_net_pnl"],
                        "scenario_count": result["expiry_scenario_count"],
                    }
                    st.rerun()
            elif fallback is not None:
                fallback_risk = fallback["risk"]
                if st.button(
                    f"Select ${fallback['strike']:,.0f} for review",
                    key=f"review_safer_pair_{catalog.environment}_{call_contract.expiry[:10]}_{call_contract.strike:g}",
                    help="Changes the dashboard pair for inspection only. A rejected pair remains blocked from TestNet entry.",
                ):
                    st.session_state["pending_research_pair"] = {
                        "environment": catalog.environment,
                        "expiry": fallback["expiry"],
                        "strike": fallback["strike"],
                        "model_as_of_utc": result["prediction_as_of_utc"],
                        "risk_level": fallback_risk["level"],
                        "probability_of_loss": fallback_risk["probability_of_loss"],
                        "worst_sampled_loss_btc_per_unit": fallback_risk["worst_simulated_loss"],
                        "cvar95_btc_per_unit": fallback["cvar95"],
                        "expected_net_pnl_btc_per_unit": fallback["expected_net_pnl"],
                        "scenario_count": result["expiry_scenario_count"],
                        "research_only": True,
                        "strategy_status": fallback["status"],
                    }
                    st.rerun()
        else:
            st.info("Risk has not been calculated for this pair in the current session. Select Calculate / refresh risk to run the expiry analysis.")
            risk_error = st.session_state.get("selected_pair_risk_error")
            if risk_error is not None and risk_error.get("key") == risk_key:
                st.error(risk_error["message"], title="Selected-pair risk unavailable")

    pair_key = (call_contract.instrument_id, put_contract.instrument_id)
    if st.session_state.get("history_pair") != pair_key:
        st.session_state["history_pair"] = pair_key
        st.session_state["live_history"] = []
        st.session_state["history_last_recorded"] = 0.0

    now_monotonic = time.monotonic()
    if (
        call_fresh
        and put_fresh
        and now_monotonic - st.session_state.get("history_last_recorded", 0.0) >= 0.9
    ):
        history = st.session_state["live_history"]
        history.append(
            {
                "Time (UTC)": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                "BTC index": index_price,
                "Call bid": call_quote.bid,
                "Call ask": call_quote.ask,
                "Put bid": put_quote.bid,
                "Put ask": put_quote.ask,
            }
        )
        del history[:-300]
        st.session_state["history_last_recorded"] = now_monotonic

    history = st.session_state["live_history"]
    if history:
        st.subheader("Recent quote history", icon=":material/history:")
        # BTC index is USD while option premiums are BTC. Keep them on
        # separate grids so the index scale cannot flatten the option quotes.
        history_frame = pd.DataFrame(history)
        history_frame["Chart time"] = history_frame["Time (UTC)"].str[-8:]
        chart_source = history_frame[
            ["Chart time", "BTC index", "Call bid", "Call ask", "Put bid", "Put ask"]
        ]
        label_interval = max(1, (len(chart_source) + 7) // 8)
        quote_history_chart = {
            "animation": False,
            "dataset": {"source": chart_source},
            "legend": {"top": 0, "type": "scroll"},
            "tooltip": {"trigger": "axis", "axisPointer": {"type": "cross"}},
            "grid": [
                {"left": 86, "right": 28, "top": 40, "height": "27%"},
                {"left": 86, "right": 28, "top": "56%", "height": "27%"},
            ],
            "xAxis": [
                {
                    "type": "category", "gridIndex": 0, "boundaryGap": False,
                    "axisLabel": {"show": False}, "axisTick": {"show": False},
                    "splitLine": {"show": False},
                },
                {
                    "type": "category", "gridIndex": 1, "boundaryGap": False,
                    "axisLabel": {"interval": label_interval, "hideOverlap": True},
                    "splitLine": {"show": False},
                },
            ],
            "yAxis": [
                {
                    "type": "value", "gridIndex": 0, "name": "BTC index · USD",
                    "scale": True, "axisLabel": {"formatter": "${value}"},
                    "splitLine": {"lineStyle": {"color": "#25344B"}},
                },
                {
                    "type": "value", "gridIndex": 1, "name": "Option premium · BTC",
                    "scale": True, "axisLabel": {"formatter": "{value} BTC"},
                    "splitLine": {"lineStyle": {"color": "#25344B"}},
                },
            ],
            "dataZoom": [
                {"type": "inside", "xAxisIndex": [0, 1], "start": 0, "end": 100},
                {"type": "slider", "xAxisIndex": [0, 1], "start": 0, "end": 100, "bottom": 0},
            ],
            "series": [
                {
                    "name": "BTC index", "type": "line", "xAxisIndex": 0, "yAxisIndex": 0,
                    "showSymbol": False, "lineStyle": {"width": 2, "color": "#F7931A"},
                    "encode": {"x": "Chart time", "y": "BTC index"},
                },
                {
                    "name": "Call bid", "type": "line", "xAxisIndex": 1, "yAxisIndex": 1,
                    "showSymbol": False, "lineStyle": {"width": 2, "color": "#18C784"},
                    "encode": {"x": "Chart time", "y": "Call bid"},
                },
                {
                    "name": "Call ask", "type": "line", "xAxisIndex": 1, "yAxisIndex": 1,
                    "showSymbol": False, "lineStyle": {"width": 2, "color": "#EA5966"},
                    "encode": {"x": "Chart time", "y": "Call ask"},
                },
                {
                    "name": "Put bid", "type": "line", "xAxisIndex": 1, "yAxisIndex": 1,
                    "showSymbol": False, "lineStyle": {"width": 2, "color": "#7398FF"},
                    "encode": {"x": "Chart time", "y": "Put bid"},
                },
                {
                    "name": "Put ask", "type": "line", "xAxisIndex": 1, "yAxisIndex": 1,
                    "showSymbol": False, "lineStyle": {"width": 2, "color": "#B28AE8"},
                    "encode": {"x": "Chart time", "y": "Put ask"},
                },
            ],
        }
        st.echarts_chart(
            quote_history_chart,
            height=500,
            alt="BTC index history in USD and Call/Put option bid and ask history in BTC, shown on separate scales.",
            key="btc_option_quote_history_chart",
        )
        st.caption("Upper scale: BTC index in USD · lower scale: option premiums in BTC · timestamps are UTC. Use the slider to zoom into the quote window.")
        csv_buffer = io.StringIO()
        writer = csv.DictWriter(csv_buffer, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
        st.download_button(
            "Download quote history CSV",
            data=csv_buffer.getvalue(),
            file_name="deribit_btc_option_quote_history.csv",
            mime="text/csv",
        )
    else:
        st.info("Waiting for fresh Call and Put quotes to build the recent quote history.")

@st.fragment(run_every="1s")
def render_testnet_execution_controls() -> None:
    """Render explicit TestNet login and the NautilusTrader-managed strategy."""
    settings = Settings.from_env()
    order_submission_enabled = _testnet_order_submission_enabled(settings)
    call_contract: OptionContract = st.session_state["live_selected_call"]
    put_contract: OptionContract = st.session_state["live_selected_put"]
    pair_key = (call_contract.instrument_id, put_contract.instrument_id)
    pair_token = "__".join(pair_key)
    confirmation_key = f"testnet_order_confirmed_{pair_token}"
    if st.session_state.pop("testnet_confirmation_reset_pair", None) == pair_token:
        st.session_state[confirmation_key] = False
    if st.session_state.get("testnet_execution_pair") != pair_key:
        st.session_state["testnet_execution_pair"] = pair_key
        st.session_state.pop("testnet_open_orders", None)
        st.session_state.pop("testnet_last_execution_report", None)
    feed: DeribitPublicFeed = st.session_state["live_feed"]
    quote_map = feed.snapshot()["quotes"]
    call_quote = stream_quote(call_contract, quote_map.get(call_contract.instrument_id))
    put_quote = stream_quote(put_contract, quote_map.get(put_contract.instrument_id))
    quotes_fresh = (
        has_fresh_quote(call_quote, max_age_seconds=5)
        and has_fresh_quote(put_quote, max_age_seconds=5)
        and call_quote.is_active
        and put_quote.is_active
    )

    st.subheader("TestNet account and orders", icon=":material/shield:")
    st.warning(
        "TestNet uses a separate Deribit account and synthetic funds. Starting the strategy connects NautilusTrader to this account. "
        "Entry orders are separate; the strategy tracks fills, cancels stale unmatched entries, and reduces one-sided exposure. "
        "A 1% stop-loss and 2% take-profit trigger reduce-only market closes.",
        title="TestNet execution",
    )
    if not settings.deribit_testnet:
        st.error("Authenticated execution is disabled because DERIBIT_TESTNET is not true.")
        return

    client: DeribitTestNetClient | None = st.session_state.get("deribit_testnet_client")
    if client is None:
        st.caption(
            "Credentials are read only from the project server environment (.env or deployment secrets). "
            "They are never requested in the browser or shown on the page."
        )
        if st.button("Connect TestNet account", type="primary", key="testnet_connect_button"):
            try:
                client_id, client_secret = _testnet_credentials()
                candidate = DeribitTestNetClient(
                    client_id=client_id,
                    _client_secret=client_secret,
                    order_submission_enabled=order_submission_enabled,
                )
                auth_info = candidate.authenticate()
                summary = candidate.get_account_summary("BTC")
                st.session_state["deribit_testnet_client"] = candidate
                st.session_state["deribit_testnet_account_summary"] = {
                    key: summary.get(key)
                    for key in (
                        "balance",
                        "equity",
                        "available_funds",
                        "initial_margin",
                    )
                    if summary.get(key) is not None
                }
                st.session_state["deribit_testnet_auth_info"] = auth_info
                client = candidate
                st.success("Authenticated with the Deribit TestNet account.")
            except (ValueError, DeribitTestNetError) as exc:
                st.error(str(exc), title="TestNet connection failed")

    if client is None:
        st.info(
            "Add DERIBIT_TESTNET_CLIENT_ID and DERIBIT_TESTNET_CLIENT_SECRET to the server-side .env file, "
            "then connect here. The app does not connect automatically."
        )
        return

    summary: dict[str, Any] = st.session_state.get("deribit_testnet_account_summary", {})
    auth_info: dict[str, Any] = st.session_state.get("deribit_testnet_auth_info", {})
    st.success(
        f"TestNet account connected · scopes: {auth_info.get('scope') or 'granted by API key'} · "
        "tokens refresh automatically when needed."
    )
    summary_fields = [
        ("Balance (BTC)", "balance"),
        ("Equity (BTC)", "equity"),
        ("Available funds (BTC)", "available_funds"),
        ("Initial margin (BTC)", "initial_margin"),
    ]
    account_values = [
        (label, summary[key]) for label, key in summary_fields if summary.get(key) is not None
    ]
    if account_values:
        metric_cols = st.columns(len(account_values))
        for column, (label, value) in zip(metric_cols, account_values):
            with column:
                try:
                    st.metric(label, f"{float(value):.8f}", border=True)
                except (TypeError, ValueError):
                    st.metric(label, str(value), border=True)

    active_service: DeribitNautilusService | None = st.session_state.get("testnet_nautilus_service")
    active_snapshot = active_service.read() if active_service is not None else {}
    active_positions = active_snapshot.get("positions", {})
    may_disconnect = active_service is None or (
        active_snapshot.get("node_state") == "stopped"
        and not any(float(value) > 1e-10 for value in active_positions.values())
    )
    button_col, info_col = st.columns([1, 2])
    with button_col:
        if st.button(
            "Disconnect this session",
            key="testnet_disconnect_button",
            disabled=not may_disconnect,
        ):
            st.session_state.pop("deribit_testnet_client", None)
            st.session_state.pop("deribit_testnet_account_summary", None)
            st.session_state.pop("deribit_testnet_auth_info", None)
            st.session_state.pop("testnet_open_orders", None)
            st.rerun(scope="fragment")
    with info_col:
        if order_submission_enabled:
            st.caption("TestNet order submission is enabled in server configuration.")
        else:
            st.caption(
                "Order submission is disabled. Set DERIBIT_TESTNET_ORDER_SUBMISSION_ENABLED=true "
                "in .env and restart Streamlit to enable it."
            )

    call_min = call_contract.min_trade_amount
    put_min = put_contract.min_trade_amount
    min_amount = max(call_min or 0.001, put_min or 0.001)
    price_step_call = call_contract.tick_size or 0.0001
    price_step_put = put_contract.tick_size or 0.0001
    default_call_price = (
        call_quote.ask if call_quote.ask is not None and call_quote.ask > 0 else price_step_call
    )
    default_put_price = (
        put_quote.ask if put_quote.ask is not None and put_quote.ask > 0 else price_step_put
    )

    service: DeribitNautilusService | None = st.session_state.get("testnet_nautilus_service")
    service_matches_pair = bool(
        service is not None
        and service.call_name == call_contract.instrument_id
        and service.put_name == put_contract.instrument_id
    )
    if service is not None and not service_matches_pair:
        previous = service.read()
        if previous.get("node_state") in {"stopped", "error"} and not any(
            float(value) > 1e-10 for value in previous.get("positions", {}).values()
        ):
            st.session_state["testnet_nautilus_service"] = None
            service = None
        else:
            st.warning(
                f"The automated monitor is still assigned to {service.call_name} and {service.put_name}. "
                "Switch back to that pair to manage it, or stop it after it is flat before selecting another pair.",
                title="Another pair is being monitored",
            )

    if service is None:
        st.markdown("#### NautilusTrader TestNet strategy")
        if not order_submission_enabled:
            st.info("Enable DERIBIT_TESTNET_ORDER_SUBMISSION_ENABLED to start the authenticated TestNet node.")
        elif st.button(
            "Start NautilusTrader TestNet strategy",
            type="primary",
            disabled=not client.authenticated,
            key="testnet_start_nautilus_strategy",
        ):
            try:
                service = _create_testnet_strategy_service(
                    client,
                    call_contract,
                    put_contract,
                    min_amount,
                    order_submission_enabled=order_submission_enabled,
                )
                st.session_state["testnet_nautilus_service"] = service
            except (ValueError, RuntimeError) as exc:
                st.error(str(exc), title="Could not start the TestNet strategy")

    service = st.session_state.get("testnet_nautilus_service")
    service_matches_pair = bool(
        service is not None
        and service.call_name == call_contract.instrument_id
        and service.put_name == put_contract.instrument_id
    )
    service_snapshot = service.read() if service_matches_pair else {}
    if service_matches_pair:
        st.markdown("#### NautilusTrader strategy status")
        node_state = service_snapshot.get("node_state", "starting")
        strategy_state = service_snapshot.get("strategy_state", "starting")
        if node_state == "error" or strategy_state in {"error", "manual_review"}:
            st.error(service_snapshot.get("message", "The strategy needs manual review."), title="Strategy needs attention")
        elif strategy_state in {"ready", "open", "entry_working", "exit_working", "recovering"}:
            st.info(service_snapshot.get("message", "TestNet strategy is running."), title=f"{node_state.title()} · {strategy_state.replace('_', ' ').title()}")
        else:
            st.warning(service_snapshot.get("message", "Connecting and reconciling the TestNet account."), title=f"{node_state.title()} · {strategy_state.replace('_', ' ').title()}")
        position_values = service_snapshot.get("positions", {})
        metric_cols = st.columns(4)
        metrics = [
            ("Call open (BTC)", position_values.get("call", 0.0)),
            ("Put open (BTC)", position_values.get("put", 0.0)),
            ("Combined entry premium", service_snapshot.get("entry_premium")),
            ("Current buyback ask", service_snapshot.get("buyback_premium")),
        ]
        for column, (label, value) in zip(metric_cols, metrics):
            with column:
                st.metric(label, "—" if value is None else f"{float(value):.8f}")
        threshold_values = [
            ("Stop-loss", service_snapshot.get("stop_loss")),
            ("Take-profit", service_snapshot.get("take_profit")),
        ]
        for label, value in threshold_values:
            if value is not None:
                st.caption(f"{label}: {float(value):.8f} BTC combined option premium")
        st.caption(f"Selected-pair open orders: {service_snapshot.get('open_orders', 'unknown')}")
        if service_snapshot.get("events"):
            with st.expander("NautilusTrader order and recovery events", expanded=False):
                st.dataframe(service_snapshot["events"], hide_index=True, alt="Recent TestNet strategy events")
        flat = not any(float(value) > 1e-10 for value in position_values.values())
        can_stop = (
            flat
            and service_snapshot.get("open_orders") == 0
            and strategy_state in {"ready", "closed", "manual_review", "error"}
            and node_state == "running"
        )
        if can_stop:
            if st.button("Stop flat TestNet strategy", key="testnet_stop_nautilus_strategy"):
                try:
                    service.stop()
                except RuntimeError as exc:
                    st.error(str(exc), title="Cannot stop the strategy")
        elif node_state in {"stopped", "error"} and flat:
            if st.button("Start or reconnect TestNet strategy", key="testnet_restart_nautilus_strategy"):
                try:
                    service.start()
                except RuntimeError as exc:
                    st.error(str(exc), title="Could not restart the strategy")

    active_pair_note = st.session_state.get("active_model_recommendation") or {}
    research_only_selected_pair = bool(
        active_pair_note.get("research_only")
        and active_pair_note.get("environment") == catalog.environment
        and active_pair_note.get("expiry") == call_contract.expiry
        and float(active_pair_note.get("strike", float("nan"))) == float(call_contract.strike)
    )
    submission_ready = (
        order_submission_enabled
        and client.authenticated
        and service_matches_pair
        and service_snapshot.get("node_state") == "running"
        and service_snapshot.get("strategy_state") == "ready"
        and quotes_fresh
        and call_min is not None
        and put_min is not None
        and call_contract.tick_size is not None
        and put_contract.tick_size is not None
        and not research_only_selected_pair
    )
    st.markdown("#### Submit a managed TestNet short straddle")
    st.caption(
        "Order amounts are BTC per option leg. Limit prices use the option's BTC price unit. "
        "NautilusTrader submits both GTC post-only sells and receives their exchange order and fill events. "
        "After entry, risk exits use reduce-only market buys."
    )
    if research_only_selected_pair:
        st.warning(
            "TestNet entry is disabled for this review-only alternative because it failed a strategy/risk filter. "
            "Choose a pair that passes the live analysis filters before submitting.",
            title="Review-only strike",
        )
    with st.form(
        f"testnet_straddle_order_form_{pair_token}",
        enter_to_submit=False,
        clear_on_submit=False,
    ):
        order_amount = st.number_input(
            "Amount per leg (BTC)",
            min_value=float(min_amount),
            max_value=100.0,
            value=float(min_amount),
            step=float(min_amount),
            format="%.8f",
            key=f"testnet_order_amount_btc_{pair_token}",
        )
        price_cols = st.columns(2)
        with price_cols[0]:
            call_limit = st.number_input(
                f"Call limit price (BTC) · {call_contract.instrument_id}",
                min_value=float(price_step_call),
                max_value=10.0,
                value=float(default_call_price),
                step=float(price_step_call),
                format="%.8f",
                key=f"testnet_call_limit_btc_{pair_token}",
            )
        with price_cols[1]:
            put_limit = st.number_input(
                f"Put limit price (BTC) · {put_contract.instrument_id}",
                min_value=float(price_step_put),
                max_value=10.0,
                value=float(default_put_price),
                step=float(price_step_put),
                format="%.8f",
                key=f"testnet_put_limit_btc_{pair_token}",
            )
        confirmed = st.checkbox(
            "I understand this submits two separate TestNet orders and either leg may remain open or fill alone.",
            key=confirmation_key,
        )
        submitted = st.form_submit_button(
            "Submit managed TestNet entry",
            type="primary",
            disabled=not submission_ready,
            width="stretch",
        )
    if submitted:
        st.session_state["testnet_confirmation_reset_pair"] = pair_token
        if not confirmed:
            st.error("Confirm the separate-leg TestNet order risk before submitting.")
        elif not quotes_fresh:
            st.error("The selected pair does not have two fresh TestNet quotes. Wait for the feed and retry.")
        elif not service_matches_pair or service is None:
            st.error("Start the NautilusTrader TestNet strategy for this pair before submitting an entry.")
        else:
            try:
                DeribitTestNetClient._validate_amount(order_amount, call_contract)
                DeribitTestNetClient._validate_amount(order_amount, put_contract)
                DeribitTestNetClient._validate_price(call_limit, call_contract)
                DeribitTestNetClient._validate_price(put_limit, put_contract)
                service.submit_entry(order_amount, call_limit, put_limit)
                st.success("Entry request queued. NautilusTrader will report each acceptance, fill, or recovery action here.")
            except (DeribitTestNetError, RuntimeError, ValueError) as exc:
                st.error(str(exc), title="TestNet entry was not queued")

    st.markdown("#### Open orders for this pair")
    if st.button("Refresh selected-pair TestNet orders", key="testnet_refresh_orders"):
        try:
            open_orders = [
                order
                for instrument_id in (call_contract.instrument_id, put_contract.instrument_id)
                for order in client.get_open_orders(instrument_id)
            ]
            st.session_state["testnet_open_orders"] = open_orders
        except DeribitTestNetError as exc:
            st.error(str(exc), title="Could not read TestNet open orders")
    open_orders = st.session_state.get("testnet_open_orders", [])
    if open_orders:
        st.dataframe(
            [
                {
                    "Order ID": order.get("order_id"),
                    "Instrument": order.get("instrument_name"),
                    "Side": order.get("direction"),
                    "Status": order.get("order_state"),
                    "Price": order.get("price"),
                    "Amount (BTC)": order.get("amount"),
                    "Filled (BTC)": order.get("filled_amount"),
                }
                for order in open_orders
            ],
            hide_index=True,
            alt="Open TestNet orders for the selected BTC Call and Put",
        )
        for order in open_orders:
            order_id = str(order.get("order_id") or "")
            instrument = str(order.get("instrument_name") or "BTC option")
            if not order_id:
                continue
            with st.form(f"cancel_testnet_order_{order_id}", enter_to_submit=False):
                st.caption(f"{instrument} · {order_id}")
                cancel_confirmed = st.checkbox(
                    "Confirm cancellation of this open TestNet order",
                    key=f"confirm_cancel_{order_id}",
                )
                cancel_clicked = st.form_submit_button(
                    "Cancel this TestNet order",
                    disabled=str(order.get("order_state", "")).lower() != "open",
                )
            if cancel_clicked:
                if not cancel_confirmed:
                    st.error(f"Confirm cancellation for {order_id}.")
                else:
                    try:
                        result = client.cancel_order(order_id)
                        st.success(
                            f"Cancellation requested for {order_id}; state: "
                            f"{result.get('order_state', 'see Deribit order status')}."
                        )
                        st.session_state.pop("testnet_open_orders", None)
                    except DeribitTestNetError as exc:
                        st.error(str(exc), title="TestNet cancellation failed")

    st.subheader("Quote history", icon=":material/history:")
    history = st.session_state["live_history"]
    if len(history) < 2:
        st.info("Collecting fresh live quotes. The chart and replay appear after two samples.")
    else:
        st.line_chart(
            history,
            x="Time (UTC)",
            y=["Call bid", "Call ask", "Put bid", "Put ask"],
            height=260,
            alt="Selected call and put bid and ask history from this browser session",
        )
        replay_key = "history_replay_index"
        if st.session_state.get(replay_key, len(history) - 1) >= len(history):
            st.session_state[replay_key] = len(history) - 1
        replay_index = st.slider(
            "Replay history sample",
            min_value=0,
            max_value=len(history) - 1,
            value=len(history) - 1,
            key=replay_key,
        )
        st.dataframe(
            [history[replay_index]],
            hide_index=True,
            alt="Selected historical quote sample for replay",
        )
        csv_buffer = io.StringIO()
        writer = csv.DictWriter(csv_buffer, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
        st.download_button(
            "Download quote history CSV",
            data=csv_buffer.getvalue(),
            file_name="deribit_btc_option_quote_history.csv",
            mime="text/csv",
        )


st.title("BTC analytics terminal", icon=":material/monitoring:")
st.caption("Live Bitcoin market data · dynamic options research · Deribit connectivity")

if st.session_state.get("market_data_source") not in {"Live Deribit", "Deribit TestNet"}:
    # Migrate older browser sessions that still stored the removed sample mode.
    st.session_state["market_data_source"] = "Live Deribit"
data_source = st.session_state.get("market_data_source", "Live Deribit")
active_mode_button = "select_live_deribit" if data_source == "Live Deribit" else "select_deribit_testnet"
active_mode_color = "#F7931A" if data_source == "Live Deribit" else "#8B7BFF"
active_mode_end = "#FFC36B" if data_source == "Live Deribit" else "#B8A9FF"
active_mode_shadow = "rgba(247,147,26,.20)" if data_source == "Live Deribit" else "rgba(139,123,255,.24)"
active_mode_text = "#111827"
st.html(
    f"""
    <style>
    .st-key-overview_page {{
      background: linear-gradient(145deg, rgba(247,147,26,.10), rgba(18,35,59,.92) 58%, rgba(12,22,38,.96));
      border: 1px solid rgba(247,147,26,.32); border-radius: 18px; padding: 1.1rem;
    }}
    .st-key-research_page {{
      background: linear-gradient(145deg, rgba(139,123,255,.13), rgba(22,25,48,.95) 55%, rgba(13,19,36,.98));
      border: 1px solid rgba(139,123,255,.38); border-radius: 18px; padding: 1.1rem;
    }}
    .st-key-chat_page {{
      background: linear-gradient(145deg, rgba(25,190,165,.12), rgba(12,39,46,.95) 55%, rgba(10,22,35,.98));
      border: 1px solid rgba(25,190,165,.34); border-radius: 18px; padding: 1.1rem;
    }}
    .st-key-testnet_page {{
      background: linear-gradient(145deg, rgba(238,132,49,.13), rgba(43,29,24,.95) 55%, rgba(19,22,34,.98));
      border: 1px solid rgba(238,132,49,.38); border-radius: 18px; padding: 1.1rem;
    }}
    .st-key-select_live_deribit button, .st-key-select_deribit_testnet button {{
      min-height: 3rem; border-radius: 12px; font-weight: 650; transition: all .18s ease;
    }}
    .st-key-select_live_deribit button {{ border-color: rgba(247,147,26,.58); }}
    .st-key-select_deribit_testnet button {{ border-color: rgba(139,123,255,.62); }}
    .st-key-{active_mode_button} button {{
      background: linear-gradient(120deg, {active_mode_color}, {active_mode_end}) !important;
      border-color: {active_mode_color} !important; color: {active_mode_text} !important;
      box-shadow: 0 5px 18px {active_mode_shadow}; font-weight: 750 !important;
    }}
    </style>
    """
)
background_testnet_service: DeribitNautilusService | None = st.session_state.get("testnet_nautilus_service")
if data_source != "Deribit TestNet" and background_testnet_service is not None:
    background_state = background_testnet_service.read()
    if background_state.get("node_state") in {"starting", "running", "stopping"}:
        st.warning(
            f"The TestNet strategy is still {background_state.get('node_state')} for "
            f"{background_testnet_service.call_name} and {background_testnet_service.put_name}. "
            "Switch to Deribit TestNet to view its orders and risk status.",
            title="Background TestNet strategy active",
        )

st.session_state.setdefault("live_catalog", None)
st.session_state.setdefault("live_catalog_error", None)
st.session_state.setdefault("testnet_catalog", None)
st.session_state.setdefault("testnet_catalog_error", None)
st.session_state.setdefault("live_feed", None)
st.session_state.setdefault("live_feed_signature", None)
st.session_state.setdefault("live_history", [])
st.session_state.setdefault("history_pair", None)
st.session_state.setdefault("history_last_recorded", 0.0)
st.session_state.setdefault("testnet_nautilus_service", None)

with st.sidebar:
    st.subheader("Workspace", icon=":material/dashboard:")
    data_source = st.session_state.get("market_data_source", "Live Deribit")
    with st.container(border=True):
        st.caption("Choose market environment")
        st.badge(
            "Production · read only" if data_source == "Live Deribit" else "TestNet · simulated account",
            icon=":material/wifi:" if data_source == "Live Deribit" else ":material/shield:",
            color="green" if data_source == "Live Deribit" else "orange",
        )
        st.button(
            "Live Deribit",
            icon=":material/wifi:",
            type="primary" if data_source == "Live Deribit" else "secondary",
            width="stretch",
            key="select_live_deribit",
            on_click=_select_market_data_source,
            args=("Live Deribit",),
        )
        st.button(
            "Deribit TestNet",
            icon=":material/shield:",
            type="primary" if data_source == "Deribit TestNet" else "secondary",
            width="stretch",
            key="select_deribit_testnet",
            on_click=_select_market_data_source,
            args=("Deribit TestNet",),
        )
    data_source = st.session_state.get("market_data_source", "Live Deribit")
    previous_source = st.session_state.get("previous_market_data_source")
    if previous_source != data_source:
        previous_feed: DeribitPublicFeed | None = st.session_state.get("live_feed")
        if previous_feed is not None:
            previous_feed.stop()
        st.session_state["live_feed"] = None
        st.session_state["live_feed_signature"] = None
        if previous_source == "Deribit TestNet":
            service: DeribitNautilusService | None = st.session_state.get("testnet_nautilus_service")
            service_state = service.read() if service is not None else {}
            if service is None or service_state.get("node_state") == "stopped":
                st.session_state.pop("deribit_testnet_client", None)
                st.session_state.pop("deribit_testnet_account_summary", None)
                st.session_state.pop("deribit_testnet_auth_info", None)
                st.session_state.pop("testnet_open_orders", None)
        st.session_state["previous_market_data_source"] = data_source

    if data_source == "Live Deribit":
        st.caption("Refresh the data before moving to another button or section.")
        if st.button("Load / refresh BTC option chain", type="primary", width="stretch", key="refresh_live_catalog"):
            st.session_state["live_catalog_error"] = None
            with st.spinner("Fetching the BTC index, open options, and chain summary..."):
                try:
                    st.session_state["live_catalog"] = deribit_live.fetch_live_btc_option_catalog()
                except Exception as exc:
                    st.session_state["live_catalog_error"] = str(exc)
    elif data_source == "Deribit TestNet":
        st.caption("Refresh the data before moving to another button or section.")
        if st.button(
            "Load / refresh TestNet BTC option chain",
            type="primary",
            width="stretch",
            key="refresh_testnet_catalog",
        ):
            st.session_state["testnet_catalog_error"] = None
            with st.spinner("Fetching the TestNet BTC index, options, and chain summary..."):
                try:
                    st.session_state["testnet_catalog"] = (
                        deribit_live.fetch_testnet_btc_option_catalog()
                    )
                except Exception as exc:
                    st.session_state["testnet_catalog_error"] = str(exc)
    st.caption("Live Deribit is read-only. TestNet execution requires an explicit server-side opt-in.")

if data_source in {"Live Deribit", "Deribit TestNet"}:
    is_testnet = data_source == "Deribit TestNet"
    catalog_key = "testnet_catalog" if is_testnet else "live_catalog"
    error_key = "testnet_catalog_error" if is_testnet else "live_catalog_error"
    environment_label = "Deribit TestNet" if is_testnet else "Deribit production"
    st.header("Deribit TestNet" if is_testnet else "Live Deribit", icon=":material/candlestick_chart:")
    st.caption(
        "Synthetic-fund account controls and strategy execution are available after explicit opt-in."
        if is_testnet
        else "Public production market data, streaming quotes, and AI research. No authenticated account or exchange orders are used."
    )
    st.badge(
        "TestNet · opt-in execution" if is_testnet else "Production · read-only",
        icon=":material/shield:" if is_testnet else ":material/wifi:",
        color="orange" if is_testnet else "green",
    )
    if st.session_state[error_key]:
        st.error(st.session_state[error_key], title="Deribit request failed")
    catalog: LiveMarketCatalog | None = st.session_state[catalog_key]
    if catalog is None:
        st.subheader("Live BTC market", icon=":material/candlestick_chart:")
        st.info(
            f"The BTC chart is available now. Load the {environment_label} option chain from the sidebar to enable pair selection and the remaining market views.",
            title="Options chain not loaded",
        )
        with st.container(key="overview_page"):
            render_btc_candlestick_chart()
        st.stop()
    st.session_state["active_catalog"] = catalog

    # Apply a candidate selected from the AI research tab before creating the
    # expiry/strike widgets, so the overview and TestNet share it.
    pending_pair = st.session_state.pop("pending_research_pair", None)
    if pending_pair is not None:
        # Accept an older in-session tuple if the app hot-reloaded while open.
        if isinstance(pending_pair, tuple):
            pending_pair = {
                "environment": pending_pair[0],
                "expiry": pending_pair[1],
                "strike": pending_pair[2],
            }
    if pending_pair is not None and pending_pair["environment"] == catalog.environment:
        candidate_expiry = pending_pair["expiry"]
        candidate_strike = pending_pair["strike"]
        st.session_state[f"{catalog.environment}_expiry_selector"] = candidate_expiry
        st.session_state[f"{catalog.environment}_strike_selector"] = candidate_strike
        st.session_state["active_model_recommendation"] = pending_pair

    # Build the matching pairs with an indexed intersection. Scanning every
    # Call against every Put here made each Streamlit rerun quadratic in the
    # size of the live Deribit chain, including every tab switch.
    calls_by_pair: dict[tuple[str, float], OptionContract] = {}
    puts_by_pair: dict[tuple[str, float], OptionContract] = {}
    for contract in catalog.calls:
        calls_by_pair.setdefault((contract.expiry, contract.strike), contract)
    for contract in catalog.puts:
        puts_by_pair.setdefault((contract.expiry, contract.strike), contract)
    common_pairs = calls_by_pair.keys() & puts_by_pair.keys()
    expiry_pairs: dict[str, set[float]] = {}
    for expiry, strike in common_pairs:
        expiry_pairs.setdefault(expiry, set()).add(strike)
    expiries = sorted(expiry_pairs)
    if not expiries:
        st.error("The current Deribit catalog has no matching Call/Put pairs.")
        render_btc_candlestick_chart()
        st.stop()
    st.subheader("Option chain", icon=":material/table_chart:")
    st.caption("Choose an expiry and the strike nearest to the current BTC index.")
    with st.container(border=True):
        selector_col, strike_col = st.columns([1, 1], gap="large")
        with selector_col:
            selected_expiry = st.selectbox(
                "Expiry (UTC)",
                expiries,
                index=0,
                format_func=lambda value: value[:10],
                key=f"{catalog.environment}_expiry_selector",
            )
        strikes = sorted(expiry_pairs[selected_expiry])
        selected_strike = min(strikes, key=lambda value: abs(value - catalog.reference_price))
        with strike_col:
            selected_strike = st.selectbox(
                "Strike (USD)",
                strikes,
                index=strikes.index(selected_strike),
                format_func=lambda value: f"${value:,.0f}",
                key=f"{catalog.environment}_strike_selector",
            )

    call_contract = calls_by_pair[(selected_expiry, selected_strike)]
    put_contract = puts_by_pair[(selected_expiry, selected_strike)]
    model_recommendation = st.session_state.get("active_model_recommendation")
    if model_recommendation and model_recommendation.get("environment") == catalog.environment:
        recommendation_matches = (
            model_recommendation.get("expiry") == selected_expiry
            and model_recommendation.get("strike") == selected_strike
        )
        if recommendation_matches:
            if model_recommendation.get("research_only"):
                st.warning(
                    f"${selected_strike:,.0f} lower-risk alternative selected for review · "
                    f"status: {model_recommendation.get('strategy_status', 'not verified')}. "
                    "This alternative did not pass every strategy/risk filter.",
                    title="Review only · not approved for entry",
                )
            else:
                st.success(
                    f"Model-selected pair is active in the dashboard · forecast as of "
                    f"{model_recommendation.get('model_as_of_utc', 'unknown')}. "
                    f"Risk: {model_recommendation.get('risk_level', 'unavailable')}.",
                    title="AI recommendation applied",
                )
            active_metrics = st.columns(4)
            with active_metrics[0]:
                loss_probability = model_recommendation.get("probability_of_loss")
                st.metric("Model scenario loss probability", "Unavailable" if loss_probability is None else f"{loss_probability:.1%}")
            with active_metrics[1]:
                cvar = model_recommendation.get("cvar95_btc_per_unit")
                st.metric("Model CVaR95", "Unavailable" if cvar is None else f"{cvar:.6f} BTC / unit")
            with active_metrics[2]:
                worst_loss = model_recommendation.get("worst_sampled_loss_btc_per_unit")
                st.metric("Worst sampled loss", "Unavailable" if worst_loss is None else f"{worst_loss:.6f} BTC / unit")
            with active_metrics[3]:
                expected_pnl = model_recommendation.get("expected_net_pnl_btc_per_unit")
                st.metric("Expected net P&L", "Unavailable" if expected_pnl is None else f"{expected_pnl:.6f} BTC / unit")
            st.caption(
                f"These estimates came from {model_recommendation.get('scenario_count', 0):,} simulated expiry paths. "
                "They refresh in AI research; this applied pair changes only when you choose a new recommendation. "
                "Worst sampled loss is not a maximum-loss bound."
            )
        else:
            st.info(
                "The current expiry/strike differs from the last AI-applied pair. Open AI research and apply its current recommendation to synchronize it.",
                title="Manual pair selected",
            )
    feed_ids = [call_contract.instrument_id, put_contract.instrument_id]
    websocket_url = TESTNET_WEBSOCKET_URL if is_testnet else PUBLIC_WEBSOCKET_URL
    feed_ids_tuple = tuple(dict.fromkeys(feed_ids))
    feed_signature = (websocket_url, *feed_ids_tuple)
    current_feed: DeribitPublicFeed | None = st.session_state.get("live_feed")
    if current_feed is None or st.session_state["live_feed_signature"] != feed_signature:
        if current_feed is not None:
            current_feed.stop()
        st.session_state["live_feed"] = DeribitPublicFeed(
            feed_ids_tuple,
            websocket_url=websocket_url,
        )
        st.session_state["live_feed_signature"] = feed_signature
    st.session_state["live_selected_call"] = call_contract
    st.session_state["live_selected_put"] = put_contract
    live_feed: DeribitPublicFeed = st.session_state["live_feed"]
    chat_feed_state = live_feed.snapshot()
    chat_feed_quotes = chat_feed_state["quotes"]
    chat_call_quote = stream_quote(call_contract, chat_feed_quotes.get(call_contract.instrument_id))
    chat_put_quote = stream_quote(put_contract, chat_feed_quotes.get(put_contract.instrument_id))
    live_chat_context = {
        "mode": data_source,
        "snapshot_time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "index": chat_feed_state["index_price"] or catalog.reference_price,
        "pair": {
            "strike": selected_strike,
            "expiry": selected_expiry[:10],
            "call": {
                "instrument": call_contract.instrument_id,
                "bid": chat_call_quote.bid,
                "ask": chat_call_quote.ask,
                "status": "fresh" if has_fresh_quote(chat_call_quote, max_age_seconds=5) else "waiting or stale",
                "quote_time_utc": chat_call_quote.timestamp,
            },
            "put_instrument": put_contract.instrument_id,
            "put": {
                "instrument": put_contract.instrument_id,
                "bid": chat_put_quote.bid,
                "ask": chat_put_quote.ask,
                "status": "fresh" if has_fresh_quote(chat_put_quote, max_age_seconds=5) else "waiting or stale",
                "quote_time_utc": chat_put_quote.timestamp,
            },
        },
    }
    feed_connected = bool(chat_feed_state["connected"])
    if chat_feed_state["error"]:
        feed_status, feed_color, feed_icon = "Feed reconnecting", "orange", ":material/sync:"
    elif feed_connected:
        feed_status, feed_color, feed_icon = "Feed connected", "green", ":material/wifi:"
    else:
        feed_status, feed_color, feed_icon = "Feed connecting", "blue", ":material/pending:"
    call_status = "Fresh" if has_fresh_quote(chat_call_quote, max_age_seconds=5) and chat_call_quote.is_active else "Stale / waiting"
    put_status = "Fresh" if has_fresh_quote(chat_put_quote, max_age_seconds=5) and chat_put_quote.is_active else "Stale / waiting"
    with st.container(horizontal=True):
        st.badge(feed_status, icon=feed_icon, color=feed_color)
        st.badge(
            "BTC index unavailable" if chat_feed_state["index_price"] is None
            else f"BTC index ${chat_feed_state['index_price']:,.2f}",
            icon=":material/currency_bitcoin:", color="orange",
        )
        st.badge(f"Call · {call_status}", icon=":material/call:", color="green" if call_status == "Fresh" else "orange")
        st.badge(f"Put · {put_status}", icon=":material/show_chart:", color="green" if put_status == "Fresh" else "orange")
    if chat_feed_state["error"]:
        st.warning(str(chat_feed_state["error"]), title="Live market feed reconnecting")
    quote_times = [
        quote.timestamp for quote in (chat_call_quote, chat_put_quote) if quote.timestamp
    ]
    latest_quote_time = max(quote_times) if quote_times else "waiting for first quote"
    st.caption(f"Latest option quote: {latest_quote_time} · Quotes older than 5 seconds are marked stale.")

    tab_labels = ["Overview", "AI research", "BTC chat"]
    if is_testnet:
        tab_labels.append("TestNet")
    view_tabs = st.tabs(
        tab_labels,
        key=f"btc_live_views_{catalog.environment}",
        on_change="rerun",
    )
    overview_tab, research_tab, chat_tab = view_tabs[:3]
    execution_tab = view_tabs[3] if is_testnet else None
    prediction_data = None
    research_frames: dict[str, pd.DataFrame] = {}
    research_error = None
    chat_selection = None
    if chat_tab.open:
        try:
            chat_snapshot = st.session_state.get("chat_hybrid_research_snapshot")
            chat_snapshot_is_fresh = (
                chat_snapshot is not None
                and time.monotonic() - chat_snapshot["cached_at_monotonic"] < 60
            )
            if chat_snapshot_is_fresh:
                prediction_data = chat_snapshot["prediction_data"]
                research_frames = chat_snapshot["research_frames"]
            else:
                with st.spinner("Loading completed multi-timeframe candles and scoring the hybrid model…"):
                    prediction_data, research_frames = _cached_live_hybrid_snapshot()
                st.session_state["chat_hybrid_research_snapshot"] = {
                    "cached_at_monotonic": time.monotonic(),
                    "prediction_data": prediction_data,
                    "research_frames": research_frames,
                }
            _validate_live_prediction_freshness(prediction_data["as_of_utc"])
            chat_prediction = MarketPrediction(**prediction_data)
            chat_catalog = _cached_live_option_catalog(catalog.environment)
            chat_settings = load_strategy_settings()
            chat_cache_key = (
                catalog.environment, chat_catalog.fetched_at, prediction_data["as_of_utc"],
                repr(chat_settings["candlestick"]), repr(chat_settings["optimizer"]),
            )
            chat_cache = st.session_state.get("live_chat_research_cache")
            if chat_cache is not None and chat_cache["key"] == chat_cache_key:
                chat_patterns = chat_cache["patterns"]
                chat_selection = chat_cache["selection"]
            else:
                chat_patterns = analyze_candlesticks(
                    research_frames,
                    as_of_utc=prediction_data["as_of_utc"],
                    config=chat_settings["candlestick"],
                )
                chat_selection = _rank_live_chain(
                    chat_catalog,
                    chat_prediction,
                    candlestick_risk_score=chat_patterns.risk_score,
                )
                st.session_state["live_chat_research_cache"] = {
                    "key": chat_cache_key,
                    "patterns": chat_patterns,
                    "selection": chat_selection,
                }
            chat_selected = chat_selection.selected
            chat_risk = (candidate_risk_summary(
                chat_selected,
                candlestick_risk_score=chat_patterns.risk_score,
                candlestick_veto_threshold=chat_patterns.veto_threshold,
            ) if chat_selected is not None else None)
            live_chat_context["hybrid_prediction"] = prediction_data
            live_chat_context["candlestick_analysis"] = asdict(chat_patterns)
            live_chat_context["strategy_research"] = {
                "selected_pair": None if chat_selection.selected is None else {
                    "call_instrument": chat_selection.selected.pair.call.instrument_id,
                    "put_instrument": chat_selection.selected.pair.put.instrument_id,
                    "strike": chat_selection.selected.pair.call.strike,
                    "expiry_utc": chat_selection.selected.pair.call.expiry,
                    "score_btc_per_unit": chat_selection.selected.score,
                    "expected_net_pnl_btc_per_unit": chat_selection.selected.expected_net_pnl,
                    "cvar95_btc_per_unit": chat_selection.selected.cvar95,
                    "risk_level": chat_risk["level"],
                    "probability_of_loss": chat_risk["probability_of_loss"],
                    "worst_sampled_loss_btc_per_unit": chat_risk["worst_simulated_loss"],
                    "margin_estimate_btc_per_unit": chat_risk["margin_requirement"],
                    "risk_notes": chat_risk["reasons"],
                },
                "eligible_pair_count": len(chat_selection.eligible_candidates),
                "decision": chat_selection.decision,
                "margin_status": "unknown; account margin is not estimated",
            }
            selected_interval = st.session_state.get("btc_candle_interval", "1 hour")
            chart_resolution = {
                "5 minutes": "5", "15 minutes": "15", "1 hour": "60",
                "2 hours": "120", "6 hours": "360", "1 day": "1D",
            }.get(selected_interval, "60")
            chart_candles = _cached_btc_candles(
                chart_resolution,
                96,
            )
            live_chat_context["candlestick_history"] = {
                "source": "Deribit BTC-PERPETUAL public OHLC candles",
                "interval": selected_interval,
                "last_12_completed_candles": chart_candles[-12:],
            }
        except Exception as exc:
            research_error = str(exc)
            live_chat_context["hybrid_model_error"] = research_error
            live_chat_context["candlestick_history_status"] = "Unavailable; no candle values were substituted."
    if overview_tab.open:
        with overview_tab:
            with st.container(key="overview_page"):
                render_btc_candlestick_chart()
                st.divider()
                st.subheader("Selected option pair", icon=":material/query_stats:")
                render_live_quotes_panel()
                with st.expander(f"Full option chain · {selected_expiry[:10]} UTC"):
                    st.caption(
                        f"Chain snapshot fetched {catalog.fetched_at}. Selected pair quotes above update from the live WebSocket."
                    )
                    st.dataframe(
                        option_chain_rows(catalog, selected_expiry),
                        hide_index=True,
                        alt="Deribit BTC Call and Put option chain for the selected expiry",
                    )
    if research_tab.open:
        with research_tab:
            with st.container(key="research_page"):
                render_auto_refreshing_ai_research(catalog.environment)
    if chat_tab.open:
        with chat_tab:
            with st.container(key="chat_page"):
                render_live_chat(live_chat_context)
    if execution_tab is not None and execution_tab.open:
        with execution_tab:
            with st.container(key="testnet_page"):
                render_testnet_execution_controls()
    st.stop()
