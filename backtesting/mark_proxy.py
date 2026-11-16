"""Held-out short-straddle replay using Deribit trade-tape mark prices only.

This is a mark-to-mark research proxy. It does not model bid/ask execution,
displayed size, queue position, margin, or guaranteed fills.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
from typing import Any

import numpy as np
import pandas as pd

from models.interfaces import MarketPrediction
from strategies.dynamic_straddle import _forecast_scenarios, _iv_fraction, _option_btc_value, _weighted_cvar95


@dataclass(frozen=True)
class _OptionMark:
    instrument: str
    expiry: datetime
    strike: float
    option_type: str
    mark_price: float
    iv: float
    available_at_ms: int


def _parse_instrument(name: str) -> tuple[datetime, float, str] | None:
    parts = name.split("-")
    if len(parts) != 4 or parts[0] != "BTC" or parts[3] not in {"C", "P"}:
        return None
    try:
        expiry_day = datetime.strptime(parts[1].upper(), "%d%b%y").replace(tzinfo=timezone.utc)
        # Deribit dated BTC options settle at 08:00 UTC on their expiry date.
        expiry = expiry_day.replace(hour=8)
        strike = float(parts[2])
    except (ValueError, OverflowError):
        return None
    if strike <= 0:
        return None
    return expiry, strike, "CALL" if parts[3] == "C" else "PUT"


def _prepare_mark_maps(trades: pd.DataFrame, query_times_ms: np.ndarray, *, max_age_ms: int):
    required = {"instrument_name", "timestamp", "mark_price", "iv"}
    missing = required.difference(trades.columns)
    if missing:
        raise ValueError(f"Option trade tape is missing required fields: {sorted(missing)}")
    frame = trades.loc[:, [name for name in ["instrument_name", "timestamp", "mark_price", "iv", "block_delayed", "trade_seq"] if name in trades]].copy()
    frame = frame.dropna(subset=["instrument_name", "timestamp", "mark_price", "iv"])
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
    frame["available_at_ms"] = frame["timestamp"]
    if "block_delayed" in frame:
        delayed = pd.to_numeric(frame["block_delayed"], errors="coerce")
        frame["available_at_ms"] = np.maximum(frame["available_at_ms"], delayed.fillna(0))
    frame = frame.loc[(frame["mark_price"] > 0) & (frame["iv"] > 0) & frame["available_at_ms"].notna()]
    parsed = {name: _parse_instrument(name) for name in frame["instrument_name"].unique()}
    frame = frame.loc[frame["instrument_name"].map(lambda name: parsed.get(name) is not None)]
    entry_maps: list[dict[str, _OptionMark]] = [dict() for _ in query_times_ms]

    for instrument, group in frame.groupby("instrument_name", sort=False):
        metadata = parsed[instrument]
        group = group.sort_values(["available_at_ms", "timestamp", *( ["trade_seq"] if "trade_seq" in group else [])], kind="stable")
        times = group["available_at_ms"].to_numpy(dtype=np.int64)
        positions = np.searchsorted(times, query_times_ms, side="right") - 1
        valid = positions >= 0
        candidate_positions = np.maximum(positions, 0)
        ages = query_times_ms - times[candidate_positions]
        valid &= (ages >= 0) & (ages <= max_age_ms)
        marks = group[["mark_price", "iv", "available_at_ms"]].to_numpy()
        for query_index in np.flatnonzero(valid):
            row = marks[candidate_positions[query_index]]
            iv = _iv_fraction(float(row[1]))
            if iv is None:
                continue
            entry_maps[int(query_index)][instrument] = _OptionMark(
                instrument=instrument,
                expiry=metadata[0],
                strike=metadata[1],
                option_type=metadata[2],
                mark_price=float(row[0]),
                iv=iv,
                available_at_ms=int(row[2]),
            )
    return entry_maps


def _pairs(marks: dict[str, _OptionMark], *, min_expiry: datetime) -> list[tuple[_OptionMark, _OptionMark]]:
    calls: dict[tuple[datetime, float], _OptionMark] = {}
    puts: dict[tuple[datetime, float], _OptionMark] = {}
    for mark in marks.values():
        if mark.expiry <= min_expiry:
            continue
        key = (mark.expiry, mark.strike)
        (calls if mark.option_type == "CALL" else puts)[key] = mark
    return [(calls[key], puts[key]) for key in calls.keys() & puts.keys()]


def _score_pair(call: _OptionMark, put: _OptionMark, spot: float, prediction: MarketPrediction,
                as_of: datetime, horizon_steps: int, cvar_lambda: float,
                fee_rate: float, slippage_rate: float) -> dict[str, Any]:
    returns, weights = _forecast_scenarios(prediction, horizon_steps)
    future_spot = spot * (1.0 + returns)
    seconds_to_expiry = (call.expiry - as_of).total_seconds()
    years_left = max(0.0, seconds_to_expiry - horizon_steps * 300) / (365.25 * 24 * 3600)
    call_exit = _option_btc_value(future_spot, call.strike, years_left, call.iv, "CALL")
    put_exit = _option_btc_value(future_spot, put.strike, years_left, put.iv, "PUT")
    entry_credit = call.mark_price + put.mark_price
    exit_scenarios = call_exit + put_exit
    pnl_scenarios = entry_credit - exit_scenarios
    expected_buyback = float(np.dot(weights, exit_scenarios))
    gross_expected_pnl = float(np.dot(weights, pnl_scenarios))
    estimated_costs = (entry_credit + expected_buyback) * (fee_rate + slippage_rate)
    expected_net_pnl = gross_expected_pnl - estimated_costs
    cvar = _weighted_cvar95(np.maximum(-pnl_scenarios, 0.0), weights) + estimated_costs
    return {
        "entry_credit_btc_per_unit": entry_credit,
        "expected_buyback_btc_per_unit": expected_buyback,
        "expected_net_pnl_btc_per_unit": expected_net_pnl,
        "cvar95_btc_per_unit": cvar,
        "score": expected_net_pnl - cvar_lambda * cvar,
    }


def _settled_mark_pnl(call: _OptionMark, put: _OptionMark, exit_marks: dict[str, _OptionMark],
                      quantity: float, fee_rate: float, slippage_rate: float) -> tuple[float, float] | None:
    call_exit = exit_marks.get(call.instrument)
    put_exit = exit_marks.get(put.instrument)
    if call_exit is None or put_exit is None:
        return None
    entry_credit = (call.mark_price + put.mark_price) * quantity
    buyback_cost = (call_exit.mark_price + put_exit.mark_price) * quantity
    costs = (entry_credit + buyback_cost) * (fee_rate + slippage_rate)
    return buyback_cost, entry_credit - buyback_cost - costs


def run_mark_price_proxy_backtest(
    predictions: pd.DataFrame,
    option_trades: pd.DataFrame,
    btc_candles: pd.DataFrame,
    *,
    horizon_steps: int = 12,
    max_mark_age_seconds: int = 300,
    cvar_lambda: float = 1.0,
    quantity: float = 1.0,
    fee_rate: float = 0.0,
    slippage_rate: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Replay out-of-sample predictions, selecting pairs from point-in-time mark updates.

    Each prediction row is evaluated as an independent one-hour decision.
    Rows overlap and must not be compounded into a portfolio-equity curve.
    """
    if predictions.empty or option_trades.empty or btc_candles.empty:
        raise ValueError("Predictions, option trades, and BTC candles must all be non-empty.")
    if horizon_steps <= 0 or max_mark_age_seconds <= 0 or quantity <= 0 or cvar_lambda < 0:
        raise ValueError("horizon, mark age, quantity, and CVaR penalty are invalid.")
    required_predictions = {"prediction_time_utc", "hybrid_expected_return", "hybrid_volatility",
                            "hybrid_return_q10", "hybrid_return_q50", "hybrid_return_q90",
                            "extreme_movement_probability", "bearish_probability", "sideways_probability",
                            "bullish_probability"}
    missing = required_predictions.difference(predictions.columns)
    if missing:
        raise ValueError(f"Held-out prediction data is missing fields: {sorted(missing)}")
    if not {"close_time_utc", "close"}.issubset(btc_candles.columns):
        raise ValueError("BTC candle data must include close_time_utc and close.")

    frame = predictions.copy()
    frame["prediction_time_utc"] = pd.to_datetime(frame["prediction_time_utc"], utc=True)
    frame = frame.sort_values("prediction_time_utc", kind="stable").reset_index(drop=True)
    # Pandas may preserve Parquet timestamps at ns resolution or in-memory test
    # timestamps at us resolution. Timestamp.value is consistently nanoseconds.
    query_times_ms = frame["prediction_time_utc"].map(lambda value: value.value // 1_000_000).to_numpy(dtype=np.int64)
    target_times_ms = query_times_ms + horizon_steps * 300_000
    all_query_times = np.concatenate([query_times_ms, target_times_ms])
    option_marks = _prepare_mark_maps(option_trades, all_query_times,
                                      max_age_ms=max_mark_age_seconds * 1000)
    entry_maps = option_marks[:len(frame)]
    exit_maps = option_marks[len(frame):]

    candles = btc_candles.copy()
    candles["close_time_utc"] = pd.to_datetime(candles["close_time_utc"], utc=True)
    candle_times_ms = candles["close_time_utc"].map(lambda value: value.value // 1_000_000).to_numpy(dtype=np.int64)
    spot_by_ms = dict(zip(candle_times_ms, candles["close"].astype(float)))
    records = []
    for row_index, row in frame.iterrows():
        now = row["prediction_time_utc"].to_pydatetime()
        spot = spot_by_ms.get(int(query_times_ms[row_index]))
        if spot is None or not np.isfinite(spot) or spot <= 0:
            continue
        prediction = MarketPrediction(
            as_of_utc=row["prediction_time_utc"].isoformat().replace("+00:00", "Z"),
            expected_return=float(row["hybrid_expected_return"]),
            return_quantiles={0.1: float(row["hybrid_return_q10"]),
                              0.5: float(row["hybrid_return_q50"]),
                              0.9: float(row["hybrid_return_q90"])},
            predicted_volatility=float(row["hybrid_volatility"]),
            bearish_probability=float(row["bearish_probability"]),
            sideways_probability=float(row["sideways_probability"]),
            bullish_probability=float(row["bullish_probability"]),
            extreme_movement_probability=float(row["extreme_movement_probability"]),
        )
        horizon = timedelta(minutes=5 * horizon_steps)
        available_pairs = _pairs(entry_maps[row_index], min_expiry=now + horizon)
        scored = []
        scoring_errors: list[str] = []
        for call, put in available_pairs:
            try:
                score = _score_pair(call, put, float(spot), prediction, now, horizon_steps,
                                    cvar_lambda, fee_rate, slippage_rate)
            except (ValueError, FloatingPointError, OverflowError) as exc:
                scoring_errors.append(str(exc))
                continue
            scored.append((score, call, put))
        if not scored:
            records.append({"prediction_time_utc": row["prediction_time_utc"].isoformat(),
                            "status": "no_fresh_pair", "candidate_count": len(available_pairs),
                            "scoring_errors": scoring_errors})
            continue
        scored.sort(key=lambda item: (-item[0]["score"], item[1].expiry, item[1].strike))
        dynamic_score, dynamic_call, dynamic_put = scored[0]
        atm_score, atm_call, atm_put = min(scored, key=lambda item: (abs(item[1].strike - float(spot)), item[1].expiry))
        record = {
            "prediction_time_utc": row["prediction_time_utc"].isoformat(),
            "target_time_utc": (row["prediction_time_utc"] + horizon).isoformat(),
            "status": "selected",
            "candidate_count": len(scored),
            "spot_usd": float(spot),
            "model_expected_return": prediction.expected_return,
            "model_volatility_per_5m": prediction.predicted_volatility,
            "cvar_penalty_lambda": cvar_lambda,
            "quantity_proxy": quantity,
            "entry_basis": "last Deribit mark price published at or before the decision time",
            "exit_basis": "last Deribit mark price published at or before the one-hour horizon",
        }
        for prefix, candidate_score, call, put in (
            ("dynamic", dynamic_score, dynamic_call, dynamic_put),
            ("atm", atm_score, atm_call, atm_put),
        ):
            pnl = _settled_mark_pnl(call, put, exit_maps[row_index], quantity,
                                    fee_rate, slippage_rate)
            record.update({
                f"{prefix}_strike": call.strike,
                f"{prefix}_expiry_utc": call.expiry.isoformat().replace("+00:00", "Z"),
                f"{prefix}_call_instrument": call.instrument,
                f"{prefix}_put_instrument": put.instrument,
                f"{prefix}_scenario_score_btc_per_unit": candidate_score["score"],
                f"{prefix}_scenario_expected_pnl_btc_per_unit": candidate_score["expected_net_pnl_btc_per_unit"],
                f"{prefix}_scenario_cvar95_btc_per_unit": candidate_score["cvar95_btc_per_unit"],
                f"{prefix}_entry_mark_credit_btc": (call.mark_price + put.mark_price) * quantity,
                f"{prefix}_exit_mark_cost_btc": pnl[0] if pnl else None,
                f"{prefix}_realized_mark_pnl_btc": pnl[1] if pnl else None,
                f"{prefix}_outcome_status": "mark_exit_available" if pnl else "no_fresh_exit_mark",
            })
        records.append(record)

    output = pd.DataFrame(records)
    summary: dict[str, Any] = {
        "method": "held-out dynamic short-straddle mark-price proxy",
        "model_prediction_rows": len(frame),
        "decision_rows": len(output),
        "selected_rows": int((output.get("status", pd.Series(dtype=str)) == "selected").sum()),
        "cvar_penalty_lambda": cvar_lambda,
        "horizon_minutes": 5 * horizon_steps,
        "max_mark_age_seconds": max_mark_age_seconds,
        "quantity_proxy": quantity,
        "fee_rate": fee_rate,
        "slippage_rate": slippage_rate,
        "entry_exit_basis": "Deribit mark_price from public executed-trade records; not executable bid/ask quotes.",
        "limitations": [
            "The mark-price tape has no historical bid/ask, displayed size, queue position, or fill evidence.",
            "Black–Scholes scenario scores hold each leg's current IV constant through the forecast horizon.",
            "Margin is unavailable and is not inferred.",
            "Each decision is an independent overlapping one-hour observation; P&L rows are not compounded.",
            "The normalized quantity is a research unit, not a venue-minimum or liquidity guarantee.",
        ],
    }
    for prefix in ("dynamic", "atm"):
        pnl_column = f"{prefix}_realized_mark_pnl_btc"
        if pnl_column in output:
            realized = pd.to_numeric(output[pnl_column], errors="coerce").dropna()
            summary[f"{prefix}_outcomes"] = int(len(realized))
            summary[f"{prefix}_outcome_coverage"] = float(len(realized) / max(1, summary["selected_rows"]))
            summary[f"{prefix}_mean_mark_pnl_btc"] = float(realized.mean()) if len(realized) else None
            summary[f"{prefix}_median_mark_pnl_btc"] = float(realized.median()) if len(realized) else None
            summary[f"{prefix}_win_rate"] = float((realized > 0).mean()) if len(realized) else None
            losses = np.maximum(-realized.to_numpy(dtype=np.float64), 0.0)
            tail_count = max(1, int(math.ceil(len(losses) * 0.05))) if len(losses) else 0
            summary[f"{prefix}_realized_cvar95_btc"] = float(np.sort(losses)[-tail_count:].mean()) if tail_count else None
    return output, summary
