"""Causal, rule-based candle-pattern context for short-straddle risk filtering.

Patterns are descriptive features, not stand-alone price forecasts. The risk
score is deliberately used as a conservative veto for volatility expansion,
not as proof that a trade will be profitable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimeframeCandleAnalysis:
    timeframe: str
    as_of_utc: str
    candle_count: int
    last_close: float | None
    patterns: tuple[str, ...]
    direction_bias: str
    range_expansion_ratio: float | None
    volume_ratio: float | None
    move_over_lookback: float | None
    risk_score: int


@dataclass(frozen=True)
class CandlestickAnalysis:
    as_of_utc: str
    timeframes: tuple[TimeframeCandleAnalysis, ...]
    risk_score: int
    risk_veto: bool
    veto_threshold: int
    warnings: tuple[str, ...] = ()


def _analyze_timeframe(
    timeframe: str,
    frame: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    lookback: int,
    expansion_ratio_limit: float,
    volume_spike_ratio: float,
    directional_move_threshold: float,
) -> TimeframeCandleAnalysis:
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{timeframe} candles are missing OHLCV columns: {sorted(required-set(frame.columns))}")
    data = frame.copy()
    if "close_time_utc" in data:
        available = pd.to_datetime(data.close_time_utc, utc=True, errors="coerce")
    elif "timestamp_utc" in data:
        interval = {"5m": "5min", "15m": "15min", "1h": "1h", "daily": "1d"}.get(timeframe)
        if interval is None:
            raise ValueError(f"Unsupported candle timeframe: {timeframe}")
        available = pd.to_datetime(data.timestamp_utc, utc=True, errors="coerce") + pd.Timedelta(interval) - pd.Timedelta(milliseconds=1)
    else:
        raise ValueError(f"{timeframe} candles need close_time_utc or timestamp_utc")
    data = data.assign(_available_at=available)
    for column in required:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = (data.dropna(subset=["_available_at", *required])
            .loc[lambda rows: rows._available_at <= as_of]
            .sort_values("_available_at", kind="stable")
            .drop_duplicates("_available_at", keep="last")
            .tail(lookback + 1)
            .reset_index(drop=True))
    if len(data) < 3:
        return TimeframeCandleAnalysis(timeframe, as_of.isoformat().replace("+00:00", "Z"), len(data),
                                       None, ("insufficient_completed_candles",), "unknown", None, None, None, 0)

    latest = data.iloc[-1]
    previous = data.iloc[-2]
    candle_range = max(float(latest.high - latest.low), 1e-12)
    body = abs(float(latest.close - latest.open))
    upper_wick = float(latest.high - max(latest.open, latest.close))
    lower_wick = float(min(latest.open, latest.close) - latest.low)
    patterns: list[str] = []
    direction = "neutral"
    points = 0

    if body / candle_range <= 0.10:
        patterns.append("doji")
    if lower_wick >= 2.0 * max(body, 1e-12) and upper_wick <= max(body, candle_range * .15) and latest.close >= latest.low + .6 * candle_range:
        patterns.append("hammer")
        direction = "bullish"
        points += 1
    if upper_wick >= 2.0 * max(body, 1e-12) and lower_wick <= max(body, candle_range * .15) and latest.close <= latest.low + .4 * candle_range:
        patterns.append("shooting_star")
        direction = "bearish"
        points += 1

    prev_body_low, prev_body_high = sorted((float(previous.open), float(previous.close)))
    curr_body_low, curr_body_high = sorted((float(latest.open), float(latest.close)))
    if previous.close < previous.open and latest.close > latest.open and curr_body_low <= prev_body_low and curr_body_high >= prev_body_high:
        patterns.append("bullish_engulfing")
        direction = "bullish"
        points += 1
    elif previous.close > previous.open and latest.close < latest.open and curr_body_low <= prev_body_low and curr_body_high >= prev_body_high:
        patterns.append("bearish_engulfing")
        direction = "bearish"
        points += 1

    if latest.high <= previous.high and latest.low >= previous.low:
        patterns.append("inside_bar")

    prior = data.iloc[:-1]
    prior_ranges = (prior.high - prior.low).tail(lookback).to_numpy(dtype=float)
    median_range = float(np.median(prior_ranges)) if len(prior_ranges) else 0.0
    range_ratio = candle_range / median_range if median_range > 0 else None
    prior_volumes = prior.volume.tail(lookback).to_numpy(dtype=float)
    median_volume = float(np.median(prior_volumes)) if len(prior_volumes) else 0.0
    volume_ratio = float(latest.volume / median_volume) if median_volume > 0 else None
    if range_ratio is not None and range_ratio >= expansion_ratio_limit:
        patterns.append("range_expansion")
        points += 1
    if volume_ratio is not None and volume_ratio >= volume_spike_ratio:
        patterns.append("volume_spike")
    if (range_ratio is not None and range_ratio >= expansion_ratio_limit
            and volume_ratio is not None and volume_ratio >= volume_spike_ratio):
        patterns.append("range_expansion_with_volume")
        points += 1

    if len(prior) >= lookback:
        prior_high = float(prior.high.tail(lookback).max())
        prior_low = float(prior.low.tail(lookback).min())
        if latest.close > prior_high:
            patterns.append("range_breakout_up")
            direction = "bullish"
            points += 2
        elif latest.close < prior_low:
            patterns.append("range_breakout_down")
            direction = "bearish"
            points += 2

    move = None
    if len(data) >= min(lookback + 1, 7):
        reference = float(data.close.iloc[-min(lookback + 1, 7)])
        move = float(latest.close / reference - 1.0) if reference > 0 else None
        if move is not None and abs(move) >= directional_move_threshold:
            impulse = "bullish_impulse" if move > 0 else "bearish_impulse"
            patterns.append(impulse)
            direction = "bullish" if move > 0 else "bearish"
            points += 1

    if not patterns:
        patterns.append("no_named_pattern")
    return TimeframeCandleAnalysis(
        timeframe=timeframe,
        as_of_utc=as_of.isoformat().replace("+00:00", "Z"),
        candle_count=len(data),
        last_close=float(latest.close),
        patterns=tuple(dict.fromkeys(patterns)),
        direction_bias=direction,
        range_expansion_ratio=None if range_ratio is None else float(range_ratio),
        volume_ratio=volume_ratio,
        move_over_lookback=move,
        risk_score=int(points),
    )


def analyze_candlesticks(
    candles_by_timeframe: dict[str, pd.DataFrame],
    *,
    as_of_utc: str,
    config: dict[str, Any] | None = None,
) -> CandlestickAnalysis:
    """Identify closed-candle patterns and short-volatility risk context."""
    settings = config or {}
    lookback = int(settings.get("lookback", 20))
    expansion_limit = float(settings.get("range_expansion_ratio", 2.0))
    volume_limit = float(settings.get("volume_spike_ratio", 1.5))
    move_limit = float(settings.get("directional_move_threshold", .0075))
    veto_threshold = int(settings.get("veto_risk_score", 2))
    if min(lookback, veto_threshold) <= 0 or not np.isfinite([expansion_limit, volume_limit, move_limit]).all():
        raise ValueError("Candlestick settings are invalid")
    as_of = pd.Timestamp(as_of_utc)
    if as_of.tzinfo is None:
        raise ValueError("as_of_utc must be timezone-aware")
    as_of = as_of.tz_convert("UTC")

    results = []
    warnings: list[str] = []
    for timeframe in ("5m", "15m", "1h", "daily"):
        frame = candles_by_timeframe.get(timeframe)
        if frame is None:
            warnings.append(f"{timeframe} candle history is unavailable")
            continue
        result = _analyze_timeframe(
            timeframe, frame, as_of=as_of, lookback=lookback,
            expansion_ratio_limit=expansion_limit,
            volume_spike_ratio=volume_limit,
            directional_move_threshold=move_limit,
        )
        results.append(result)
        if result.candle_count < lookback + 1:
            warnings.append(f"{timeframe} has too few completed candles for full lookback analysis")

    fast = {item.timeframe: item for item in results if item.timeframe in {"5m", "15m", "1h"}}
    scores = [item.risk_score for item in fast.values()]
    overall_score = max(scores, default=0)
    directional = [item.direction_bias for item in fast.values() if item.direction_bias in {"bullish", "bearish"}]
    if len(directional) >= 2 and directional.count(directional[0]) >= 2:
        overall_score += 1
        warnings.append(f"Directional candle context is aligned across {directional.count(directional[0])} timeframes")
    return CandlestickAnalysis(
        as_of_utc=as_of.isoformat().replace("+00:00", "Z"),
        timeframes=tuple(results),
        risk_score=int(overall_score),
        risk_veto=overall_score >= veto_threshold,
        veto_threshold=veto_threshold,
        warnings=tuple(warnings),
    )
