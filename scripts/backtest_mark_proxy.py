"""Run a held-out, explicitly mark-price-proxy BTC short-straddle replay."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config.data_config import DataPipelineConfig
from backtesting.straddle_backtester import run_mark_proxy_comparison


def main() -> None:
    config = DataPipelineConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=config.processed_dir / "model_artifacts" / "heldout_test_predictions.parquet")
    parser.add_argument("--option-trades", type=Path, default=config.processed_dir / "btc_option_trades.parquet")
    parser.add_argument("--btc-candles", type=Path, default=config.processed_dir / "btc_5m.parquet")
    parser.add_argument("--output-dir", type=Path,
                        help="Defaults to a new timestamped proxy-run directory; existing reports are preserved.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing report files in an explicitly supplied output directory.")
    parser.add_argument("--max-mark-age-seconds", type=int, default=300)
    parser.add_argument("--cvar-penalty-lambda", type=float, default=1.0)
    parser.add_argument("--quantity", type=float, default=1.0, help="Normalized research unit; not a live order size")
    parser.add_argument("--fee-rate", type=float, default=0.0)
    parser.add_argument("--slippage-rate", type=float, default=0.0)
    args = parser.parse_args()
    output_dir = args.output_dir or (config.reports_dir / "mark_proxy_backtest_runs" /
                                     datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    if not args.overwrite and ((output_dir / "heldout_mark_proxy_trades.parquet").exists()
                               or (output_dir / "heldout_mark_proxy_report.json").exists()):
        parser.error(f"Output already exists at {output_dir}; choose a new --output-dir or pass --overwrite.")

    for path in (args.predictions, args.option_trades, args.btc_candles):
        if not path.is_file():
            parser.error(f"Required input is missing: {path}")
    predictions = pd.read_parquet(args.predictions)
    options = pd.read_parquet(args.option_trades)
    btc = pd.read_parquet(args.btc_candles)
    trades, summary = run_mark_proxy_comparison(
        predictions,
        options,
        btc,
        max_mark_age_seconds=args.max_mark_age_seconds,
        cvar_lambda=args.cvar_penalty_lambda,
        quantity=args.quantity,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    trades_path = output_dir / "heldout_mark_proxy_trades.parquet"
    report_path = output_dir / "heldout_mark_proxy_report.json"
    trades.to_parquet(trades_path, index=False)
    summary.update({
        "prediction_file": str(args.predictions),
        "option_trade_file": str(args.option_trades),
        "btc_candle_file": str(args.btc_candles),
        "trade_output_file": str(trades_path),
        "trade_output_rows": len(trades),
    })
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
