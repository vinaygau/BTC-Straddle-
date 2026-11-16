"""Package the small inference bundle and bounded BTC history for Streamlit deploys."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from config.data_config import DataPipelineConfig
from models.market_regime_detector import REGIME_FEATURES


MODEL_FILES = (
    "feature_scaler.npz",
    "lstm_multitask.pt",
    "training_report.json",
    "xgboost_return.json",
    "xgboost_volatility.json",
    "xgboost_direction.json",
    "xgboost_extreme.json",
)


def package_assets(
    *,
    model_dir: Path,
    feature_path: Path,
    candle_path: Path,
    output_dir: Path,
    history_days: int = 365,
) -> dict[str, object]:
    """Copy inference-only model files and a recent scenario-history slice."""
    if history_days < 30:
        raise ValueError("At least 30 days of history are required for the deployment bundle.")
    missing = [name for name in MODEL_FILES if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Model bundle is incomplete; missing: {', '.join(missing)}")

    report = json.loads((model_dir / "training_report.json").read_text(encoding="utf-8"))
    if report.get("model_mode") != "hybrid":
        raise ValueError("The deployment bundle requires promoted hybrid model artifacts.")

    feature_columns = ["prediction_time_utc", "5m_close", *REGIME_FEATURES]
    feature_frame = pd.read_parquet(feature_path, columns=list(dict.fromkeys(feature_columns)))
    candle_frame = pd.read_parquet(candle_path, columns=["close_time_utc", "close"])
    feature_frame["prediction_time_utc"] = pd.to_datetime(
        feature_frame["prediction_time_utc"], utc=True, errors="coerce"
    )
    candle_frame["close_time_utc"] = pd.to_datetime(
        candle_frame["close_time_utc"], utc=True, errors="coerce"
    )
    feature_frame = feature_frame.dropna(subset=["prediction_time_utc"])
    candle_frame = candle_frame.dropna(subset=["close_time_utc"])
    latest = min(
        feature_frame["prediction_time_utc"].max(),
        candle_frame["close_time_utc"].max(),
    )
    cutoff = latest - pd.Timedelta(days=history_days)
    feature_frame = feature_frame.loc[feature_frame["prediction_time_utc"] >= cutoff].copy()
    candle_frame = candle_frame.loc[candle_frame["close_time_utc"] >= cutoff].copy()
    if len(feature_frame) < 5000 or len(candle_frame) < 5000:
        raise ValueError("The selected history window contains too few aligned BTC observations.")

    model_output = output_dir / "model_artifacts"
    history_output = output_dir / "research_history"
    model_output.mkdir(parents=True, exist_ok=True)
    history_output.mkdir(parents=True, exist_ok=True)
    for name in MODEL_FILES:
        shutil.copy2(model_dir / name, model_output / name)
    feature_frame.to_parquet(
        history_output / "btc_multitimeframe_features.parquet",
        index=False,
        compression="zstd",
    )
    candle_frame.to_parquet(
        history_output / "btc_5m.parquet",
        index=False,
        compression="zstd",
    )
    return {
        "model_files": len(MODEL_FILES),
        "history_start_utc": cutoff.isoformat(),
        "history_end_utc": latest.isoformat(),
        "feature_rows": len(feature_frame),
        "candle_rows": len(candle_frame),
        "output_dir": str(output_dir),
    }


def main() -> None:
    config = DataPipelineConfig()
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=config.processed_dir / "model_artifacts")
    parser.add_argument(
        "--features", type=Path,
        default=config.processed_dir / "btc_multitimeframe_features.parquet",
    )
    parser.add_argument("--candles", type=Path, default=config.processed_dir / "btc_5m.parquet")
    parser.add_argument("--output-dir", type=Path, default=project_root / "deployment_assets")
    parser.add_argument("--history-days", type=int, default=365)
    args = parser.parse_args()
    result = package_assets(
        model_dir=args.model_dir,
        feature_path=args.features,
        candle_path=args.candles,
        output_dir=args.output_dir,
        history_days=args.history_days,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
