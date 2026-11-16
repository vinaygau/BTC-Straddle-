"""Central settings for the historical candle-data pipeline."""
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_TIMEFRAMES = ("5m", "15m", "1h", "daily")
TIMEFRAME_MINUTES = {"5m": 5, "15m": 15, "1h": 60, "daily": 1440}


@dataclass(frozen=True)
class DataPipelineConfig:
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    reports_dir: Path = PROJECT_ROOT / "data" / "reports"
    timezone: str = "UTC"
    output_format: str = "parquet"
    duplicate_policy: str = "source-priority, then first timestamp occurrence"
    live_5m_max_age_seconds: int = 600

