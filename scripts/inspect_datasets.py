"""Inspect all supported candle exports beneath the configured raw directory."""
import argparse
from pathlib import Path
from config.data_config import DataPipelineConfig
from data.data_preprocessor import inspect_directory
from data.data_quality_report import write_quality_reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DataPipelineConfig().raw_dir)
    parser.add_argument("--reports-dir", type=Path, default=DataPipelineConfig().reports_dir)
    args = parser.parse_args()
    reports, _ = inspect_directory(args.raw_dir)
    paths = write_quality_reports(reports, args.reports_dir, "dataset_inspection")
    for item in reports:
        print(f"{item['file_name']}: {item.get('records', 0)} rows; tf={item.get('timeframe')}; included={item.get('included_in_btcusdt_pipeline')}")
    print(f"Reports: {paths[0]} | {paths[1]}")


if __name__ == "__main__": main()
