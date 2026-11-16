"""Write inspectable JSON and CSV data-quality artifacts."""
import json
from pathlib import Path
import pandas as pd


def write_quality_reports(reports: list[dict], output_dir: str | Path, stem: str = "dataset_quality") -> tuple[Path, Path]:
    directory = Path(output_dir); directory.mkdir(parents=True, exist_ok=True)
    json_path, csv_path = directory / f"{stem}.json", directory / f"{stem}.csv"
    json_path.write_text(json.dumps(reports, indent=2, default=str), encoding="utf-8")
    rows = []
    for item in reports:
        row = {k: v for k, v in item.items() if k not in ("columns", "dtypes", "missing_values", "validation", "missing_candle_intervals")}
        row.update({f"validation_{k}": v for k, v in item.get("validation", {}).items()})
        row["missing_values_json"] = json.dumps(item.get("missing_values", {}), sort_keys=True)
        row["missing_candle_intervals_count"] = sum(x.get("missing_candles", 0) for x in item.get("missing_candle_intervals", []))
        rows.append(row)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return json_path, csv_path

