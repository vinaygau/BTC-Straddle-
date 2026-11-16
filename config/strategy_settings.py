"""Load project research settings from JSON-compatible YAML files.

JSON is a strict subset of YAML, which lets these config files stay portable
without adding a runtime YAML parser dependency.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


@lru_cache(maxsize=1)
def load_strategy_settings() -> dict[str, Any]:
    loaded: dict[str, Any] = {}
    for name in ("scenario_config.yaml", "optimizer_config.yaml", "backtest_config.yaml", "candlestick_config.yaml"):
        path = _CONFIG_DIR / name
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Invalid JSON-compatible YAML project config: {path}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"Configuration root must be a mapping: {path}")
        loaded.update(data)
    return loaded
