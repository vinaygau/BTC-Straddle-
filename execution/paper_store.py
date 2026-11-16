"""Local SQLite persistence for paper positions and their append-only event history."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from execution.paper_trader import PaperPosition, position_from_dict, position_to_dict


class PaperTradeStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """CREATE TABLE IF NOT EXISTS paper_positions (
                    position_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL
                )"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS paper_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp_utc TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    position_id TEXT,
                    payload_json TEXT NOT NULL
                )"""
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        return connection

    def save_position(self, position: PaperPosition) -> None:
        payload = position_to_dict(position)
        position_id = position.position_id or f"legacy:{position.call_instrument_id}:{position.opened_at}"
        payload["position_id"] = position_id
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO paper_positions(position_id,payload_json,updated_at_utc)
                   VALUES(?,?,?) ON CONFLICT(position_id) DO UPDATE SET
                   payload_json=excluded.payload_json, updated_at_utc=excluded.updated_at_utc""",
                (position_id, json.dumps(payload, allow_nan=False), now),
            )

    def load_active_position(self) -> PaperPosition | None:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM paper_positions ORDER BY updated_at_utc DESC"
            ).fetchall()
        for row in rows:
            payload = json.loads(row["payload_json"])
            if float(payload.get("open_amount", 0.0)) > 1e-12 and payload.get("status") not in {"CLOSED", "REJECTED"}:
                return position_from_dict(payload)
        return None

    def append_event(self, event_type: str, position_id: str | None, payload: dict[str, Any]) -> None:
        event_time = payload.get("timestamp_utc") or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        body = dict(payload)
        body.setdefault("timestamp_utc", event_time)
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO paper_events(timestamp_utc,event_type,position_id,payload_json) VALUES(?,?,?,?)",
                (event_time, event_type, position_id, json.dumps(body, default=str, allow_nan=False)),
            )

    def list_events(self, limit: int = 500) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT timestamp_utc,event_type,position_id,payload_json FROM paper_events ORDER BY event_id DESC LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()
        result = []
        for row in reversed(rows):
            payload = json.loads(row["payload_json"])
            result.append({"Time (UTC)": row["timestamp_utc"], "Event": row["event_type"],
                           "Position ID": row["position_id"], **payload})
        return result
