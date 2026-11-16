"""Read-only public Deribit production or TestNet WebSocket feed."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from typing import Any

from websockets.sync.client import connect


PUBLIC_WEBSOCKET_URL = "wss://www.deribit.com/ws/api/v2"
TESTNET_WEBSOCKET_URL = "wss://test.deribit.com/ws/api/v2"
_ALLOWED_WEBSOCKET_URLS = {PUBLIC_WEBSOCKET_URL, TESTNET_WEBSOCKET_URL}


class DeribitPublicFeed:
    """Maintain latest public index and ticker notifications on a daemon thread."""

    def __init__(
        self,
        instrument_ids: tuple[str, ...],
        *,
        websocket_url: str = PUBLIC_WEBSOCKET_URL,
    ) -> None:
        if websocket_url not in _ALLOWED_WEBSOCKET_URLS:
            raise ValueError("Only the configured Deribit production and TestNet feeds are supported.")
        self.instrument_ids = tuple(dict.fromkeys(instrument_ids))
        self.websocket_url = websocket_url
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="deribit-public-feed",
            daemon=True,
        )
        self._connected = False
        self._error: str | None = None
        self._index_price: float | None = None
        self._quotes: dict[str, dict[str, Any]] = {}
        self._last_message_at: str | None = None
        self._thread.start()

    def snapshot(self) -> dict[str, Any]:
        """Return a thread-safe copy of the latest feed state."""
        with self._lock:
            return {
                "connected": self._connected,
                "error": self._error,
                "index_price": self._index_price,
                "quotes": {key: value.copy() for key, value in self._quotes.items()},
                "last_message_at": self._last_message_at,
            }

    def stop(self) -> None:
        """Request a clean stop; recv() is bounded to one second."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        delay = 1.0
        channels = [
            f"ticker.{instrument_id}.100ms" for instrument_id in self.instrument_ids
        ]
        channels.append("deribit_price_index.btc_usd")

        while not self._stop.is_set():
            try:
                with connect(
                    self.websocket_url,
                    open_timeout=10,
                    close_timeout=2,
                    ping_interval=20,
                    ping_timeout=20,
                ) as websocket:
                    websocket.send(
                        json.dumps(
                            {
                                "jsonrpc": "2.0",
                                "id": 1,
                                "method": "public/subscribe",
                                "params": {"channels": channels},
                            }
                        )
                    )
                    with self._lock:
                        self._connected = True
                        self._error = None
                    delay = 1.0

                    while not self._stop.is_set():
                        try:
                            raw_message = websocket.recv(timeout=1)
                        except TimeoutError:
                            continue
                        self._handle_message(raw_message)
            except Exception as exc:
                with self._lock:
                    self._connected = False
                    self._error = f"{type(exc).__name__}: {exc}"
                if self._stop.wait(delay):
                    break
                delay = min(delay * 2, 8.0)

        with self._lock:
            self._connected = False

    def _handle_message(self, raw_message: str | bytes) -> None:
        try:
            payload = json.loads(raw_message)
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
            return

        error = payload.get("error")
        if error:
            message = error.get("message", "Deribit subscription error")
            with self._lock:
                self._error = str(message)
            return

        params = payload.get("params")
        if payload.get("method") != "subscription" or not isinstance(params, dict):
            return
        channel = params.get("channel")
        data = params.get("data")
        if not isinstance(channel, str) or not isinstance(data, dict):
            return

        if channel == "deribit_price_index.btc_usd":
            try:
                price = float(data["price"])
            except (KeyError, TypeError, ValueError):
                return
            with self._lock:
                self._index_price = price
                self._last_message_at = datetime.now(timezone.utc).isoformat()
            return

        if channel.startswith("ticker."):
            instrument_id = str(data.get("instrument_name") or "")
            if instrument_id not in self.instrument_ids:
                return
            quote = {
                "bid": _optional_float(data.get("best_bid_price")),
                "ask": _optional_float(data.get("best_ask_price")),
                "bid_size": _optional_float(data.get("best_bid_amount")),
                "ask_size": _optional_float(data.get("best_ask_amount")),
                "timestamp": _iso_from_milliseconds(data.get("timestamp")),
                "state": str(data.get("state") or "unknown").lower(),
            }
            with self._lock:
                self._quotes[instrument_id] = quote
                self._last_message_at = datetime.now(timezone.utc).isoformat()


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _iso_from_milliseconds(value: Any) -> str | None:
    try:
        timestamp = datetime.fromtimestamp(float(value) / 1000.0, tz=timezone.utc)
    except (TypeError, ValueError, OverflowError, OSError):
        return None
    return timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
