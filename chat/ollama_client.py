"""Small local-only client for the Ollama chat and model-list APIs."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Sequence
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_TIMEOUT_SECONDS = 180


class OllamaChatError(RuntimeError):
    """Readable error returned when the local model service is unavailable."""


def list_local_models() -> list[str]:
    """Return installed local model names, or raise OllamaChatError."""
    request = Request(f"{OLLAMA_BASE_URL}/api/tags", headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise OllamaChatError(f"Ollama returned HTTP {exc.code} while listing models.") from exc
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise OllamaChatError(
            "Could not reach Ollama at 127.0.0.1:11434. Start the Ollama app, then refresh models."
        ) from exc

    models = payload.get("models", [])
    return sorted(
        {
            str(item.get("name"))
            for item in models
            if isinstance(item, dict) and item.get("name")
        }
    )


def stream_chat_response(
    model: str,
    history: Sequence[dict[str, str]],
    market_context: dict[str, Any],
    *,
    timeout_seconds: float = OLLAMA_TIMEOUT_SECONDS,
    num_predict: int | None = None,
) -> Iterator[str]:
    """Stream a grounded response from the selected local Ollama model."""
    if not model.strip():
        raise OllamaChatError("Choose an installed Ollama model to start the chat.")

    system_message = {
        "role": "system",
        "content": (
            "You are a conversational BTC and options dashboard assistant. Use the current "
            "dashboard snapshot below as the only source for live prices, quotes, expiry, "
            "and displayed AI research. State when a value is missing or stale; never invent "
            "current market data. When a quote status is fresh, say it was fresh at its "
            "timestamp; do not claim it is accurate or guaranteed. Explain concepts and "
            "calculations clearly, but do not tell the user to buy, sell, or open a trade. "
            "The dashboard does not execute orders in Live Deribit mode. The snapshot may be out of "
            "date by the time you answer.\n\n"
            "Current dashboard snapshot (JSON):\n"
            f"{json.dumps(market_context, ensure_ascii=False, separators=(',', ':'))}"
        ),
    }
    messages = [system_message]
    messages.extend(
        {"role": item["role"], "content": item["content"]}
        for item in history[-20:]
        if item.get("role") in {"user", "assistant"} and item.get("content")
    )
    options: dict[str, Any] = {"temperature": 0.2}
    if num_predict is not None:
        options["num_predict"] = int(num_predict)
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": options,
        }
    ).encode("utf-8")
    request = Request(
        f"{OLLAMA_BASE_URL}/api/chat",
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/x-ndjson"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            for raw_line in response:
                if not raw_line.strip():
                    continue
                try:
                    payload = json.loads(raw_line.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    raise OllamaChatError("Ollama returned an invalid streaming response.") from exc
                if payload.get("error"):
                    raise OllamaChatError(str(payload["error"]))
                message = payload.get("message") or {}
                content = message.get("content")
                if content:
                    yield str(content)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise OllamaChatError(f"Ollama chat request failed (HTTP {exc.code}): {detail}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise OllamaChatError(
            "The local Ollama chat request failed. Check that Ollama is running and the model is ready."
        ) from exc
