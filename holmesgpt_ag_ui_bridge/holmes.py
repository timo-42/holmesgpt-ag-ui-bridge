from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class HolmesSSEvent:
    event: str | None
    data: dict[str, Any]


class HolmesClient:
    def __init__(self, *, base_url: str, api_key: str | None, timeout_seconds: float) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = httpx.Timeout(timeout_seconds, connect=30.0)

    @property
    def chat_url(self) -> str:
        return f"{self._base_url}/api/chat"

    @property
    def model_url(self) -> str:
        return f"{self._base_url}/api/model"

    @property
    def health_url(self) -> str:
        return f"{self._base_url}/healthz"

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "text/event-stream"}
        if self._api_key:
            headers["X-API-Key"] = self._api_key
        return headers

    async def stream_chat(self, payload: dict[str, Any]) -> AsyncIterator[HolmesSSEvent]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream(
                "POST",
                self.chat_url,
                json=payload,
                headers=self._headers(),
            ) as response:
                response.raise_for_status()
                async for event in parse_sse_lines(response.aiter_lines()):
                    yield event

    async def get_json(self, url: str) -> Any:
        headers = {}
        if self._api_key:
            headers["X-API-Key"] = self._api_key
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            return response.json()


async def parse_sse_lines(lines: AsyncIterator[str]) -> AsyncIterator[HolmesSSEvent]:
    event_type: str | None = None
    data_lines: list[str] = []

    async for raw_line in lines:
        line = raw_line.rstrip("\r")
        if line == "":
            event = _build_event(event_type, data_lines)
            if event is not None:
                yield event
            event_type = None
            data_lines = []
            continue

        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]

        if field == "event":
            event_type = value
        elif field == "data":
            data_lines.append(value)

    event = _build_event(event_type, data_lines)
    if event is not None:
        yield event


def _build_event(event_type: str | None, data_lines: list[str]) -> HolmesSSEvent | None:
    if not data_lines:
        return None

    data_text = "\n".join(data_lines)
    try:
        data = json.loads(data_text)
    except json.JSONDecodeError:
        data = {"content": data_text}
    if not isinstance(data, dict):
        data = {"content": data}
    return HolmesSSEvent(event=event_type, data=data)
