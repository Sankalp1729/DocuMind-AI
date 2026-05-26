from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterator

import requests


@dataclass(slots=True)
class StreamingEvent:
    event: str
    payload: dict


class StreamingClientError(RuntimeError):
    pass


class StreamingClient:
    def __init__(self, api_base_url: str, timeout_seconds: int = 180, session_id: str | None = None):
        self.api_base_url = api_base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.session_id = session_id

    def stream_question(self, question: str) -> Iterator[StreamingEvent]:
        url = f"{self.api_base_url}/chat/stream"
        with requests.post(
            url,
            json={"question": question},
            stream=True,
            timeout=self.timeout_seconds,
            headers={"Accept": "text/event-stream", "X-Session-ID": self.session_id} if self.session_id else {"Accept": "text/event-stream"},
        ) as response:
            if response.status_code >= 400:
                try:
                    payload = response.json()
                    detail = payload.get("detail") or payload.get("message") or response.text
                except ValueError:
                    detail = response.text or f"HTTP {response.status_code}"
                raise StreamingClientError(f"{detail} (HTTP {response.status_code})")

            event_name = "message"
            data_lines: list[str] = []

            for raw_line in response.iter_lines(decode_unicode=True):
                if raw_line is None:
                    continue

                line = raw_line.strip()
                if not line:
                    if data_lines:
                        payload = json.loads("\n".join(data_lines))
                        yield StreamingEvent(event=event_name, payload=payload)
                        event_name = "message"
                        data_lines = []
                    continue

                if line.startswith("event:"):
                    event_name = line.removeprefix("event:").strip()
                elif line.startswith("data:"):
                    data_lines.append(line.removeprefix("data:").strip())
