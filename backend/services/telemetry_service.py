from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Deque, Dict, List

from backend.core.config import DATA_DIR
from backend.persistence.models import RetrievalTraceRecord
from backend.services.database_service import DatabaseService
from backend.schemas.telemetry import RetrievalTrace, TelemetryEvent, TelemetrySnapshot


@dataclass(slots=True)
class TimingContext:
    event_type: str
    start_ms: float = 0.0
    end_ms: float = 0.0
    metadata: Dict[str, Any] | None = None

    def duration(self) -> float:
        return self.end_ms - self.start_ms


class TelemetryService:
    """Collects retrieval timing, traces, and persisted telemetry snapshots."""

    def __init__(self, storage_dir: Path | None = None, max_traces: int = 1000, database_service: DatabaseService | None = None):
        self._lock = RLock()
        self._timings: Dict[str, List[TimingContext]] = defaultdict(list)
        self._events: List[TelemetryEvent] = []
        self._traces: Deque[RetrievalTrace] = deque(maxlen=max_traces)
        self.storage_dir = Path(storage_dir or DATA_DIR / "telemetry")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._trace_log_path = self.storage_dir / "retrieval_traces.jsonl"
        self.database_service = database_service

    def start_timer(self, event_type: str, metadata: Dict[str, Any] | None = None) -> TimingContext:
        return TimingContext(event_type=event_type, start_ms=time.perf_counter() * 1000, metadata=metadata or {})

    def end_timer(self, ctx: TimingContext) -> float:
        ctx.end_ms = time.perf_counter() * 1000
        with self._lock:
            self._timings[ctx.event_type].append(ctx)
            self._events.append(
                TelemetryEvent(
                    event_type=ctx.event_type,
                    duration_ms=ctx.duration(),
                    metadata=ctx.metadata or {},
                )
            )
        return ctx.duration()

    def record_event(self, event_type: str, duration_ms: float, metadata: Dict[str, Any] | None = None) -> None:
        with self._lock:
            self._events.append(
                TelemetryEvent(
                    event_type=event_type,
                    duration_ms=duration_ms,
                    metadata=metadata or {},
                )
            )

    def record_retrieval_trace(self, trace: RetrievalTrace) -> None:
        with self._lock:
            self._traces.append(trace)
            payload = trace.model_dump(mode="json")
            with self._trace_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            if self.database_service is not None:
                with self.database_service.session_scope() as session:
                    session.add(
                        RetrievalTraceRecord(
                            trace_id=trace.trace_id,
                            query=trace.query,
                            payload_json=payload,
                            created_at=datetime.now(timezone.utc),
                        )
                    )

    def list_traces(self, limit: int = 20, query: str | None = None) -> List[RetrievalTrace]:
        with self._lock:
            traces = list(self._traces)

        if query:
            query_lower = query.lower()
            traces = [trace for trace in traces if query_lower in trace.query.lower() or (trace.expanded_query and query_lower in trace.expanded_query.lower())]

        return traces[-limit:]

    def get_summary(self, limit: int = 10) -> TelemetrySnapshot:
        with self._lock:
            event_averages: Dict[str, float] = {}
            for event_type, contexts in self._timings.items():
                if contexts:
                    event_averages[event_type] = sum(context.duration() for context in contexts) / len(contexts)

            return TelemetrySnapshot(
                total_traces=len(self._traces),
                event_averages_ms=event_averages,
                recent_traces=list(self._traces)[-limit:],
            )

    def reset(self) -> None:
        with self._lock:
            self._timings.clear()
            self._events.clear()
            self._traces.clear()

