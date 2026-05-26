from __future__ import annotations

import threading
from collections import Counter, defaultdict
from datetime import datetime, timezone
from statistics import mean

from backend.core.config import ENABLE_ANALYTICS_PERSISTENCE
from backend.persistence.models import AnalyticsEventRecord
from backend.services.cache_service import CacheService
from backend.services.database_service import DatabaseService


class MetricsService:
    def __init__(self, database_service: DatabaseService | None = None, cache_service: CacheService | None = None):
        self._lock = threading.RLock()
        self._counters = Counter()
        self._latencies_ms: dict[str, list[float]] = defaultdict(list)
        self.database_service = database_service
        self.cache_service = cache_service

    def _persist_event(self, event_name: str, metric_name: str | None, value: float, dimensions: dict | None = None) -> None:
        if not (self.database_service and ENABLE_ANALYTICS_PERSISTENCE):
            return

        with self.database_service.session_scope() as session:
            session.add(
                AnalyticsEventRecord(
                    event_name=event_name,
                    metric_name=metric_name,
                    value=value,
                    dimensions_json=dimensions or {},
                    created_at=datetime.now(timezone.utc),
                )
            )

    def increment(self, metric_name: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[metric_name] += amount
        if self.cache_service is not None:
            self.cache_service.increment_analytics_counter(metric_name, amount)
        self._persist_event("counter", metric_name, float(amount))

    def observe_latency(self, metric_name: str, value_ms: float) -> None:
        with self._lock:
            self._latencies_ms[metric_name].append(value_ms)
        self._persist_event("latency", metric_name, value_ms)

    def snapshot(self) -> dict:
        with self._lock:
            latency_summary = {}
            for metric_name, values in self._latencies_ms.items():
                latency_summary[metric_name] = {
                    "count": len(values),
                    "avg_ms": round(mean(values), 2) if values else 0,
                    "max_ms": round(max(values), 2) if values else 0,
                }

            return {
                "counters": dict(self._counters),
                "latencies": latency_summary,
            }