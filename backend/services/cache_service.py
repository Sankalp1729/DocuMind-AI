from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

import redis

from backend.core.config import (
    ENABLE_REDIS_CACHE,
    REDIS_ANALYTICS_TTL_SECONDS,
    REDIS_CACHE_TTL_SECONDS,
    REDIS_ENABLED,
    REDIS_KEY_PREFIX,
    REDIS_RETRIEVAL_CACHE_TTL_SECONDS,
    REDIS_SESSION_TTL_SECONDS,
    REDIS_STREAM_TTL_SECONDS,
    REDIS_URL,
)


logger = logging.getLogger(__name__)


class CacheService:
    def __init__(self, redis_url: str = REDIS_URL, enabled: bool = REDIS_ENABLED and ENABLE_REDIS_CACHE):
        self.enabled = enabled
        self.redis_url = redis_url
        self._client = None

        if not enabled:
            return

        try:
            self._client = redis.Redis.from_url(redis_url, decode_responses=True)
            self._client.ping()
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("Redis cache unavailable, falling back to in-process storage: %s", exc)
            self._client = None
            self.enabled = False

    def is_available(self) -> bool:
        return self.enabled and self._client is not None

    def _key(self, namespace: str, *parts: str) -> str:
        normalized_parts = [part.replace(" ", "_") for part in parts if part]
        return ":".join([REDIS_KEY_PREFIX, namespace, *normalized_parts])

    @staticmethod
    def _stable_digest(*parts: str) -> str:
        digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
        return digest[:24]

    def get_json(self, key: str) -> Any | None:
        if not self.is_available():
            return None
        payload = self._client.get(key)
        return json.loads(payload) if payload else None

    def set_json(self, key: str, payload: Any, ttl_seconds: int = REDIS_CACHE_TTL_SECONDS) -> None:
        if not self.is_available():
            return
        self._client.set(key, json.dumps(payload, ensure_ascii=False, default=str), ex=ttl_seconds)

    def delete(self, key: str) -> None:
        if self.is_available():
            self._client.delete(key)

    def session_key(self, session_id: str) -> str:
        return self._key("session", session_id)

    def get_session_state(self, session_id: str) -> dict[str, Any] | None:
        payload = self.get_json(self.session_key(session_id))
        return payload if isinstance(payload, dict) else None

    def set_session_state(self, session_id: str, payload: dict[str, Any]) -> None:
        self.set_json(self.session_key(session_id), payload, ttl_seconds=REDIS_SESSION_TTL_SECONDS)

    def retrieval_key(self, question: str, top_k: int) -> str:
        digest = self._stable_digest(question.strip().lower(), str(top_k))
        return self._key("retrieval", digest)

    def get_retrieval_cache(self, question: str, top_k: int) -> dict[str, Any] | None:
        payload = self.get_json(self.retrieval_key(question, top_k))
        return payload if isinstance(payload, dict) else None

    def set_retrieval_cache(self, question: str, top_k: int, payload: dict[str, Any]) -> None:
        self.set_json(self.retrieval_key(question, top_k), payload, ttl_seconds=REDIS_RETRIEVAL_CACHE_TTL_SECONDS)

    def streaming_key(self, session_id: str, question: str) -> str:
        digest = self._stable_digest(session_id, question.strip().lower())
        return self._key("stream", digest)

    def get_streaming_cache(self, session_id: str, question: str) -> dict[str, Any] | None:
        payload = self.get_json(self.streaming_key(session_id, question))
        return payload if isinstance(payload, dict) else None

    def set_streaming_cache(self, session_id: str, question: str, payload: dict[str, Any]) -> None:
        self.set_json(self.streaming_key(session_id, question), payload, ttl_seconds=REDIS_STREAM_TTL_SECONDS)

    def increment_analytics_counter(self, metric_name: str, amount: int = 1) -> None:
        if self.is_available():
            key = self._key("analytics", metric_name)
            self._client.incrby(key, amount)
            self._client.expire(key, REDIS_ANALYTICS_TTL_SECONDS)
