from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from threading import RLock
from typing import Any
from uuid import uuid4

from backend.persistence.models import ConversationRecord, MessageRecord, TokenUsageRecord
from backend.services.cache_service import CacheService
from backend.services.database_service import DatabaseService
from backend.services.workspace_service import WorkspaceService
from backend.utils.token_usage import estimate_turn_tokens
from backend.core.metrics import TOKEN_USAGE


@dataclass(slots=True)
class ConversationMessage:
    role: str
    content: str
    sources: list[dict[str, Any]] | None = None
    latency_ms: int | None = None


class ConversationService:
    def __init__(
        self,
        database_service: DatabaseService,
        cache_service: CacheService | None = None,
        workspace_service: WorkspaceService | None = None,
    ):
        self.database_service = database_service
        self.cache_service = cache_service
        self.workspace_service = workspace_service
        self._lock = RLock()
        self._ensure_workspace_service()

    def _ensure_workspace_service(self) -> None:
        if self.workspace_service is None:
            return

    @staticmethod
    def _serialize_message(message: MessageRecord) -> dict[str, Any]:
        return {
            "role": message.role,
            "content": message.content,
            "sources": message.sources_json or [],
            "latency_ms": message.latency_ms,
            "created_at": message.created_at.isoformat() if message.created_at else None,
        }

    def create_conversation(
        self,
        conversation_id: str | None = None,
        user_id: str | None = None,
        workspace_id: str | None = None,
    ) -> str:
        conversation_id = conversation_id or str(uuid4())

        now = datetime.now(timezone.utc)
        with self._lock, self.database_service.session_scope() as session:
            conversation = session.get(ConversationRecord, conversation_id)
            if conversation is None:
                conversation = ConversationRecord(
                    id=conversation_id,
                    user_id=user_id,
                    workspace_id=workspace_id,
                    created_at=now,
                    updated_at=now,
                )
                session.add(conversation)
            else:
                conversation.user_id = user_id or conversation.user_id
                conversation.workspace_id = workspace_id or conversation.workspace_id
                conversation.updated_at = now
            session.flush()

        return conversation_id

    def append_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        sources: list[dict[str, Any]] | None = None,
        latency_ms: int | None = None,
        workspace_id: str | None = None,
        token_usage: dict[str, int] | None = None,
        model_name: str | None = None,
    ) -> None:
        now = datetime.now(timezone.utc)
        self.create_conversation(conversation_id, workspace_id=workspace_id)

        with self._lock, self.database_service.session_scope() as session:
            session.add(
                MessageRecord(
                    conversation_id=conversation_id,
                    role=role,
                    content=content,
                    sources_json=sources or [],
                    latency_ms=latency_ms,
                    created_at=now,
                )
            )
            conversation = session.get(ConversationRecord, conversation_id)
            if conversation is not None:
                conversation.updated_at = now
                if workspace_id:
                    conversation.workspace_id = workspace_id
            session.flush()

        if self.cache_service is not None:
            history = self.get_history(conversation_id)
            self.cache_service.set_session_state(
                conversation_id,
                {
                    "conversation_id": conversation_id,
                    "workspace_id": workspace_id,
                    "messages": history[-20:],
                    "updated_at": now.isoformat(),
                },
            )

        if token_usage:
            self.record_token_usage(
                conversation_id=conversation_id,
                session_id=conversation_id,
                prompt_tokens=token_usage.get("prompt_tokens", 0),
                completion_tokens=token_usage.get("completion_tokens", 0),
                total_tokens=token_usage.get("total_tokens", 0),
                model_name=model_name,
                metadata={"role": role},
            )

    def record_token_usage(
        self,
        conversation_id: str | None,
        session_id: str | None,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        model_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self.database_service.session_scope() as session:
            session.add(
                TokenUsageRecord(
                    conversation_id=conversation_id,
                    session_id=session_id,
                    model_name=model_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    metadata_json=metadata or {},
                    created_at=datetime.now(timezone.utc),
                )
            )
        # Prometheus counters for tokens
        try:
            model = model_name or "unknown"
            TOKEN_USAGE.labels(model=model, type="prompt").inc(prompt_tokens or 0)
            TOKEN_USAGE.labels(model=model, type="completion").inc(completion_tokens or 0)
        except Exception:
            pass

    def get_history(self, conversation_id: str) -> list[dict[str, Any]]:
        if self.cache_service is not None:
            cached = self.cache_service.get_session_state(conversation_id)
            if cached and isinstance(cached.get("messages"), list):
                return cached["messages"]

        with self._lock, self.database_service.session_scope() as session:
            rows = (
                session.query(MessageRecord)
                .filter(MessageRecord.conversation_id == conversation_id)
                .order_by(MessageRecord.id.asc())
                .all()
            )

        history = []
        for row in rows:
            history.append(self._serialize_message(row))

        if self.cache_service is not None:
            self.cache_service.set_session_state(conversation_id, {"conversation_id": conversation_id, "messages": history})

        return history

    def list_conversations(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock, self.database_service.session_scope() as session:
            rows = (
                session.query(ConversationRecord)
                .order_by(ConversationRecord.updated_at.desc())
                .limit(limit)
                .all()
            )

        return [
            {
                "id": row.id,
                "user_id": row.user_id,
                "workspace_id": row.workspace_id,
                "created_at": row.created_at.isoformat(),
                "updated_at": row.updated_at.isoformat(),
            }
            for row in rows
        ]

    def estimate_and_record_usage(
        self,
        conversation_id: str,
        question: str,
        answer: str,
        context: str | None = None,
        model_name: str | None = None,
    ) -> dict[str, int]:
        usage = estimate_turn_tokens(question, answer, context=context)
        self.record_token_usage(
            conversation_id=conversation_id,
            session_id=conversation_id,
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"],
            model_name=model_name,
            metadata={"question_length": len(question), "answer_length": len(answer)},
        )
        return usage