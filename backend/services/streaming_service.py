from __future__ import annotations

import os
import asyncio
import json
import logging
import time

from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from backend.core.config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    STREAMING_BUFFER_SIZE,
    STREAMING_TIMEOUT_SECONDS,
    TOP_K,
)
from backend.security.context import current_tenant_id, current_workspace_id
from backend.schemas.streaming import StreamingEvent
from backend.services.cache_service import CacheService
from backend.services.conversation_service import ConversationService
from backend.services.metrics_service import MetricsService
from backend.services.rag_service import RagService
from backend.utils.token_usage import estimate_turn_tokens
from backend.core.metrics import RETRIEVAL_REQUESTS, RETRIEVAL_LATENCY
from opentelemetry import trace


logger = logging.getLogger(__name__)


prompt_template = """
You are an intelligent AI assistant.

Answer the question ONLY using the provided context.

If the answer is not present in the context, say:
"I could not find the answer in the uploaded documents."

Context:
{context}

Question:
{question}

Answer:
"""


prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"],
)


class StreamingService:
    def __init__(
        self,
        rag_service: RagService,
        conversation_service: ConversationService | None = None,
        metrics_service: MetricsService | None = None,
        cache_service: CacheService | None = None,
    ):
        self.rag_service = rag_service
        self.conversation_service = conversation_service
        self.metrics_service = metrics_service
        self.cache_service = cache_service
        self.llm = ChatOllama(
            model=os.getenv("DOCUMIND_OLLAMA_MODEL", OLLAMA_MODEL),
            temperature=0,
            streaming=True,
            base_url=os.getenv("DOCUMIND_OLLAMA_BASE_URL", OLLAMA_BASE_URL),
        )

    def _sse(self, event: str, payload: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    def _build_prompt_inputs(self, question: str):
        result = self.rag_service.retrieve(question)
        if result is None:
            return None, None, None, None

        context, results, retrieval_explanation = result
        sources = [
            {
                "source": doc.metadata.get("source_file"),
                "page": doc.metadata.get("page"),
                "preview": doc.page_content[:500],
            }
            for doc in results
        ]

        return context, sources, results, retrieval_explanation

    async def stream_answer(self, question: str, session_id: str = "anonymous", workspace_id: str | None = None):
        workspace_token = None
        tenant_token = None
        if workspace_id:
            workspace_token = current_workspace_id.set(workspace_id)
            tenant_token = current_tenant_id.set(workspace_id)

        cached_stream = self.cache_service.get_streaming_cache(session_id, question) if self.cache_service is not None else None
        tracer = trace.get_tracer(__name__)
        try:
            RETRIEVAL_REQUESTS.labels(origin="stream").inc()
        except Exception:
            pass
        try:
            if cached_stream:
                yield self._sse("start", StreamingEvent(event="start", question=question).model_dump())
                answer = cached_stream.get("answer", "")
                yield self._sse("token", StreamingEvent(event="token", token=answer).model_dump())
                yield self._sse(
                    "complete",
                    StreamingEvent(
                        event="complete",
                        answer=answer,
                        sources=cached_stream.get("sources"),
                        retrieval_explanation=cached_stream.get("retrieval_explanation"),
                        groundedness=cached_stream.get("groundedness"),
                        latency_ms=cached_stream.get("latency_ms"),
                    ).model_dump(),
                )
                if self.conversation_service is not None:
                    self.conversation_service.create_conversation(session_id, workspace_id=workspace_id)
                    self.conversation_service.append_message(session_id, "user", question, workspace_id=workspace_id)
                    self.conversation_service.append_message(session_id, "assistant", answer, sources=cached_stream.get("sources"), latency_ms=cached_stream.get("latency_ms"), workspace_id=workspace_id)
                    token_usage = estimate_turn_tokens(question, answer, context=None)
                    self.conversation_service.record_token_usage(
                        conversation_id=session_id,
                        session_id=session_id,
                        prompt_tokens=token_usage["prompt_tokens"],
                        completion_tokens=token_usage["completion_tokens"],
                        total_tokens=token_usage["total_tokens"],
                        model_name=os.getenv("DOCUMIND_OLLAMA_MODEL", OLLAMA_MODEL),
                        metadata={"cache_hit": True},
                    )
                return

            context, sources, results, retrieval_explanation = self._build_prompt_inputs(question)

            if context is None:
                yield self._sse(
                    "error",
                    StreamingEvent(event="error", error="No vector store is available. Upload a document first or restore vector_store/.").model_dump(),
                )
                return

            yield self._sse(
                "start",
                StreamingEvent(event="start", question=question).model_dump(),
            )

            # Start a tracing span for the streaming session
            try:
                span = tracer.start_span("streaming.session", attributes={"session_id": session_id})
                span.end()
            except Exception:
                pass

            if self.conversation_service is not None:
                self.conversation_service.create_conversation(session_id, workspace_id=workspace_id)
                self.conversation_service.append_message(session_id, "user", question, workspace_id=workspace_id)

            chain = prompt | self.llm
            buffer: list[str] = []
            answer_parts: list[str] = []
            start_time = time.perf_counter()
            queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=STREAMING_BUFFER_SIZE)
            stop_event = asyncio.Event()

            def producer() -> None:
                try:
                    for chunk in chain.stream({"context": context, "question": question}):
                        if stop_event.is_set():
                            break
                        token = getattr(chunk, "content", None) or str(chunk)
                        if token:
                            asyncio.run_coroutine_threadsafe(queue.put(token), loop)
                    asyncio.run_coroutine_threadsafe(queue.put(None), loop)
                except Exception as exc:
                    logger.exception("Streaming generation failed")
                    asyncio.run_coroutine_threadsafe(queue.put(f"__ERROR__:{exc}"), loop)
                    asyncio.run_coroutine_threadsafe(queue.put(None), loop)

            loop = asyncio.get_running_loop()
            producer_task = asyncio.create_task(asyncio.to_thread(producer))

            try:
                while True:
                    try:
                        token = await asyncio.wait_for(queue.get(), timeout=STREAMING_TIMEOUT_SECONDS)
                    except asyncio.TimeoutError:
                        stop_event.set()
                        yield self._sse(
                            "error",
                            StreamingEvent(event="error", error="Streaming timed out before completion.").model_dump(),
                        )
                        return

                    if token is None:
                        break

                    if token.startswith("__ERROR__:"):
                        stop_event.set()
                        yield self._sse(
                            "error",
                            StreamingEvent(event="error", error=token.removeprefix("__ERROR__:")).model_dump(),
                        )
                        return

                    buffer.append(token)
                    answer_parts.append(token)

                    if len(buffer) >= 2:
                        chunk_text = "".join(buffer)
                        buffer.clear()
                        yield self._sse(
                            "token",
                            StreamingEvent(event="token", token=chunk_text).model_dump(),
                        )

                if buffer:
                    yield self._sse(
                        "token",
                        StreamingEvent(event="token", token="".join(buffer)).model_dump(),
                    )

                answer = "".join(answer_parts)
                latency_ms = int((time.perf_counter() - start_time) * 1000)

                from backend.evaluation.groundedness import GroundednessScorer

                scorer = GroundednessScorer()
                groundedness = scorer.score_groundedness(answer, [doc.page_content for doc in results])

                yield self._sse(
                    "complete",
                    StreamingEvent(
                        event="complete",
                        answer=answer,
                        sources=sources,
                        retrieval_explanation=retrieval_explanation.model_dump() if retrieval_explanation else None,
                        groundedness=groundedness,
                        latency_ms=latency_ms,
                    ).model_dump(),
                )

                if self.conversation_service is not None:
                    self.conversation_service.append_message(
                        session_id,
                        "assistant",
                        answer,
                        sources=sources,
                        latency_ms=latency_ms,
                        workspace_id=workspace_id,
                    )
                    token_usage = estimate_turn_tokens(question, answer, context=context)
                    self.conversation_service.record_token_usage(
                        conversation_id=session_id,
                        session_id=session_id,
                        prompt_tokens=token_usage["prompt_tokens"],
                        completion_tokens=token_usage["completion_tokens"],
                        total_tokens=token_usage["total_tokens"],
                        model_name=os.getenv("DOCUMIND_OLLAMA_MODEL", OLLAMA_MODEL),
                        metadata={"streaming": True},
                    )

                if self.metrics_service is not None:
                    self.metrics_service.increment("stream_requests_total")
                    self.metrics_service.observe_latency("stream_latency_ms", latency_ms)

                try:
                    RETRIEVAL_LATENCY.observe(latency_ms / 1000.0)
                except Exception:
                    pass

                if self.cache_service is not None:
                    self.cache_service.set_streaming_cache(
                        session_id,
                        question,
                        {
                            "answer": answer,
                            "sources": sources,
                            "retrieval_explanation": retrieval_explanation.model_dump(mode="json") if retrieval_explanation else None,
                            "groundedness": groundedness,
                            "latency_ms": latency_ms,
                        },
                    )
            except asyncio.CancelledError:
                stop_event.set()
                if self.metrics_service is not None:
                    self.metrics_service.increment("stream_cancellations_total")
                raise
            finally:
                stop_event.set()
                producer_task.cancel()
        finally:
            if workspace_token is not None:
                current_workspace_id.reset(workspace_token)
            if tenant_token is not None:
                current_tenant_id.reset(tenant_token)