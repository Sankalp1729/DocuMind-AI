from __future__ import annotations

import uuid
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy import func

from backend.core.config import (
    ENABLE_ADMIN_ENDPOINTS,
    ENABLE_AGENTIC_RAG,
    ENABLE_ANALYTICS_PERSISTENCE,
    ENABLE_AUTH,
    ENABLE_CONVERSATION_PERSISTENCE,
    ENABLE_HYBRID_RETRIEVAL,
    ENABLE_METRICS,
    ENABLE_RATE_LIMITING,
    ENABLE_REDIS_CACHE,
    TOP_K,
    REDIS_RETRIEVAL_CACHE_TTL_SECONDS,
    REDIS_SESSION_TTL_SECONDS,
    REDIS_STREAM_TTL_SECONDS,
)
from backend.persistence.models import (
    AnalyticsEventRecord,
    ConversationRecord,
    MessageRecord,
    RetrievalTraceRecord,
    TokenUsageRecord,
    WorkspaceRecord,
)
from backend.services.auth_service import validate_admin_api_key


router = APIRouter(prefix="/admin", tags=["admin"])


def require_admin(request: Request, api_key: str | None):
    if validate_admin_api_key(api_key):
        return

    roles = set(getattr(request.state, "current_roles", ()) or ())
    if roles.intersection({"owner", "admin"}):
        return

    if getattr(request.state, "current_user", None) is not None and getattr(request.state, "authenticated", False):
        if roles.intersection({"service"}):
            return

        raise HTTPException(status_code=401, detail="Invalid admin API key")


def _usage_summary(request: Request) -> dict[str, object]:
    database_service = getattr(request.app.state, "database_service", None)
    if database_service is None:
        return {}

    with database_service.session_scope() as session:
        conversation_count = session.query(func.count(ConversationRecord.id)).scalar() or 0
        message_count = session.query(func.count(MessageRecord.id)).scalar() or 0
        token_count = session.query(func.count(TokenUsageRecord.id)).scalar() or 0
        analytics_count = session.query(func.count(AnalyticsEventRecord.id)).scalar() or 0
        retrieval_trace_count = session.query(func.count(RetrievalTraceRecord.id)).scalar() or 0
        workspace_count = session.query(func.count(WorkspaceRecord.id)).scalar() or 0

        token_totals = session.query(
            func.coalesce(func.sum(TokenUsageRecord.prompt_tokens), 0),
            func.coalesce(func.sum(TokenUsageRecord.completion_tokens), 0),
            func.coalesce(func.sum(TokenUsageRecord.total_tokens), 0),
            func.max(TokenUsageRecord.created_at),
        ).one()

    prompt_tokens, completion_tokens, total_tokens, latest_token_event = token_totals

    return {
        "conversations": conversation_count,
        "messages": message_count,
        "token_usage_records": token_count,
        "analytics_events": analytics_count,
        "retrieval_traces": retrieval_trace_count,
        "workspaces": workspace_count,
        "prompt_tokens": int(prompt_tokens or 0),
        "completion_tokens": int(completion_tokens or 0),
        "total_tokens": int(total_tokens or 0),
        "latest_token_event": latest_token_event.isoformat() if latest_token_event else None,
    }


@router.get("/metrics")
def get_metrics(request: Request, x_api_key: str | None = Header(default=None)):
    require_admin(request, x_api_key)
    return request.app.state.metrics_service.snapshot()


@router.get("/debug/state")
def debug_state(request: Request, x_api_key: str | None = Header(default=None)):
    require_admin(request, x_api_key)
    vector_store_service = request.app.state.vector_store_service
    retrieval_service = request.app.state.retrieval_service
    cache_service = request.app.state.cache_service
    metrics_snapshot = request.app.state.metrics_service.snapshot()
    
    return {
        "vector_store_ready": vector_store_service.is_ready(),
        "bm25_loaded": retrieval_service.bm25 is not None if retrieval_service else False,
        "conversations": request.app.state.conversation_service.list_conversations(limit=10),
        "metrics": metrics_snapshot,
        "redis_ready": cache_service.is_available() if cache_service else False,
        "database_ready": hasattr(request.app.state, "database_service"),
        "feature_flags": {
            "admin_endpoints": ENABLE_ADMIN_ENDPOINTS,
            "auth": ENABLE_AUTH,
            "agentic_rag": ENABLE_AGENTIC_RAG,
            "analytics_persistence": ENABLE_ANALYTICS_PERSISTENCE,
            "conversation_persistence": ENABLE_CONVERSATION_PERSISTENCE,
            "hybrid_retrieval": ENABLE_HYBRID_RETRIEVAL,
            "metrics": ENABLE_METRICS,
            "rate_limiting": ENABLE_RATE_LIMITING,
            "redis_cache": ENABLE_REDIS_CACHE,
        },
        "retrieval_configuration": {
            "top_k": TOP_K,
            "bm25_loaded": retrieval_service.bm25 is not None if retrieval_service else False,
        },
        "cache": {
            "available": cache_service.is_available() if cache_service else False,
            "session_cache_ttl_seconds": REDIS_SESSION_TTL_SECONDS,
            "stream_cache_ttl_seconds": REDIS_STREAM_TTL_SECONDS,
            "retrieval_cache_ttl_seconds": REDIS_RETRIEVAL_CACHE_TTL_SECONDS,
        },
        "usage": _usage_summary(request),
    }


@router.get("/retrieval-debug")
def retrieval_debug(request: Request, query: str, x_api_key: str | None = Header(default=None)):
    """Debug retrieval pipeline for a query."""
    require_admin(request, x_api_key)
    
    if not query or len(query.strip()) == 0:
        raise HTTPException(status_code=400, detail="Query parameter required")
    
    rag_service = request.app.state.rag_service
    telemetry_service = request.app.state.telemetry_service
    
    # Run retrieval
    result = rag_service.retrieve(query)
    if result is None:
        raise HTTPException(status_code=404, detail="Retrieval failed or no documents available")
    
    context, results, retrieval_explanation = result
    traces = telemetry_service.list_traces(limit=1, query=query)
    
    return JSONResponse({
        "query": query,
        "retrieval_explanation": retrieval_explanation.model_dump() if retrieval_explanation else None,
        "num_results": len(results),
        "trace": traces[-1].model_dump(mode="json") if traces else None,
        "top_sources": [
            {
                "source": r.metadata.get("source_file"),
                "page": r.metadata.get("page"),
                "preview": r.page_content[:200],
            }
            for r in results[:5]
        ],
    })


@router.post("/evaluate")
def evaluate_benchmark(
    request: Request,
    run_id: Optional[str] = None,
    dataset_name: Optional[str] = None,
    top_k: int = 10,
    x_api_key: str | None = Header(default=None),
):
    """Run a benchmark evaluation on the current retrieval stack."""
    require_admin(request, x_api_key)
    
    run_id = run_id or str(uuid.uuid4())[:8]
    evaluation_service = request.app.state.evaluation_service
    dataset_name = dataset_name or "documind_baseline"

    try:
        benchmark_result = evaluation_service.run_dataset_benchmark(
            rag_service=request.app.state.rag_service,
            dataset_name=dataset_name,
            benchmark_id=run_id,
            top_k=top_k,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    filepath = evaluation_service.save_benchmark_result(run_id, benchmark_result)
    
    return JSONResponse({
        "run_id": run_id,
        "dataset_name": dataset_name,
        "metrics": benchmark_result.retrieval_metrics,
        "saved_to": str(filepath),
        "dashboard": evaluation_service.benchmark_dashboard(),
    })


@router.get("/evaluation/datasets")
def list_evaluation_datasets(request: Request, x_api_key: str | None = Header(default=None)):
    require_admin(request, x_api_key)
    evaluation_service = request.app.state.evaluation_service
    datasets = evaluation_service.list_datasets()
    return JSONResponse({"datasets": [dataset.model_dump(mode="json") for dataset in datasets]})


@router.get("/evaluation/history")
def evaluation_history(request: Request, x_api_key: str | None = Header(default=None)):
    require_admin(request, x_api_key)
    evaluation_service = request.app.state.evaluation_service
    history = evaluation_service.load_benchmark_history()
    return JSONResponse({"history": [entry.model_dump(mode="json") for entry in history]})


@router.get("/evaluation/leaderboard")
def evaluation_leaderboard(request: Request, x_api_key: str | None = Header(default=None)):
    require_admin(request, x_api_key)
    evaluation_service = request.app.state.evaluation_service
    leaderboard = evaluation_service.leaderboard()
    return JSONResponse({"leaderboard": [entry.model_dump(mode="json") for entry in leaderboard]})


@router.get("/benchmarks")
def list_benchmarks(request: Request, x_api_key: str | None = Header(default=None)):
    """List all benchmark results."""
    require_admin(request, x_api_key)
    evaluation_service = request.app.state.evaluation_service
    results = evaluation_service.list_benchmark_results()
    return JSONResponse({"benchmarks": results, "dashboard": evaluation_service.benchmark_dashboard()})


@router.get("/retrieval-trace")
def retrieval_trace(request: Request, query: str, x_api_key: str | None = Header(default=None)):
    """Detailed trace of retrieval pipeline execution."""
    require_admin(request, x_api_key)
    
    if not query or len(query.strip()) == 0:
        raise HTTPException(status_code=400, detail="Query parameter required")
    
    rag_service = request.app.state.rag_service
    
    result = rag_service.retrieve(query)
    if result is None:
        raise HTTPException(status_code=404, detail="Retrieval failed")
    
    context, results, retrieval_explanation = result
    telemetry_service = request.app.state.telemetry_service
    traces = telemetry_service.list_traces(limit=5, query=query)
    
    return JSONResponse({
        "query": query,
        "telemetry": telemetry_service.get_summary().model_dump(mode="json"),
        "recent_traces": [trace.model_dump(mode="json") for trace in traces],
        "retrieval_explanation": retrieval_explanation.model_dump() if retrieval_explanation else None,
        "num_results": len(results),
    })
