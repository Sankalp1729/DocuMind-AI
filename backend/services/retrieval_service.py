from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from backend.retrieval.bm25_retriever import BM25Retriever
from backend.retrieval.query_expansion import pseudo_relevance_feedback
from backend.retrieval.retrieval_fusion import reciprocal_rank_fusion
from backend.retrieval.vector_retriever import VectorRetriever
from backend.schemas.retrieval import RetrievalOrigin
from backend.schemas.telemetry import RetrievalTrace
from backend.services.telemetry_service import TelemetryService
from backend.core.metrics import RETRIEVAL_REQUESTS, RETRIEVAL_LATENCY
from opentelemetry import trace


logger = logging.getLogger(__name__)


class RetrievalService:
    """Orchestrates hybrid retrieval: BM25 + vector + RRF + reranking."""
    
    def __init__(self, vector_store_service, telemetry_service: Optional[TelemetryService] = None):
        self.vs_service = vector_store_service
        self.telemetry_service = telemetry_service
        self.bm25 = None
        self.vector_retriever = VectorRetriever(vector_store_service)
        self._reranker = None  # Lazy loaded
        self._documents_texts = None
        self._document_indices_to_ids = {}

    @staticmethod
    def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
        return max(minimum, min(maximum, value))

    @staticmethod
    def _normalize_scores(raw_scores: Dict[int, float], higher_is_better: bool = True) -> Dict[int, float]:
        if not raw_scores:
            return {}

        values = list(raw_scores.values())
        minimum = min(values)
        maximum = max(values)
        if maximum == minimum:
            return {key: 1.0 for key in raw_scores}

        span = maximum - minimum
        normalized: Dict[int, float] = {}
        for key, score in raw_scores.items():
            normalized[key] = (score - minimum) / span
            if not higher_is_better:
                normalized[key] = 1.0 - normalized[key]
        return normalized

    @staticmethod
    def _hallucination_risk(confidence: float) -> str:
        if confidence >= 0.75:
            return "low"
        if confidence >= 0.5:
            return "medium"
        return "high"

    def _candidate_confidence(
        self,
        doc_idx: int,
        rank: int,
        bm25_norm: Dict[int, float],
        vector_norm: Dict[int, float],
        rrf_norm: Dict[int, float],
        reranker_norm: Dict[int, float],
        bm25_score: Optional[float],
        vector_score: Optional[float],
        rrf_score: Optional[float],
        reranker_score: Optional[float],
    ) -> Tuple[float, Dict[str, float], str, str]:
        bm25_component = bm25_norm.get(doc_idx, 0.0)
        vector_component = vector_norm.get(doc_idx, 0.0)
        rrf_component = rrf_norm.get(doc_idx, 0.0)
        reranker_component = reranker_norm.get(doc_idx, 0.0)

        agreement = 0.0
        if bm25_score is not None and vector_score is not None:
            agreement = 1.0
        elif bm25_score is not None or vector_score is not None:
            agreement = 0.7

        rank_bonus = 1.0 if rank == 0 else max(0.0, 1.0 - (rank * 0.12))
        confidence = self._clamp(
            0.10
            + 0.18 * bm25_component
            + 0.18 * vector_component
            + 0.24 * rrf_component
            + 0.22 * reranker_component
            + 0.08 * agreement
            + 0.10 * rank_bonus,
        )

        score_breakdown = {
            "bm25_normalized": round(bm25_component, 4),
            "vector_normalized": round(vector_component, 4),
            "rrf_normalized": round(rrf_component, 4),
            "reranker_normalized": round(reranker_component, 4),
            "agreement": round(agreement, 4),
            "rank_bonus": round(rank_bonus, 4),
        }

        risk = self._hallucination_risk(confidence)
        if risk == "low":
            reasoning = "Strong agreement between sparse, dense, fusion, and reranker signals"
        elif risk == "medium":
            reasoning = "Partial agreement across retrieval signals"
        else:
            reasoning = "Weak retrieval agreement or low reranker support"

        return confidence, score_breakdown, risk, reasoning
    
    @property
    def reranker(self):
        """Lazy-load reranker to avoid heavy dependencies."""
        if self._reranker is None:
            try:
                from backend.retrieval.reranker import CrossEncoderReranker
                self._reranker = CrossEncoderReranker()
            except ImportError as e:
                logger.warning(f"Failed to load reranker: {e}, continuing without reranking")
                return None
        return self._reranker

    def _extract_documents_from_faiss(self) -> List[str]:
        """Extract all document texts from FAISS for BM25."""
        if self._documents_texts is not None:
            return self._documents_texts

        # Try to extract texts from the underlying vector store service
        try:
            vs = self.vs_service.get_vector_store()
            if vs is None:
                return []

            # Qdrant-like backend returns (backend, collection)
            if isinstance(vs, tuple) and len(vs) == 2:
                backend, collection = vs
                docs = backend.list_documents(collection)
                texts = []
                for idx, (doc_id, text, payload) in enumerate(docs):
                    texts.append(text)
                    self._document_indices_to_ids[idx] = doc_id
                self._documents_texts = texts
                return texts

            # FAISS/langchain vectorstore path
            vector_store = vs
            texts = []
            if hasattr(vector_store, 'docstore'):
                docstore = vector_store.docstore
                index_to_docstore_id = getattr(vector_store, 'index_to_docstore_id', {})
                for idx in range(len(index_to_docstore_id)):
                    doc_id = index_to_docstore_id[idx]
                    doc = docstore.search(doc_id)
                    if doc:
                        texts.append(doc.page_content)
                        self._document_indices_to_ids[idx] = doc_id

            self._documents_texts = texts
            return texts
        except Exception as e:
            logger.warning("Failed to extract documents from vector store: %s", e)
            return []

    def _ensure_bm25(self):
        """Build BM25 index if not already built."""
        if self.bm25 is not None:
            return
        
        documents_texts = self._extract_documents_from_faiss()
        if not documents_texts:
            import logging
            logging.getLogger(__name__).warning("Cannot build BM25: no documents extracted from FAISS")
            return
        
        self.bm25 = BM25Retriever.from_texts(documents_texts)

    def hybrid_retrieve(self, query: str, top_k: int = 10) -> Tuple[List, Dict[str, Any]]:
        """Perform hybrid retrieval with structured scores and telemetry."""
        trace_id = str(uuid4())
        # Prometheus metric: count retrieval requests
        try:
            RETRIEVAL_REQUESTS.labels(origin="hybrid").inc()
        except Exception:
            pass
        tracer = trace.get_tracer(__name__)
        total_start = time.perf_counter()
        stage_timings: Dict[str, float] = {}

        stage_start = time.perf_counter()
        self._ensure_bm25()
        stage_timings["bm25_index_build_ms"] = (time.perf_counter() - stage_start) * 1000

        documents_texts = self._extract_documents_from_faiss()
        if not documents_texts or self.bm25 is None:
            vector_start = time.perf_counter()
            vector_results = self.vector_retriever.retrieve(query, top_k=top_k)
            stage_timings["vector_retrieval_ms"] = (time.perf_counter() - vector_start) * 1000

            results = [doc for _doc_id, _score, doc in vector_results[:top_k]]
            vector_scores: Dict[int, float] = {}
            for rank, (doc_id, score, doc) in enumerate(vector_results[:top_k]):
                doc_idx = int(doc_id) if str(doc_id).isdigit() else rank
                vector_scores[doc_idx] = float(score)

            vector_norm = self._normalize_scores(vector_scores, higher_is_better=False)
            origins = []
            for rank, (doc_id, score, doc) in enumerate(vector_results[:top_k]):
                doc_idx = int(doc_id) if str(doc_id).isdigit() else rank
                confidence, score_breakdown, risk, reasoning = self._candidate_confidence(
                    doc_idx=doc_idx,
                    rank=rank,
                    bm25_norm={},
                    vector_norm=vector_norm,
                    rrf_norm={},
                    reranker_norm={},
                    bm25_score=None,
                    vector_score=float(score),
                    rrf_score=None,
                    reranker_score=None,
                )
                origins.append(
                    RetrievalOrigin(
                        source=doc.metadata.get("source_file", "unknown"),
                        origin="vector",
                        rank=rank,
                        document_id=str(doc_id),
                        bm25_score=None,
                        vector_score=float(score),
                        rrf_score=None,
                        reranker_score=None,
                        confidence=confidence,
                        score_breakdown=score_breakdown,
                        confidence_reasoning=reasoning,
                        hallucination_risk=risk,
                    )
                )

            latency_ms = (time.perf_counter() - total_start) * 1000
            stage_timings["total_ms"] = latency_ms
            retrieval_confidence = max((origin.confidence for origin in origins), default=0.0)
            hallucination_risk = max(
                (origin.hallucination_risk or "high" for origin in origins),
                key=lambda value: {"low": 0, "medium": 1, "high": 2}.get(value, 2),
                default="high",
            )

            retrieval_info = {
                "expanded_query": None,
                "num_dense": len(results),
                "num_sparse": 0,
                "num_fused": 0,
                "num_reranked": len(results),
                "origins": origins,
                "trace_id": trace_id,
                "stage_timings_ms": stage_timings,
                "retrieval_confidence": retrieval_confidence,
                "hallucination_risk": hallucination_risk,
                "latency_ms": latency_ms,
            }

            if self.telemetry_service is not None:
                self.telemetry_service.record_retrieval_trace(
                    RetrievalTrace(
                        trace_id=trace_id,
                        query=query,
                        expanded_query=None,
                        timestamp=datetime.now(timezone.utc),
                        total_ms=latency_ms,
                        retrieval_confidence=retrieval_confidence,
                        hallucination_risk=hallucination_risk,
                        num_dense_candidates=len(results),
                        num_sparse_candidates=0,
                        num_fused=0,
                        num_reranked=len(results),
                        stage_timings_ms=stage_timings,
                        origins=origins,
                        metadata={"fallback": True},
                    )
                )

            return results, retrieval_info

        query_expansion_start = time.perf_counter()
        bm25_hits = self.bm25.retrieve(query, top_k=min(top_k * 2, len(documents_texts)))
        bm25_doc_ids = [i for i, _ in bm25_hits]
        docs_for_expansion = [documents_texts[i] for i in bm25_doc_ids[:3] if i < len(documents_texts)]
        expansion_terms = pseudo_relevance_feedback(docs_for_expansion, top_m=3, top_terms=5) if docs_for_expansion else []
        expanded_query = query + " " + " ".join(expansion_terms) if expansion_terms else query
        stage_timings["query_expansion_ms"] = (time.perf_counter() - query_expansion_start) * 1000

        vector_start = time.perf_counter()
        vector_hits = self.vector_retriever.retrieve(expanded_query, top_k=min(top_k * 2, len(documents_texts)))
        stage_timings["vector_retrieval_ms"] = (time.perf_counter() - vector_start) * 1000

        bm25_scores = {int(doc_id): float(score) for doc_id, score in bm25_hits}
        vector_scores = {int(doc_id): float(score) for doc_id, score, _doc in vector_hits if str(doc_id).isdigit()}

        fusion_start = time.perf_counter()
        bm25_list = [(str(doc_id), score) for doc_id, score in bm25_scores.items()]
        vector_list = [(str(doc_id), score) for doc_id, score in vector_scores.items()]
        fused = reciprocal_rank_fusion([bm25_list, vector_list], k=60)
        stage_timings["fusion_ms"] = (time.perf_counter() - fusion_start) * 1000

        top_candidate_indices = [int(doc_id) for doc_id, _ in fused[: top_k * 2] if str(doc_id).isdigit()]
        candidate_texts = [documents_texts[i] for i in top_candidate_indices if i < len(documents_texts)]

        rerank_start = time.perf_counter()
        if self.reranker:
            try:
                reranked = self.reranker.rerank(query, candidate_texts)
            except Exception as exc:
                logger.warning("Reranking failed: %s, using RRF ordering", exc)
                reranked = [(i, 0.0) for i in range(min(top_k, len(top_candidate_indices)))]
        else:
            reranked = [(i, 0.0) for i in range(min(top_k, len(top_candidate_indices)))]
        stage_timings["reranking_ms"] = (time.perf_counter() - rerank_start) * 1000

        reranker_scores: Dict[int, float] = {}
        for candidate_position, score in reranked:
            if candidate_position < len(top_candidate_indices):
                reranker_scores[top_candidate_indices[candidate_position]] = float(score)

        rrf_scores = {int(doc_id): float(score) for doc_id, score in fused if str(doc_id).isdigit()}
        bm25_norm = self._normalize_scores(bm25_scores, higher_is_better=True)
        vector_norm = self._normalize_scores(vector_scores, higher_is_better=False)
        rrf_norm = self._normalize_scores(rrf_scores, higher_is_better=True)
        reranker_norm = self._normalize_scores(reranker_scores, higher_is_better=True)

        results = []
        origins = []
        vector_store = self.vs_service.get_vector_store()

        for rank, (candidate_position, reranker_score) in enumerate(reranked[:top_k]):
            if candidate_position >= len(top_candidate_indices):
                continue

            doc_idx = top_candidate_indices[candidate_position]
            if doc_idx >= len(documents_texts):
                continue

            doc_id = self._document_indices_to_ids.get(doc_idx)
            if not (vector_store and hasattr(vector_store, "docstore") and doc_id):
                continue

            doc = vector_store.docstore.search(doc_id)
            if not doc:
                continue

            results.append(doc)

            bm25_score = bm25_scores.get(doc_idx)
            vector_score = vector_scores.get(doc_idx)
            rrf_score = rrf_scores.get(doc_idx)
            reranker_score_value = float(reranker_score)
            confidence, score_breakdown, hallucination_risk, confidence_reasoning = self._candidate_confidence(
                doc_idx=doc_idx,
                rank=rank,
                bm25_norm=bm25_norm,
                vector_norm=vector_norm,
                rrf_norm=rrf_norm,
                reranker_norm=reranker_norm,
                bm25_score=bm25_score,
                vector_score=vector_score,
                rrf_score=rrf_score,
                reranker_score=reranker_score_value,
            )

            origins.append(
                RetrievalOrigin(
                    source=doc.metadata.get("source_file", "unknown"),
                    origin="fused" if bm25_score is not None and vector_score is not None else ("vector" if vector_score is not None else "bm25"),
                    rank=rank,
                    document_id=doc_id,
                    bm25_score=bm25_score,
                    vector_score=vector_score,
                    rrf_score=rrf_score,
                    reranker_score=reranker_score_value,
                    confidence=confidence,
                    score_breakdown=score_breakdown,
                    confidence_reasoning=confidence_reasoning,
                    hallucination_risk=hallucination_risk,
                )
            )

        latency_ms = (time.perf_counter() - total_start) * 1000
        stage_timings["total_ms"] = latency_ms
        retrieval_confidence = max((origin.confidence for origin in origins), default=0.0)
        risk_rank = {"low": 0, "medium": 1, "high": 2}
        hallucination_risk = max((origin.hallucination_risk or "high" for origin in origins), key=lambda value: risk_rank.get(value, 2), default="high")

        retrieval_info = {
            "expanded_query": expanded_query if expansion_terms else None,
            "num_dense": len(vector_hits),
            "num_sparse": len(bm25_hits),
            "num_fused": len(fused),
            "num_reranked": len(reranked),
            "origins": origins,
            "trace_id": trace_id,
            "stage_timings_ms": stage_timings,
            "retrieval_confidence": retrieval_confidence,
            "hallucination_risk": hallucination_risk,
            "latency_ms": latency_ms,
        }

        if self.telemetry_service is not None:
            self.telemetry_service.record_retrieval_trace(
                RetrievalTrace(
                    trace_id=trace_id,
                    query=query,
                    expanded_query=expanded_query if expansion_terms else None,
                    timestamp=datetime.now(timezone.utc),
                    total_ms=latency_ms,
                    retrieval_confidence=retrieval_confidence,
                    hallucination_risk=hallucination_risk,
                    num_dense_candidates=len(vector_hits),
                    num_sparse_candidates=len(bm25_hits),
                    num_fused=len(fused),
                    num_reranked=len(reranked),
                    stage_timings_ms=stage_timings,
                    origins=origins,
                    metadata={"candidate_count": len(top_candidate_indices)},
                )
            )

            # observe retrieval latency
            try:
                RETRIEVAL_LATENCY.observe(latency_ms / 1000.0)
            except Exception:
                pass

            # Add tracing span
            try:
                with tracer.start_as_current_span("retrieval.hybrid", attributes={"trace_id": trace_id, "num_results": len(results)}):
                    pass
            except Exception:
                pass

            return results, retrieval_info

