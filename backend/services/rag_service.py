import logging
import time
from dataclasses import asdict
from typing import Optional, Tuple

from backend.core.config import TOP_K, ENABLE_AGENTIC_RAG
from backend.rag.rag_chain import generate_answer
from backend.services.cache_service import CacheService
from backend.services.vector_store_service import VectorStoreService
from backend.services.retrieval_service import RetrievalService
from backend.services.telemetry_service import TelemetryService
from backend.evaluation.groundedness import GroundednessScorer
from backend.schemas.retrieval import RetrievalExplanation, GroundednessScore


logger = logging.getLogger(__name__)


class RagService:
    def __init__(
        self,
        vector_store_service: VectorStoreService,
        retrieval_service: Optional[RetrievalService] = None,
        telemetry_service: Optional[TelemetryService] = None,
        cache_service: CacheService | None = None,
        agentic_service: object | None = None,
    ):
        self.vector_store_service = vector_store_service
        self.retrieval_service = retrieval_service
        self.telemetry_service = telemetry_service or TelemetryService()
        self.cache_service = cache_service
        self.agentic_service = agentic_service
        self.groundedness_scorer = GroundednessScorer()

    def refresh_vector_store(self):
        return self.vector_store_service.load_vector_store()

    def ingest_chunks(self, chunks):
        return self.vector_store_service.rebuild_vector_store(chunks)

    def answer(self, question: str):
        """Generate answer with full retrieval explainability."""
        if ENABLE_AGENTIC_RAG and self.agentic_service is not None:
            agentic_result = self.agentic_service.answer(question)
            if agentic_result is not None:
                result = self.retrieve(question)
                sources = []
                if result is not None:
                    _, docs, _ = result
                    sources = [
                        {
                            "source": doc.metadata.get("source_file"),
                            "page": doc.metadata.get("page"),
                            "preview": doc.page_content[:500],
                        }
                        for doc in docs
                    ]
                groundedness_result = self.groundedness_scorer.score_groundedness(agentic_result["answer"], [item["preview"] for item in sources if item.get("preview")]) if sources else {"score": 0.0, "reasoning": "No sources"}
                groundedness = GroundednessScore(**groundedness_result)
                return {
                    "question": question,
                    "answer": agentic_result["answer"],
                    "sources": sources,
                    "retrieval_explanation": result[2] if result is not None else None,
                    "groundedness": groundedness,
                    "agent_plan": asdict(agentic_result.get("plan")) if agentic_result.get("plan") is not None else None,
                    "agent_reflections": agentic_result.get("reflections"),
                    "agent_observations": [asdict(obs) for obs in agentic_result.get("observations", [])],
                }

        result = self.retrieve(question)

        if result is None:
            return None

        context, results, retrieval_explanation = result

        try:
            answer = generate_answer(context, question)
        except Exception:
            logger.exception("RAG answer generation failed")
            raise

        sources = [
            {
                "source": doc.metadata.get("source_file"),
                "page": doc.metadata.get("page"),
                "preview": doc.page_content[:500],
            }
            for doc in results
        ]

        # Score groundedness
        passage_texts = [doc.page_content for doc in results]
        groundedness_result = self.groundedness_scorer.score_groundedness(answer, passage_texts)
        groundedness = GroundednessScore(**groundedness_result)

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "retrieval_explanation": retrieval_explanation,
            "groundedness": groundedness,
        }

    def retrieve(self, question: str) -> Optional[Tuple]:
        """Hybrid retrieve: use RetrievalService if available, otherwise fall back to vector store."""
        start_time = time.perf_counter()

        if self.cache_service is not None:
            cached_payload = self.cache_service.get_retrieval_cache(question, TOP_K)
            if cached_payload:
                from langchain_core.documents import Document

                cached_results = [Document(page_content=item["page_content"], metadata=item.get("metadata", {})) for item in cached_payload.get("documents", [])]
                retrieval_explanation = RetrievalExplanation.model_validate(cached_payload["retrieval_explanation"])
                context = cached_payload.get("context") or "\n\n".join(doc.page_content for doc in cached_results)
                return context, cached_results, retrieval_explanation

        if self.retrieval_service:
            # Use hybrid retrieval with RetrievalService
            results, retrieval_info = self.retrieval_service.hybrid_retrieve(question, top_k=TOP_K)
            
            if not results:
                return None
            
            # Build retrieval explanation
            retrieval_latency_ms = (time.perf_counter() - start_time) * 1000
            retrieval_explanation = RetrievalExplanation(
                query=question,
                expanded_query=retrieval_info.get("expanded_query"),
                trace_id=retrieval_info.get("trace_id"),
                num_dense_candidates=retrieval_info.get("num_dense", 0),
                num_sparse_candidates=retrieval_info.get("num_sparse", 0),
                num_fused=retrieval_info.get("num_fused", 0),
                num_reranked=len(results),
                origins=retrieval_info.get("origins", []),
                retrieval_confidence=retrieval_info.get("retrieval_confidence", 0.0),
                hallucination_risk=retrieval_info.get("hallucination_risk"),
                stage_timings_ms=retrieval_info.get("stage_timings_ms", {}),
                latency_ms=retrieval_latency_ms,
            )
            
            context = "\n\n".join(doc.page_content for doc in results)
            if self.cache_service is not None:
                self.cache_service.set_retrieval_cache(
                    question,
                    TOP_K,
                    {
                        "context": context,
                        "documents": [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in results],
                        "retrieval_explanation": retrieval_explanation.model_dump(mode="json"),
                    },
                )
            return context, results, retrieval_explanation
        
        else:
            # Fallback to basic vector search
            vector_store = self.vector_store_service.get_vector_store()

            if vector_store is None:
                return None

            results = vector_store.similarity_search(question, k=TOP_K)
            context = "\n\n".join(doc.page_content for doc in results)
            
            retrieval_latency_ms = (time.perf_counter() - start_time) * 1000
            retrieval_explanation = RetrievalExplanation(
                query=question,
                expanded_query=None,
                trace_id=None,
                num_dense_candidates=len(results),
                num_sparse_candidates=0,
                num_fused=0,
                num_reranked=len(results),
                origins=[],
                retrieval_confidence=0.0,
                hallucination_risk="high",
                stage_timings_ms={},
                latency_ms=retrieval_latency_ms,
            )

            if self.cache_service is not None:
                self.cache_service.set_retrieval_cache(
                    question,
                    TOP_K,
                    {
                        "context": context,
                        "documents": [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in results],
                        "retrieval_explanation": retrieval_explanation.model_dump(mode="json"),
                    },
                )
            
            return context, results, retrieval_explanation
