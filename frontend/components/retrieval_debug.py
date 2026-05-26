import streamlit as st
from typing import Optional, Dict, Any


def render_retrieval_debug(retrieval_explanation: Optional[Dict[str, Any]]):
    """Render detailed retrieval explanation and diagnostics."""
    if not retrieval_explanation:
        return
    
    with st.expander("📊 Retrieval Pipeline Diagnostics", expanded=False):
        st.markdown("### Query Processing")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Original Query", retrieval_explanation.get("query", ""))
        
        if retrieval_explanation.get("expanded_query"):
            with col2:
                st.metric("Expanded Query", retrieval_explanation.get("expanded_query", ""))

        meta_cols = st.columns(3)
        with meta_cols[0]:
            st.metric("Trace ID", retrieval_explanation.get("trace_id", "n/a"))
        with meta_cols[1]:
            st.metric("Confidence", f"{retrieval_explanation.get('retrieval_confidence', 0.0):.0%}")
        with meta_cols[2]:
            st.metric("Hallucination Risk", retrieval_explanation.get("hallucination_risk", "unknown").upper())
        
        st.markdown("### Retrieval Breakdown")
        metrics_cols = st.columns(4)
        
        with metrics_cols[0]:
            st.metric("Sparse (BM25)", retrieval_explanation.get("num_sparse_candidates", 0))
        with metrics_cols[1]:
            st.metric("Dense (Vector)", retrieval_explanation.get("num_dense_candidates", 0))
        with metrics_cols[2]:
            st.metric("After Fusion", retrieval_explanation.get("num_fused", 0))
        with metrics_cols[3]:
            st.metric("After Rerank", retrieval_explanation.get("num_reranked", 0))
        
        st.markdown("### Performance")
        st.metric("Total Latency (ms)", f"{retrieval_explanation.get('latency_ms', 0):.1f}")
        if retrieval_explanation.get("stage_timings_ms"):
            st.json(retrieval_explanation.get("stage_timings_ms"))
        
        st.markdown("### Document Origins & Scores")
        origins = retrieval_explanation.get("origins", [])
        if origins:
            for i, origin in enumerate(origins):
                with st.container():
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        title = f"**{i+1}. {origin['source']}**"
                        if origin.get("origin"):
                            title = f"{title} ({origin['origin']})"
                        st.write(title)
                    
                    with col2:
                        confidence = origin.get("confidence", 0.5)
                        st.progress(min(1.0, max(0.0, confidence)))
                    
                    with col3:
                        st.caption(f"{confidence:.0%}")
                    
                    # Show detailed scores
                    score_cols = st.columns(4)
                    
                    if origin.get("bm25_score"):
                        with score_cols[0]:
                            st.caption(f"BM25: {origin['bm25_score']:.2f}")
                    
                    if origin.get("vector_score"):
                        with score_cols[1]:
                            st.caption(f"Vector: {origin['vector_score']:.3f}")
                    
                    if origin.get("rrf_score"):
                        with score_cols[2]:
                            st.caption(f"RRF: {origin['rrf_score']:.3f}")
                    
                    if origin.get("reranker_score"):
                        with score_cols[3]:
                            st.caption(f"Rerank: {origin['reranker_score']:.2f}")

                    if origin.get("score_breakdown"):
                        st.caption(f"Score breakdown: {origin['score_breakdown']}")
                    if origin.get("confidence_reasoning"):
                        st.caption(f"Reasoning: {origin['confidence_reasoning']}")
                    if origin.get("hallucination_risk"):
                        st.caption(f"Risk: {origin['hallucination_risk']}")


def render_groundedness_indicator(groundedness: Optional[Dict[str, Any]]):
    """Render groundedness/hallucination indicator."""
    if not groundedness:
        return
    
    with st.container():
        is_grounded = groundedness.get("is_grounded", False)
        confidence = groundedness.get("confidence", 0.5)
        risk = groundedness.get("hallucination_risk", "medium")
        reasoning = groundedness.get("reasoning", "")
        
        # Color based on risk
        if risk == "low":
            color = "🟢"
        elif risk == "medium":
            color = "🟡"
        else:
            color = "🔴"
        
        st.write(f"{color} **Groundedness Assessment**")
        
        cols = st.columns([1, 1, 1])
        with cols[0]:
            st.metric("Grounded", "✓ Yes" if is_grounded else "✗ No")
        with cols[1]:
            st.metric("Confidence", f"{confidence:.0%}")
        with cols[2]:
            st.metric("Hallucination Risk", risk.upper())
        
        if reasoning:
            st.caption(f"__{reasoning}__")
        if groundedness.get("support_summary"):
            st.caption(groundedness.get("support_summary"))
        if groundedness.get("hallucination_signals"):
            st.caption(f"Signals: {', '.join(groundedness.get('hallucination_signals', []))}")
