from __future__ import annotations

import streamlit as st


def render_citation_cards(citations: list[dict], retrieval_explanation: dict | None = None) -> None:
    """Render citation cards with optional retrieval metadata."""
    if not citations:
        st.caption("No citations returned for this answer.")
        return

    st.markdown("#### Sources")
    
    # Map citation sources to retrieval origins for detailed scoring
    origins_by_source = {}
    if retrieval_explanation:
        for origin in retrieval_explanation.get("origins", []):
            origins_by_source[origin["source"]] = origin
    
    for index, citation in enumerate(citations, start=1):
        source = citation.get("source") or "Unknown source"
        page = citation.get("page")
        preview = citation.get("preview") or "No preview available."
        
        # Get retrieval metadata if available
        origin = origins_by_source.get(source)

        title = f"{index}. {source}"
        if page is not None:
            title = f"{title} • page {page}"
        
        # Add confidence indicator to title
        if origin and origin.get("confidence"):
            confidence = origin["confidence"]
            confidence_pct = f"{confidence:.0%}"
            title = f"{title} [{confidence_pct}]"

        with st.expander(title, expanded=False):
            left, right = st.columns([2, 1])
            with left:
                st.markdown(f"**File**: {source}")
                st.markdown(f"**Page**: {page if page is not None else 'N/A'}")
            with right:
                st.markdown("**Preview**")
                st.caption("Retrieved chunk excerpt")

            st.code(preview, language="text")
            
            # Show detailed retrieval scores if available
            if origin:
                st.markdown("---")
                st.markdown("**Retrieval Scores**")
                
                score_cols = st.columns(4)
                
                if origin.get("bm25_score") is not None:
                    with score_cols[0]:
                        st.metric("BM25", f"{origin['bm25_score']:.2f}")
                
                if origin.get("vector_score") is not None:
                    with score_cols[1]:
                        st.metric("Vector", f"{origin['vector_score']:.3f}")
                
                if origin.get("rrf_score") is not None:
                    with score_cols[2]:
                        st.metric("RRF", f"{origin['rrf_score']:.3f}")
                
                if origin.get("reranker_score") is not None:
                    with score_cols[3]:
                        st.metric("Rerank", f"{origin['reranker_score']:.2f}")
                
                st.caption(f"Origin: {origin.get('origin', 'unknown')}")
                if origin.get("confidence_reasoning"):
                    st.caption(origin["confidence_reasoning"])
                if origin.get("score_breakdown"):
                    st.caption(f"Breakdown: {origin['score_breakdown']}")
                if origin.get("hallucination_risk"):
                    st.caption(f"Risk: {origin['hallucination_risk']}")

