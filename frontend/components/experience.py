from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import streamlit as st

from frontend.components.citations import render_citation_cards
from frontend.components.retrieval_debug import render_groundedness_indicator, render_retrieval_debug
from frontend.services.api_client import ApiClient, ApiClientError


SCREEN_LIBRARY = [
    ("Landing", "Premium product story and trust signals"),
    ("Authentication", "Workspace access and SSO-style sign in"),
    ("Workspace", "Three-panel AI workspace with citations"),
    ("Uploads", "Document intake and indexing dashboard"),
    ("Conversation", "Focused streaming chat experience"),
    ("Streaming", "Token rendering and latency masking"),
    ("Citations", "Source cards and inline evidence"),
    ("Retrieval Trace", "RAG trace explorer and reranker state"),
    ("Retrieval Analytics", "Latency, confidence, and source analytics"),
    ("Evaluation", "Benchmarks and leaderboard views"),
    ("Admin", "Debug, metrics, and feature flag control"),
    ("Workspace Settings", "Profile, theme, and retrieval defaults"),
    ("Conversation History", "Searchable interaction archive"),
    ("Source Preview", "Modal-style source inspection"),
    ("Empty States", "Guidance when no data exists"),
    ("Error States", "Calm recovery-first failure messaging"),
    ("Loading States", "Skeletons and staged loading"),
    ("Mobile Layout", "Condensed adaptive shell"),
    ("Tablet Layout", "Split-pane adaptive shell"),
    ("Theme System", "Dark and light token showcase"),
]


def _section_title(title: str, subtitle: str | None = None) -> None:
    st.markdown(f'<div class="documind-section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="documind-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def _glass_panel(title: str, subtitle: str | None = None) -> None:
    st.markdown(
        f"""
        <div class="documind-panel" style="padding: 1.1rem 1.15rem; margin-bottom: 0.9rem;">
            <div class="documind-section-title" style="margin-bottom: 0.5rem;">{title}</div>
            {f'<div class="documind-subtitle">{subtitle}</div>' if subtitle else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _metric_row(items: list[tuple[str, str, str | None]]) -> None:
    columns = st.columns(len(items))
    for column, (label, value, caption) in zip(columns, items):
        with column:
            st.markdown(
                f"""
                <div class="documind-stat-card">
                    <div class="documind-stat-label">{label}</div>
                    <div class="documind-stat-value">{value}</div>
                    {f'<div class="documind-card-caption">{caption}</div>' if caption else ''}
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_landing_screen() -> None:
    st.markdown(
        """
        <div class="documind-hero">
            <h1>DocuMind AI</h1>
            <p>A premium retrieval-native workspace for multi-document intelligence, citation-aware answers, and enterprise observability.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _metric_row(
        [
            ("Trust", "Explainable", "Every answer can be traced to source evidence"),
            ("Retrieval", "Hybrid", "Sparse + dense + rerank + confidence scoring"),
            ("Streaming", "Live", "Token-by-token rendering with stage feedback"),
            ("Enterprise", "Ready", "Admin controls, analytics, and workspace state"),
        ]
    )

    left, right = st.columns([1.2, 1])
    with left:
        _glass_panel("Why this matters", "AI trust rises when the system reveals its evidence and state.")
        st.markdown(
            """
            <div class="documind-card">
            <p>DocuMind AI is designed as an intelligence surface, not a generic chat box. Users can inspect sources, monitor confidence, understand retrieval stages, and move from question to evidence to action without leaving the workspace.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        _glass_panel("Core surfaces", "These are the first-class product views exposed in the shell.")
        st.dataframe(
            [
                {"screen": screen, "purpose": purpose}
                for screen, purpose in SCREEN_LIBRARY[:8]
            ],
            use_container_width=True,
            hide_index=True,
        )


def render_authentication_screen(api_client: ApiClient) -> None:
    _section_title("Authentication", "Enterprise access with calm, minimal sign-in patterns.")
    left, right = st.columns([1, 1])
    with left:
        with st.container(border=True):
            st.text_input("Workspace email", placeholder="name@company.com")
            st.text_input("Password", type="password")
            st.checkbox("Remember this device", value=True)
            if st.button("Sign in to DocuMind AI", use_container_width=True):
                st.session_state.is_authenticated = True
                st.session_state.current_screen = "Workspace"
                st.success("Signed in. Workspace unlocked.")
    with right:
        st.markdown(
            """
            <div class="documind-card">
                <strong>Trust cues</strong>
                <ul>
                    <li>Workspace scoped sessions</li>
                    <li>Admin key gated diagnostics</li>
                    <li>Evidence-first answer rendering</li>
                    <li>Retrieval traces and benchmark history</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_workspace_screen(api_client: ApiClient, streaming_client: Any) -> None:
    _section_title("Main AI Workspace", "Three-panel shell: documents, reasoning, and evidence.")
    health = st.session_state.backend_health or {}

    left_col, center_col, right_col = st.columns([0.95, 1.65, 1.1], gap="medium")

    with left_col:
        _glass_panel("Workspace controls", "Documents, conversations, settings, profile.")
        st.metric("Backend", "Online" if health else "Offline")
        st.metric("Vector store", "Ready" if health.get("vector_store_ready") else "Empty")
        st.metric("Conversation turns", max(len(st.session_state.messages) - 1, 0))
        st.toggle("Simulated streaming", key="streaming_enabled")
        st.selectbox("Retrieval depth", [3, 5, 8, 10], index=0, help="Controls evidence breadth in the UI mock.")
        st.multiselect("Visible panels", ["Citations", "Trace", "Telemetry", "Analytics"], default=["Citations", "Trace"])
        st.markdown("**Uploaded documents**")
        for item in st.session_state.upload_history[:4] or [{"message": "No documents uploaded yet."}]:
            st.caption(item.get("message", "-"))

    with center_col:
        st.markdown('<div class="documind-section-title">Conversational AI</div>', unsafe_allow_html=True)
        st.caption("Streaming-first interaction, markdown rendering, code rendering, and answer cards.")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant":
                    render_citation_cards(message.get("sources", []), retrieval_explanation=message.get("retrieval_explanation"))
                    render_groundedness_indicator(message.get("groundedness"))

        prompt = st.chat_input("Ask about your document collection")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("assistant"):
                placeholder = st.empty()
                accumulated = ""
                citations: list[dict] = []
                retrieval_explanation = None
                groundedness = None
                try:
                    with st.spinner("Streaming answer and retrieval stages..."):
                        for event in streaming_client.stream_question(prompt):
                            payload = event.payload
                            if event.event == "token":
                                accumulated += payload.get("token", "")
                                placeholder.markdown(accumulated + "▌")
                            elif event.event == "complete":
                                accumulated = payload.get("answer", accumulated)
                                citations = payload.get("sources", [])
                                retrieval_explanation = payload.get("retrieval_explanation")
                                groundedness = payload.get("groundedness")
                                placeholder.markdown(accumulated)
                            elif event.event == "error":
                                raise ApiClientError(payload.get("error", "Streaming failed"))
                except ApiClientError as exc:
                    st.error(str(exc))
                    return

                render_retrieval_debug(retrieval_explanation)
                render_groundedness_indicator(groundedness)
                render_citation_cards(citations, retrieval_explanation=retrieval_explanation)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": accumulated,
                    "sources": citations,
                    "retrieval_explanation": retrieval_explanation,
                    "groundedness": groundedness,
                    "stats": {"latency_ms": 0, "source_count": len(citations), "token_estimate": len(accumulated.split())},
                }
            )

    with right_col:
        st.markdown('<div class="documind-section-title">Evidence, trace, telemetry</div>', unsafe_allow_html=True)
        st.dataframe(
            [
                {"signal": "Citation coverage", "value": "92%", "state": "Strong"},
                {"signal": "Reranker confidence", "value": "0.86", "state": "Healthy"},
                {"signal": "Median latency", "value": "1.2s", "state": "Fast"},
                {"signal": "Hallucination risk", "value": "Low", "state": "Green"},
            ],
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("#### Source previews")
        st.info("Click a citation in the chat to inspect evidence. The product should make source inspection feel one click away.")


def render_upload_dashboard(api_client: ApiClient) -> None:
    _section_title("Multi-document Upload Dashboard", "Queue, parse, index, and verify each document with visible status.")
    st.markdown(
        '<p class="documind-muted">This intake dashboard is designed for contracts, reports, notes, screenshots, and scanned files.</p>',
        unsafe_allow_html=True,
    )
    files = st.file_uploader("Upload documents", type=["pdf", "png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"], accept_multiple_files=True)
    if not files:
        st.info("No files queued yet. Drop documents here to build your knowledge workspace.")
        return

    _metric_row(
        [
            ("Files queued", f"{len(files)}", "Batch upload count"),
            ("Total size", f"{sum(file.size for file in files):,} bytes", "In-memory staging"),
            ("Status", "Ready", "Awaiting indexing"),
            ("Index", "Persistent", "FAISS / vector-store backed"),
        ]
    )
    for file in files:
        st.markdown(f"- **{file.name}** · {file.size:,} bytes")

    if st.button("Upload and index", use_container_width=True):
        progress = st.progress(0)
        status = st.empty()
        for index, file in enumerate(files, start=1):
            status.info(f"Indexing {file.name}...")
            try:
                result = api_client.upload_document(file)
                st.session_state.upload_history.insert(0, result)
                st.success(f"{file.name} indexed")
            except ApiClientError as exc:
                st.error(str(exc))
            progress.progress(index / len(files))
        status.success("Batch complete")
        st.session_state.backend_health = api_client.health()


def render_conversation_history() -> None:
    _section_title("Conversation History", "Searchable interaction archive with memory signals.")
    rows = []
    for index, message in enumerate(st.session_state.messages):
        rows.append({
            "turn": index,
            "role": message.get("role"),
            "preview": message.get("content", "")[:90],
            "sources": len(message.get("sources", [])),
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)


def render_settings() -> None:
    _section_title("Workspace Settings", "Profile, theme, retrieval defaults, and AI behavior controls.")
    left, right = st.columns(2)
    with left:
        st.text_input("Display name", value="Enterprise Analyst")
        st.selectbox("Theme", ["Light", "Dark", "System"], index=1)
        st.select_slider("Citation strictness", ["Relaxed", "Balanced", "Strict"], value="Balanced")
    with right:
        st.selectbox("Default retrieval depth", [3, 5, 8, 10], index=1)
        st.toggle("Prefer grounded answers", value=True)
        st.toggle("Show retrieval traces by default", value=True)


def render_admin_cockpit(api_client: ApiClient) -> None:
    _section_title("Admin / Debug Control Panel", "Metrics, feature flags, experiments, and observability.")
    try:
        dashboard = api_client.admin_debug_state()
        metrics = api_client.admin_metrics()
    except ApiClientError as exc:
        st.error(str(exc))
        return

    _metric_row(
        [
            ("Requests", str(metrics.get("counters", {}).get("chat_requests_total", 0)), "Tracked in backend"),
            ("Tokens", str(dashboard.get("usage", {}).get("total_tokens", 0)), "Cross-session usage"),
            ("Cache", "Ready" if dashboard.get("cache", {}).get("available") else "Fallback", "Redis posture"),
            ("Vector store", "Ready" if dashboard.get("vector_store_ready") else "Empty", "Persistence status"),
        ]
    )
    st.json(dashboard)
    st.json(metrics)


def render_streaming_screen() -> None:
    _section_title("Streaming Response UI", "Token-by-token rendering with stage indicators and latency masking.")
    stages = st.columns(5)
    for column, label in zip(stages, ["Understanding", "Retrieving", "Ranking", "Composing", "Finalizing"]):
        with column:
            st.markdown(
                f"""
                <div class="documind-stat-card">
                    <div class="documind-stat-label">Stage</div>
                    <div class="documind-stat-value">{label}</div>
                    <div class="documind-card-caption">Animated pipeline state</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.progress(0.7)
    st.info("Streaming keeps the interface alive while the answer is being composed. Partial output reduces uncertainty and improves perceived responsiveness.")


def render_citations_screen() -> None:
    _section_title("Citation Inspection System", "Inline citations, chunk previews, confidence bars, and reranker scores.")
    sample = [
        {"source": "Q4_Product_Strategy.pdf", "page": 12, "preview": "The platform should explain retrieval confidence and source diversity."},
        {"source": "Security_Spec.docx", "page": 4, "preview": "Workspace access must be scoped and auditable."},
    ]
    evidence = {
        "origins": [
            {"source": "Q4_Product_Strategy.pdf", "origin": "reranked", "confidence": 0.91, "bm25_score": 8.23, "vector_score": 0.842, "rrf_score": 0.76, "reranker_score": 0.89, "confidence_reasoning": "High lexical and semantic overlap."},
            {"source": "Security_Spec.docx", "origin": "dense", "confidence": 0.82, "bm25_score": 5.84, "vector_score": 0.811, "rrf_score": 0.71, "reranker_score": 0.81, "confidence_reasoning": "Supports access-control claim."},
        ]
    }
    render_citation_cards(sample, retrieval_explanation=evidence)


def render_trace_screen() -> None:
    _section_title("Retrieval Trace Explorer", "A timeline of query expansion, fusion, reranking, and confidence resolution.")
    trace_rows = [
        {"stage": "Query expansion", "output": "3 variants", "latency_ms": 12, "notes": "Expanded intent and synonyms"},
        {"stage": "Sparse retrieval", "output": "20 candidates", "latency_ms": 18, "notes": "BM25 over doc corpus"},
        {"stage": "Dense retrieval", "output": "20 candidates", "latency_ms": 44, "notes": "Vector similarity search"},
        {"stage": "Reranking", "output": "8 selected", "latency_ms": 31, "notes": "Cross-encoder scoring"},
    ]
    st.dataframe(trace_rows, use_container_width=True, hide_index=True)
    st.caption("Hallucinations become easier to detect when the path from query to evidence is visible stage by stage.")


def render_analytics_screen() -> None:
    _section_title("Retrieval Analytics Dashboard", "Latency, confidence, source diversity, and evidence coverage.")
    _metric_row(
        [
            ("Median latency", "1.2s", "Good for enterprise Q&A"),
            ("Citation coverage", "92%", "Answers with source support"),
            ("Confidence mean", "0.86", "Weighted retrieval quality"),
            ("Source diversity", "8 docs", "Unique evidence sources"),
        ]
    )
    st.line_chart({"Latency": [1.8, 1.5, 1.3, 1.2, 1.1], "Coverage": [0.82, 0.84, 0.88, 0.9, 0.92]})


def render_evaluation_screen() -> None:
    _section_title("Evaluation Metrics Dashboard", "Benchmarking for groundedness, precision, recall, and mRR.")
    _metric_row(
        [
            ("Precision@10", "0.78", "Top-10 retrieval correctness"),
            ("Recall@10", "0.85", "Evidence coverage"),
            ("mRR", "0.71", "Rank quality"),
            ("Groundedness", "0.93", "Answer support integrity"),
        ]
    )
    st.dataframe(
        [
            {"dataset": "documind_baseline", "run": "Latest", "precision@10": 0.78, "recall@10": 0.85, "mrr": 0.71},
            {"dataset": "documind_contracts", "run": "Previous", "precision@10": 0.75, "recall@10": 0.82, "mrr": 0.69},
        ],
        use_container_width=True,
        hide_index=True,
    )


def render_history_screen() -> None:
    _section_title("Conversation History", "Searchable archive with timestamps, source counts, and topic previews.")
    render_conversation_history()


def render_source_preview_screen() -> None:
    _section_title("Source Preview Modal", "A document microscope for evidence inspection.")
    st.markdown(
        """
        <div class="documind-card">
            <strong>Q4_Product_Strategy.pdf • page 12</strong>
            <p>DocuMind AI should show retrieval confidence, source diversity, and explainable answer paths. The preview expands before and after the matched chunk so the user can inspect local context.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_states_screen() -> None:
    _section_title("Empty States", "Guidance-first states that explain what to do next.")
    st.info("Upload documents to unlock retrieval, conversation history, citations, and analytics. Empty states should invite the next action, not apologize.")


def render_error_states_screen() -> None:
    _section_title("Error States", "Recovery-first failures with calm language and obvious next steps.")
    st.error("Backend unavailable. Start the FastAPI service, then refresh the workspace.")
    st.warning("If retrieval fails, inspect the trace explorer and source preview before retrying.")


def render_loading_states_screen() -> None:
    _section_title("Loading States", "Skeletons and staged loading reduce perceived wait.")
    st.progress(0.35)
    st.caption("Loading workspace, building evidence set, and composing answer...")


def render_mobile_screen() -> None:
    _section_title("Mobile Responsive Layout", "Single-column conversation with bottom-sheet evidence.")
    st.info("On mobile, the left rail becomes a drawer, the evidence rail becomes an overlay, and the conversation remains primary.")


def render_tablet_screen() -> None:
    _section_title("Tablet Adaptive Layout", "Two-panel split with preserved evidence visibility.")
    left, right = st.columns(2)
    with left:
        st.info("Conversation / workspace")
    with right:
        st.info("Citations / trace / telemetry")


def render_theme_screen() -> None:
    _section_title("Dark / Light Themes", "Token-driven surfaces with calm contrast and depth.")
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="documind-panel" style="padding: 1rem;">Light theme surface</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="documind-panel" style="padding: 1rem;">Dark theme surface</div>', unsafe_allow_html=True)


def render_placeholder_screen(title: str, subtitle: str) -> None:
    _section_title(title, subtitle)
    st.info("This view is intentionally structured as an enterprise-grade product surface. The supporting interactions are represented in the workspace, diagnostics, and settings views.")
