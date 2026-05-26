from __future__ import annotations

import os

import streamlit as st

from frontend.components.citations import render_citation_cards
from frontend.components.chat import render_chat
from frontend.components.retrieval_debug import render_retrieval_debug, render_groundedness_indicator
from frontend.services.api_client import ApiClient, ApiClientError
from frontend.utils.session import initialize_session_state, reset_chat_history


APP_TITLE = "DocuMind AI"
APP_DESCRIPTION = "Enterprise RAG workspace with document upload, grounded chat, and retrieval diagnostics."


def _apply_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(79, 70, 229, 0.18), transparent 30%),
                radial-gradient(circle at top right, rgba(16, 185, 129, 0.14), transparent 24%),
                linear-gradient(180deg, #08111f 0%, #0b1526 42%, #0f172a 100%);
            color: #e5eefb;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .documind-shell {
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 24px;
            background: rgba(9, 17, 34, 0.76);
            box-shadow: 0 20px 60px rgba(2, 8, 23, 0.4);
            padding: 1.2rem 1.3rem;
        }
        .documind-kicker {
            text-transform: uppercase;
            letter-spacing: 0.24em;
            font-size: 0.72rem;
            color: #8fb8ff;
            margin-bottom: 0.35rem;
        }
        .documind-title {
            font-size: 2.4rem;
            font-weight: 800;
            margin: 0;
            color: #f8fbff;
        }
        .documind-subtitle {
            color: #b7c6df;
            max-width: 72ch;
            margin-top: 0.5rem;
        }
        .documind-muted {
            color: #9eb0cb;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _api_client() -> ApiClient:
    return ApiClient(
        base_url=st.session_state.api_base_url,
        session_id=st.session_state.session_id,
        admin_api_key=st.session_state.admin_api_key or None,
    )


def _render_header() -> None:
    st.markdown(
        """
        <div class="documind-shell">
          <div class="documind-kicker">Connected knowledge workspace</div>
          <h1 class="documind-title">DocuMind AI</h1>
          <p class="documind-subtitle">Upload a document, ask a question, and inspect retrieval evidence without leaving the app.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_overview(api_client: ApiClient) -> None:
    left, right, third = st.columns(3)

    try:
        health = api_client.health()
        backend_status = "Healthy"
        vector_status = "Ready" if health.get("vector_store_ready") else "Not ready"
        backend_message = health.get("message", "Backend responding")
    except ApiClientError as exc:
        backend_status = "Unavailable"
        vector_status = "Unknown"
        backend_message = str(exc)

    with left:
        st.metric("Backend", backend_status)
        st.caption(backend_message)
    with right:
        st.metric("Vector store", vector_status)
        st.caption("Loaded from the backend service")
    with third:
        st.metric("Session", st.session_state.session_id[:8])
        st.caption("Isolates each browser session")


def _render_upload_panel(api_client: ApiClient) -> None:
    st.subheader("Document Upload")
    st.markdown(
        '<p class="documind-muted">Upload a PDF, image, or spreadsheet. The backend stores it, ingests text, and updates retrieval context.</p>',
        unsafe_allow_html=True,
    )

    upload = st.file_uploader(
        "Choose a document",
        type=["pdf", "png", "jpg", "jpeg", "xlsx", "csv", "docx", "txt"],
        accept_multiple_files=False,
    )

    if not upload:
        return

    if st.button("Upload to backend", type="primary"):
        with st.spinner("Sending document to the ingestion service..."):
            try:
                response = api_client.upload_document(upload)
            except ApiClientError as exc:
                st.error(str(exc))
                return

        st.success(response.get("message", "Document uploaded"))
        cols = st.columns(4)
        cols[0].metric("Chunks", response.get("chunks_created", 0))
        cols[1].metric("Files", response.get("files_processed", 0))
        cols[2].metric("Tables", response.get("tables_extracted", 0))
        cols[3].metric("Images", response.get("images_extracted", 0))

        if response.get("stored_file"):
            st.caption(f"Stored at: {response['stored_file']}")


def _render_diagnostics(api_client: ApiClient) -> None:
    st.subheader("Diagnostics")
    query = st.text_input("Retrieval query", value="What are the key revenue drivers in Q4?")
    diag_col, trace_col = st.columns(2)

    with diag_col:
        if st.button("Load retrieval debug"):
            try:
                debug = api_client.admin_retrieval_debug(query)
            except ApiClientError as exc:
                st.error(str(exc))
            else:
                render_retrieval_debug(debug)
                render_groundedness_indicator(debug.get("groundedness"))

    with trace_col:
        if st.button("Load retrieval trace"):
            try:
                trace = api_client.admin_retrieval_trace(query)
            except ApiClientError as exc:
                st.error(str(exc))
            else:
                st.json(trace)

    st.divider()

    try:
        metrics = api_client.admin_metrics()
    except ApiClientError as exc:
        st.caption(f"Admin metrics unavailable: {exc}")
    else:
        st.markdown("#### Admin Metrics")
        st.json(metrics)

    try:
        state = api_client.admin_debug_state()
    except ApiClientError as exc:
        st.caption(f"Debug state unavailable: {exc}")
    else:
        st.markdown("#### Debug State")
        st.json(state)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📄", layout="wide")
    initialize_session_state()
    _apply_theme()

    with st.sidebar:
        st.markdown("### Control Panel")
        st.session_state.api_base_url = st.text_input("Backend URL", value=st.session_state.api_base_url)
        st.session_state.admin_api_key = st.text_input("Admin API key", value=st.session_state.admin_api_key, type="password")
        st.session_state.streaming_enabled = st.toggle("Typewriter streaming", value=st.session_state.streaming_enabled)

        if st.button("Reset conversation"):
            reset_chat_history()
            st.rerun()

        st.caption("The frontend talks to the FastAPI backend through `/health`, `/documents/upload`, `/chat/ask`, and admin diagnostics endpoints.")

    _render_header()
    st.write("")

    api_client = _api_client()
    _render_overview(api_client)

    chat_tab, documents_tab, diagnostics_tab, sources_tab = st.tabs(["Chat", "Documents", "Diagnostics", "How it works"])

    with chat_tab:
        render_chat(api_client)

    with documents_tab:
        _render_upload_panel(api_client)

    with diagnostics_tab:
        _render_diagnostics(api_client)

    with sources_tab:
        st.subheader("How the app is linked")
        st.markdown(
            """
            - The Streamlit UI calls the FastAPI backend over HTTP.
            - Uploads go to `POST /documents/upload`.
            - Chat requests go to `POST /chat/ask`.
            - Diagnostics read from admin endpoints such as `GET /admin/metrics` and `GET /admin/debug/state`.
            """
        )
        st.code(f"DOCUMIND_API_BASE_URL={st.session_state.api_base_url}", language="text")


if __name__ == "__main__":
    main()