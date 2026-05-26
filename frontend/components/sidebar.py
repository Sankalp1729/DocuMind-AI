from __future__ import annotations

import streamlit as st

from frontend.services.api_client import ApiClient, ApiClientError
from frontend.components.experience import SCREEN_LIBRARY
from frontend.utils.session import reset_chat_history


def render_sidebar(api_client: ApiClient) -> None:
    with st.sidebar:
        st.markdown(
            """
            <div class="documind-panel" style="padding: 1rem;">
                <div class="documind-section-title">Control Center</div>
                <div style="font-size: 1.4rem; font-weight: 800; color: var(--documind-ink); line-height: 1.1;">DocuMind AI</div>
                <div class="documind-subtitle">FastAPI + LangChain + FAISS + Ollama</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="documind-section-title">Connection</div>', unsafe_allow_html=True)
        st.text_input("Backend URL", key="api_base_url", help="Point this to the running FastAPI backend.")
        st.text_input("Admin API Key", key="admin_api_key", type="password", help="Unlock retrieval traces, benchmarks, and debug endpoints.")
        st.caption(f"Session ID: {st.session_state.session_id}")

        st.markdown('<div class="documind-section-title">Navigation</div>', unsafe_allow_html=True)
        screen_names = [screen for screen, _ in SCREEN_LIBRARY]
        st.session_state.current_screen = st.radio(
            "Select screen",
            screen_names,
            index=screen_names.index(st.session_state.current_screen) if st.session_state.current_screen in screen_names else 0,
            label_visibility="collapsed",
        )

        health = st.session_state.backend_health or {}
        backend_online = bool(health)
        vector_ready = bool(health.get("vector_store_ready"))

        st.markdown('<div class="documind-section-title">Status Snapshot</div>', unsafe_allow_html=True)
        st.metric("Backend", "Online" if backend_online else "Offline")
        st.metric("Vector store", "Ready" if vector_ready else "Empty")
        st.metric("Chat turns", max(len(st.session_state.messages) - 1, 0))

        if st.button("Refresh backend status", use_container_width=True):
            try:
                st.session_state.backend_health = api_client.health()
                st.session_state.last_error = None
                st.success("Backend is reachable.")
            except ApiClientError as exc:
                st.session_state.backend_health = None
                st.session_state.last_error = str(exc)
                st.error(str(exc))

        if st.session_state.admin_api_key:
            st.success("Admin diagnostics enabled")
        else:
            st.info("Enter an admin API key to unlock benchmarks and retrieval traces.")

        st.markdown('<div class="documind-section-title">Controls</div>', unsafe_allow_html=True)
        st.toggle("Simulated streaming", key="streaming_enabled", help="Preview token streaming even when the backend is idle.")
        st.selectbox("Theme mode", ["Light", "Dark"], key="theme_mode", help="Visual theme token for the workspace shell.")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Clear conversation", use_container_width=True):
                reset_chat_history()
                st.rerun()
        with col_b:
            st.button("Copy session", use_container_width=True, disabled=True, help="Reserved for a future export action.")

        st.markdown('<div class="documind-section-title">Model</div>', unsafe_allow_html=True)
        st.info("Ollama / Llama 3")

        if st.session_state.upload_history:
            st.markdown('<div class="documind-section-title">Recent uploads</div>', unsafe_allow_html=True)
            for item in st.session_state.upload_history[:3]:
                st.caption(f"{item.get('message')}")
