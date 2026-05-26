from __future__ import annotations

import os

import streamlit as st
import streamlit.components.v1 as components

from frontend.services.api_client import ApiClient, ApiClientError
from frontend.utils.session import initialize_session_state


APP_TITLE = "DocuMind AI"
DESIGN_URL = os.getenv("DOCUMIND_DESIGN_URL", "http://127.0.0.1:5173")


def _api_client() -> ApiClient:
    return ApiClient(base_url=st.session_state.api_base_url, session_id=st.session_state.session_id)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📄", layout="wide", initial_sidebar_state="collapsed")
    initialize_session_state()

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"],
        [data-testid="stHeader"],
        footer,
        #MainMenu {
            display: none !important;
        }
        .block-container {
            padding: 0 !important;
            max-width: 100% !important;
        }
        .main .block-container {
            padding-top: 0 !important;
        }
        .stApp {
            background: #08111f;
        }
        .documind-shell {
            margin: 1rem 1rem 0.75rem;
            padding: 1rem 1.15rem;
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(11, 21, 38, 0.95), rgba(8, 17, 31, 0.9));
            border: 1px solid rgba(148, 163, 184, 0.15);
            color: #e5eefb;
        }
        .documind-title {
            font-size: 1.6rem;
            font-weight: 800;
            margin: 0;
        }
        .documind-subtitle {
            margin-top: 0.35rem;
            color: #b7c6df;
        }
        .documind-muted {
            color: #9eb0cb;
            font-size: 0.95rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.8, 1])
    with left:
        st.markdown(
            """
            <div class="documind-shell">
              <div class="documind-title">DocuMind AI</div>
              <div class="documind-subtitle">The new UI/UX design bundle is shown below, and the backend stays linked through Streamlit.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown("<div class='documind-shell'>", unsafe_allow_html=True)
        st.markdown("**Backend**")
        st.session_state.api_base_url = st.text_input("Backend URL", value=st.session_state.api_base_url, label_visibility="collapsed")
        try:
            health = _api_client().health()
            st.success(health.get("message", "Backend healthy"))
            st.caption(f"Vector store ready: {health.get('vector_store_ready', False)}")
        except ApiClientError as exc:
            st.error(str(exc))
        st.markdown("**Design source**")
        st.code(DESIGN_URL, language="text")
        st.caption("Streamlit is now a wrapper around the design bundle, not the old shell.")
        st.markdown("</div>", unsafe_allow_html=True)

    components.iframe(DESIGN_URL, width=1600, height=1200, scrolling=True)


if __name__ == "__main__":
    main()