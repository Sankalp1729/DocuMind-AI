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
    st.set_page_config(page_title=APP_TITLE, page_icon="📄", layout="wide")
    initialize_session_state()

    st.markdown(
        """
        <style>
        .documind-shell {
            margin-bottom: 1rem;
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

    with st.sidebar:
        st.markdown("### Backend Link")
        st.session_state.api_base_url = st.text_input("Backend URL", value=st.session_state.api_base_url)
        try:
            health = _api_client().health()
            st.success(health.get("message", "Backend healthy"))
            st.caption(f"Vector store ready: {health.get('vector_store_ready', False)}")
        except ApiClientError as exc:
            st.error(str(exc))

        st.markdown("### Design Source")
        st.code(DESIGN_URL, language="text")
        st.caption("This Streamlit page now embeds the new design bundle instead of the older custom shell.")

    st.markdown(
        """
        <div class="documind-shell">
          <div class="documind-title">DocuMind AI</div>
          <div class="documind-subtitle">The new UI/UX design bundle is shown below, and the backend stays linked through the Streamlit sidebar.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    components.iframe(DESIGN_URL, height=1200, scrolling=True)


if __name__ == "__main__":
    main()