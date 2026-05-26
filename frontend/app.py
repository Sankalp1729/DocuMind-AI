from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

FRONTEND_DIR = Path(__file__).resolve().parent
if str(FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(FRONTEND_DIR))

from frontend.components.experience import (
    render_admin_cockpit,
    render_authentication_screen,
    render_conversation_history,
    render_landing_screen,
    render_placeholder_screen,
    render_settings,
    render_upload_dashboard,
    render_workspace_screen,
)
from frontend.components.sidebar import render_sidebar
from frontend.components.streaming_chat import render_streaming_chat
from frontend.components.diagnostics import render_diagnostics
from frontend.components.uploader import render_uploader
from frontend.services.api_client import ApiClient
from frontend.services.streaming_client import StreamingClient
from frontend.utils.session import initialize_session_state


st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --documind-ink: #0f172a;
            --documind-slate: #334155;
            --documind-muted: #64748b;
            --documind-border: rgba(148, 163, 184, 0.22);
            --documind-surface: rgba(255, 255, 255, 0.78);
            --documind-accent: #0ea5e9;
            --documind-accent-2: #14b8a6;
            --documind-accent-3: #f59e0b;
            --documind-surface-strong: rgba(255, 255, 255, 0.92);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(14, 165, 233, 0.14), transparent 28%),
                radial-gradient(circle at top right, rgba(20, 184, 166, 0.12), transparent 24%),
                linear-gradient(180deg, rgba(248, 250, 252, 0.96) 0%, rgba(241, 245, 249, 1) 100%);
            color: var(--documind-ink);
        }

        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2.2rem;
            max-width: 1440px;
        }

        .documind-hero {
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(15, 118, 110, 0.84));
            border: 1px solid rgba(255, 255, 255, 0.16);
            border-radius: 28px;
            padding: 1.6rem 1.8rem;
            margin-bottom: 1rem;
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.18);
        }

        .documind-hero h1 {
            margin: 0;
            font-size: 2.15rem;
            line-height: 1.05;
            color: #f8fafc;
        }

        .documind-hero p {
            margin: 0.4rem 0 0;
            color: rgba(226, 232, 240, 0.86);
            max-width: 70ch;
        }

        .documind-card {
            border: 1px solid var(--documind-border);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            background: var(--documind-surface);
            backdrop-filter: blur(10px);
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05);
        }

        .documind-panel {
            border: 1px solid var(--documind-border);
            border-radius: 22px;
            background: var(--documind-surface-strong);
            box-shadow: 0 16px 40px rgba(15, 23, 42, 0.07);
        }

        .documind-section-title {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            margin: 0 0 0.8rem;
            padding: 0.35rem 0.8rem;
            border-radius: 999px;
            background: rgba(14, 165, 233, 0.12);
            color: var(--documind-ink);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .documind-subtitle {
            color: var(--documind-muted);
            margin-top: -0.2rem;
            margin-bottom: 1rem;
        }

        .documind-muted {
            color: var(--documind-muted);
            font-size: 0.92rem;
        }

        .documind-stat-card {
            border: 1px solid var(--documind-border);
            border-radius: 18px;
            padding: 1rem 1rem 0.9rem;
            background: rgba(255, 255, 255, 0.82);
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
        }

        .documind-stat-label {
            color: var(--documind-muted);
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.35rem;
        }

        .documind-stat-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--documind-ink);
            line-height: 1.1;
        }

        .documind-card-caption {
            margin-top: 0.35rem;
            color: var(--documind-muted);
            font-size: 0.85rem;
        }

        .stButton>button {
            border-radius: 999px;
            border: 1px solid rgba(15, 23, 42, 0.08);
            background: linear-gradient(135deg, var(--documind-accent), var(--documind-accent-2));
            color: white;
            font-weight: 600;
            box-shadow: 0 10px 24px rgba(14, 165, 233, 0.18);
        }

        .stTextInput input,
        .stTextArea textarea,
        .stNumberInput input,
        .stSelectbox div[data-baseweb="select"],
        .stMultiSelect div[data-baseweb="select"],
        .stFileUploader {
            border-radius: 16px !important;
        }

        .stExpander {
            border: 1px solid var(--documind-border);
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.66);
            overflow: hidden;
        }

        .stExpander summary {
            font-weight: 700;
        }

        .stChatInput textarea {
            border-radius: 20px !important;
            border: 1px solid rgba(14, 165, 233, 0.18) !important;
            box-shadow: 0 14px 30px rgba(15, 23, 42, 0.06);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            padding: 0.6rem 1rem;
            background: rgba(255, 255, 255, 0.58);
        }

        .stTabs [data-baseweb="tab"] p {
            font-size: 0.96rem;
            font-weight: 700;
        }

        @media (prefers-color-scheme: dark) {
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(14, 165, 233, 0.20), transparent 28%),
                    radial-gradient(circle at top right, rgba(20, 184, 166, 0.18), transparent 24%),
                    linear-gradient(180deg, #020617 0%, #0f172a 100%);
            }

            .documind-hero {
                background: linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(8, 47, 73, 0.92));
                border-color: rgba(148, 163, 184, 0.14);
            }

            .documind-hero p,
            .documind-muted {
                color: rgba(203, 213, 225, 1);
            }

            .documind-stat-card,
            .documind-card,
            .documind-panel {
                background: rgba(15, 23, 42, 0.74);
                border-color: rgba(148, 163, 184, 0.18);
                color: #e2e8f0;
            }

            .documind-stat-value {
                color: #f8fafc;
            }

            .documind-section-title {
                background: rgba(20, 184, 166, 0.14);
                color: #f8fafc;
            }

            .stTabs [data-baseweb="tab"] {
                background: rgba(15, 23, 42, 0.56);
            }

            .stExpander {
                background: rgba(15, 23, 42, 0.72);
                border-color: rgba(148, 163, 184, 0.18);
            }

            .stChatInput textarea {
                background: rgba(15, 23, 42, 0.82);
                color: #f8fafc;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def get_api_client(base_url: str, session_id: str, admin_api_key: str) -> ApiClient:
    return ApiClient(base_url=base_url, session_id=session_id, admin_api_key=admin_api_key)


@st.cache_resource(show_spinner=False)
def get_streaming_client(base_url: str, session_id: str) -> StreamingClient:
    return StreamingClient(api_base_url=base_url, session_id=session_id)


def render_header(api_client: ApiClient) -> None:
    health = st.session_state.backend_health
    vector_status = "Ready" if health and health.get("vector_store_ready") else "Empty"

    st.markdown(
        """
        <div class="documind-hero">
            <h1>DocuMind AI</h1>
            <p>Enterprise-grade document intelligence with persistent retrieval, source citations, telemetry, caching, and control-plane visibility.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Backend", "Online" if health else "Offline")
    col2.metric("Vector Store", vector_status)
    col3.metric("Streaming", "On" if st.session_state.streaming_enabled else "Off")
    col4.metric("Turns", max(len(st.session_state.messages) - 1, 0))

    st.caption("Premium control plane: analytics dashboard, feature flags, billing scaffolding, and retrieval experiments are available through the workspace navigation.")


def render_screen(api_client: ApiClient, streaming_client: StreamingClient) -> None:
    current_screen = st.session_state.current_screen

    if current_screen == "Landing":
        render_landing_screen()
    elif current_screen == "Authentication":
        render_authentication_screen(api_client)
    elif current_screen == "Workspace":
        render_workspace_screen(api_client, streaming_client)
    elif current_screen == "Uploads":
        render_upload_dashboard(api_client)
    elif current_screen == "Conversation":
        render_streaming_chat(streaming_client)
    elif current_screen == "Streaming":
        from frontend.components.experience import render_streaming_screen

        render_streaming_screen()
    elif current_screen == "Citations":
        from frontend.components.experience import render_citations_screen

        render_citations_screen()
    elif current_screen == "Retrieval Trace":
        from frontend.components.experience import render_trace_screen

        render_trace_screen()
    elif current_screen == "Retrieval Analytics":
        from frontend.components.experience import render_analytics_screen

        render_analytics_screen()
    elif current_screen == "Evaluation":
        from frontend.components.experience import render_evaluation_screen

        render_evaluation_screen()
    elif current_screen == "Admin":
        render_admin_cockpit(api_client)
    elif current_screen == "Workspace Settings":
        render_settings()
    elif current_screen == "Conversation History":
        from frontend.components.experience import render_history_screen

        render_history_screen()
    elif current_screen == "Source Preview":
        from frontend.components.experience import render_source_preview_screen

        render_source_preview_screen()
    elif current_screen == "Empty States":
        from frontend.components.experience import render_empty_states_screen

        render_empty_states_screen()
    elif current_screen == "Error States":
        from frontend.components.experience import render_error_states_screen

        render_error_states_screen()
    elif current_screen == "Loading States":
        from frontend.components.experience import render_loading_states_screen

        render_loading_states_screen()
    elif current_screen == "Mobile Layout":
        from frontend.components.experience import render_mobile_screen

        render_mobile_screen()
    elif current_screen == "Tablet Layout":
        from frontend.components.experience import render_tablet_screen

        render_tablet_screen()
    elif current_screen == "Theme System":
        from frontend.components.experience import render_theme_screen

        render_theme_screen()
    else:
        render_workspace_screen(api_client, streaming_client)


def main() -> None:
    initialize_session_state()
    apply_styles()

    api_client = get_api_client(st.session_state.api_base_url, st.session_state.session_id, st.session_state.admin_api_key)
    streaming_client = get_streaming_client(st.session_state.api_base_url, st.session_state.session_id)

    with st.spinner("Checking backend status"):
        if st.session_state.backend_health is None:
            try:
                st.session_state.backend_health = api_client.health()
                st.session_state.last_error = None
            except Exception as exc:
                st.session_state.backend_health = None
                st.session_state.last_error = str(exc)

    render_sidebar(api_client)
    render_header(api_client)

    if st.session_state.last_error and st.session_state.backend_health is None:
        st.warning(f"Backend unavailable: {st.session_state.last_error}")
        st.info("The shell still renders in landing and auth modes. Start the FastAPI backend to unlock live workspace data.")

    render_screen(api_client, streaming_client)


if __name__ == "__main__":
    main()
