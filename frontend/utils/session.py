from __future__ import annotations

import os
from uuid import uuid4

import streamlit as st


def initialize_session_state() -> None:
    defaults = {
        "messages": [
            {
                "role": "assistant",
                "content": "Upload PDFs in the Documents tab and ask questions here.",
                "sources": [],
                "stats": {},
            }
        ],
        "backend_health": None,
        "last_error": None,
        "streaming_enabled": True,
        "upload_history": [],
        "api_base_url": os.getenv("DOCUMIND_API_BASE_URL", "http://localhost:8000"),
        "admin_api_key": os.getenv("DOCUMIND_ADMIN_API_KEY", ""),
        "session_id": str(uuid4()),
        "current_screen": "Landing",
        "is_authenticated": False,
        "theme_mode": "Dark",
    }

    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def reset_chat_history() -> None:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Upload PDFs in the Documents tab and ask questions here.",
            "sources": [],
            "stats": {},
        }
    ]
