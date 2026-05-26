from __future__ import annotations

import time

import streamlit as st

from frontend.components.citations import render_citation_cards
from frontend.services.api_client import ApiClient, ApiClientError


def _stream_answer(answer: str, placeholder, enabled: bool) -> None:
    if not enabled:
        placeholder.markdown(answer)
        return

    rendered = []
    for token in answer.split():
        rendered.append(token)
        placeholder.markdown(" ".join(rendered) + "▌")
        time.sleep(0.012)

    placeholder.markdown(" ".join(rendered))


def _render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            citations = message.get("sources", [])
            stats = message.get("stats", {})
            if stats:
                st.caption(
                    f"Latency: {stats.get('latency_ms', 0)} ms | "
                    f"Sources: {stats.get('source_count', 0)} | "
                    f"Approx. tokens: {stats.get('token_estimate', 0)}"
                )
            render_citation_cards(citations)


def render_chat(api_client: ApiClient) -> None:
    st.subheader("Conversation")
    st.markdown(
        '<p class="documind-muted">Ask questions about the uploaded PDFs. Answers are grounded in retrieved passages and rendered with citations.</p>',
        unsafe_allow_html=True,
    )

    for message in st.session_state.messages:
        _render_message(message)

    prompt = st.chat_input("Ask a question about your documents")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        with st.spinner("Retrieving context and generating the answer..."):
            start = time.perf_counter()
            try:
                response = api_client.ask_question(prompt)
            except ApiClientError as exc:
                st.error(str(exc))
                st.session_state.last_error = str(exc)
                return

            latency_ms = int((time.perf_counter() - start) * 1000)
            answer = response.get("answer", "")
            citations = response.get("sources", [])

            _stream_answer(answer, placeholder, st.session_state.streaming_enabled)

        stats = {
            "latency_ms": latency_ms,
            "source_count": len(citations),
            "token_estimate": len(answer.split()),
        }

        st.caption(
            f"Latency: {latency_ms} ms | Sources: {len(citations)} | Approx. tokens: {len(answer.split())}"
        )
        render_citation_cards(citations)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": citations,
            "stats": stats,
        }
    )