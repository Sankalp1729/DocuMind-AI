from __future__ import annotations

import streamlit as st

from frontend.components.citations import render_citation_cards
from frontend.components.retrieval_debug import render_retrieval_debug, render_groundedness_indicator
from frontend.services.streaming_client import StreamingClient, StreamingClientError


def render_streaming_chat(streaming_client: StreamingClient) -> None:
    st.subheader("Conversation")
    st.markdown(
        '<p class="documind-muted">Questions are streamed from the backend as tokens arrive from Ollama.</p>',
        unsafe_allow_html=True,
    )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_citation_cards(
                    message.get("sources", []),
                    retrieval_explanation=message.get("retrieval_explanation"),
                )
                render_groundedness_indicator(message.get("groundedness"))

    prompt = st.chat_input("Ask a question about your documents")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        accumulated = ""
        citations = []
        retrieval_explanation = None
        groundedness = None
        latency_ms = None

        try:
            with st.spinner("Streaming response from Llama 3..."):
                for event in streaming_client.stream_question(prompt):
                    payload = event.payload
                    if event.event == "start":
                        placeholder.markdown("▌")
                    elif event.event == "token":
                        accumulated += payload.get("token", "")
                        placeholder.markdown(accumulated + "▌")
                    elif event.event == "complete":
                        accumulated = payload.get("answer", accumulated)
                        citations = payload.get("sources", [])
                        retrieval_explanation = payload.get("retrieval_explanation")
                        groundedness = payload.get("groundedness")
                        latency_ms = payload.get("latency_ms")
                        placeholder.markdown(accumulated)
                    elif event.event == "error":
                        raise StreamingClientError(payload.get("error", "Streaming failed"))
        except StreamingClientError as exc:
            st.error(str(exc))
            st.session_state.last_error = str(exc)
            return

        if latency_ms is not None:
            st.caption(
                f"Latency: {latency_ms} ms | Sources: {len(citations)} | Approx. tokens: {len(accumulated.split())}"
            )
        
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
            "stats": {
                "latency_ms": latency_ms or 0,
                "source_count": len(citations),
                "token_estimate": len(accumulated.split()),
            },
        }
    )