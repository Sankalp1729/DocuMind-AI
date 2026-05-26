from __future__ import annotations

import streamlit as st

from frontend.services.api_client import ApiClient, ApiClientError


def render_uploader(api_client: ApiClient) -> None:
    st.markdown('<div class="documind-section-title">Document Library</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="documind-muted">Upload PDFs, scanned contracts, screenshots, or image files to rebuild the retrieval index.</p>',
        unsafe_allow_html=True,
    )

    files = st.file_uploader(
        "Choose documents",
        type=["pdf", "png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
        help="Uploads are sent to the FastAPI backend and re-indexed into FAISS with OCR when needed.",
    )

    if not files:
        st.info("Select one or more documents to begin indexing. Supported file types are shown directly in the uploader.")
        return

    preview_cols = st.columns(2)
    preview_cols[0].metric("Selected files", len(files))
    preview_cols[1].metric("Total size", f"{sum(file.size for file in files):,} bytes")

    st.markdown("#### Files queued for upload")
    for uploaded_file in files:
        st.markdown(f"- **{uploaded_file.name}** · {uploaded_file.size:,} bytes")

    if not st.button("Upload and index documents", use_container_width=True):
        return

    progress = st.progress(0)
    status = st.empty()
    results = []

    try:
        for index, uploaded_file in enumerate(files, start=1):
            status.info(f"Uploading {uploaded_file.name}...")
            result = api_client.upload_document(uploaded_file)
            results.append(result)
            st.session_state.upload_history.insert(0, result)
            progress.progress(index / len(files))

        status.success("All documents uploaded and indexed successfully.")
        st.session_state.backend_health = api_client.health()
        st.session_state.last_error = None
        st.success(f"Indexed {len(results)} file(s) into the persistent vector store.")
        st.caption("If the index is still empty above, refresh the backend status from the sidebar to update the health snapshot.")
        for item in results:
            st.markdown(
                f"- **{item.get('message')}** · chunks: {item.get('chunks_created')} · stored: {item.get('stored_file')} · ocr confidence: {item.get('average_ocr_confidence') or 'n/a'}"
            )
    except ApiClientError as exc:
        status.error(str(exc))
        st.session_state.last_error = str(exc)
        st.error("Document upload failed. Check the backend logs and retry.")
    finally:
        progress.empty()
