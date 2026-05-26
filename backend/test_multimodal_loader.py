from __future__ import annotations

from langchain_core.documents import Document

from backend.rag.multimodal_loader import ExtractionStats, load_multimodal_documents, load_supported_documents


def test_load_multimodal_documents_dispatches_supported_files(tmp_path, monkeypatch) -> None:
    pdf_path = tmp_path / "contract.pdf"
    pdf_path.write_bytes(b"pdf")
    image_path = tmp_path / "screenshot.png"
    image_path.write_bytes(b"image")
    ignored_path = tmp_path / "notes.txt"
    ignored_path.write_text("ignored")

    def fake_pdf_extractor(file_path, engine, languages, stats):
        stats.documents_created += 1
        return [Document(page_content=f"PDF:{file_path.name}", metadata={"source_file": file_path.name})]

    def fake_image_extractor(file_path, engine, languages, stats):
        stats.documents_created += 1
        return [Document(page_content=f"IMG:{file_path.name}", metadata={"source_file": file_path.name})]

    monkeypatch.setattr("backend.rag.multimodal_loader._extract_pdf_documents", fake_pdf_extractor)
    monkeypatch.setattr("backend.rag.multimodal_loader._extract_image_document", fake_image_extractor)

    loaded = load_multimodal_documents(tmp_path)

    assert [document.page_content for document in loaded.documents] == ["PDF:contract.pdf", "IMG:screenshot.png"]
    assert loaded.stats.files_processed == 2
    assert loaded.stats.documents_created == 2


def test_load_supported_documents_returns_plain_list(tmp_path, monkeypatch) -> None:
    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(b"pdf")

    def fake_pdf_extractor(file_path, engine, languages, stats):
        return [Document(page_content="scan text", metadata={"source_file": file_path.name})]

    monkeypatch.setattr("backend.rag.multimodal_loader._extract_pdf_documents", fake_pdf_extractor)
    monkeypatch.setattr("backend.rag.multimodal_loader._extract_image_document", lambda *args, **kwargs: [])

    documents = load_supported_documents(tmp_path)

    assert len(documents) == 1
    assert documents[0].metadata["source_file"] == "scan.pdf"


def test_extraction_stats_tracks_average_confidence() -> None:
    stats = ExtractionStats()
    stats.record_ocr_confidence(80.0)
    stats.record_ocr_confidence(60.0)

    assert stats.average_ocr_confidence == 70.0
