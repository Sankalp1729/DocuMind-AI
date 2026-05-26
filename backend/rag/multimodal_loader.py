from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document

from backend.rag.ocr import ocr_image

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
OCR_TEXT_THRESHOLD = 40

try:
    import fitz
except ImportError:  # pragma: no cover - optional dependency
    fitz = None

try:
    import pdfplumber
except ImportError:  # pragma: no cover - optional dependency
    pdfplumber = None

from PIL import Image


@dataclass(slots=True)
class ExtractionStats:
    files_processed: int = 0
    documents_created: int = 0
    pages_processed: int = 0
    ocr_pages: int = 0
    images_extracted: int = 0
    tables_extracted: int = 0
    ocr_confidence_sum: float = 0.0
    ocr_confidence_count: int = 0

    def record_ocr_confidence(self, confidence: float | None) -> None:
        if confidence is None:
            return
        self.ocr_confidence_sum += confidence
        self.ocr_confidence_count += 1

    @property
    def average_ocr_confidence(self) -> float | None:
        if not self.ocr_confidence_count:
            return None
        return round(self.ocr_confidence_sum / self.ocr_confidence_count, 2)


@dataclass(slots=True)
class LoadedDocuments:
    documents: list[Document] = field(default_factory=list)
    stats: ExtractionStats = field(default_factory=ExtractionStats)


def _supported_file(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_SUFFIXES


def _base_metadata(source_file: Path, **kwargs) -> dict[str, object]:
    metadata: dict[str, object] = {
        "source_file": source_file.name,
        "source_path": str(source_file),
    }
    metadata.update(kwargs)
    return metadata


def _document_from_text(text: str, source_file: Path, **metadata: object) -> Document | None:
    cleaned_text = text.strip()
    if not cleaned_text:
        return None
    return Document(page_content=cleaned_text, metadata=_base_metadata(source_file, **metadata))


def _render_pdf_page(page) -> Image.Image:
    pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    mode = "RGB" if pixmap.n < 4 else "RGBA"
    image = Image.frombytes(mode, [pixmap.width, pixmap.height], pixmap.samples)
    return image.convert("RGB")


def _extract_tables_from_page(pdf_path: Path, page, page_number: int) -> list[Document]:
    if pdfplumber is None:
        return []

    table_documents: list[Document] = []
    try:
        tables = page.extract_tables() or []
        for table_index, table in enumerate(tables, start=1):
            rows = ["\t".join((cell or "").strip() for cell in row) for row in table if row]
            table_text = "\n".join(row for row in rows if row).strip()
            if not table_text:
                continue
            table_documents.append(
                Document(
                    page_content=f"Table {table_index} on page {page_number}\n{table_text}",
                    metadata=_base_metadata(
                        pdf_path,
                        source_type="pdf_table",
                        page=page_number,
                        table_index=table_index,
                        extraction_method="pdfplumber",
                    ),
                )
            )
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("Table extraction failed for %s: %s", pdf_path.name, exc)

    return table_documents


def _extract_embedded_images(pdf_path: Path, page, page_number: int, engine: str | None, languages: str | None) -> list[Document]:
    image_documents: list[Document] = []
    images = page.get_images(full=True)

    for image_index, image_info in enumerate(images, start=1):
        xref = image_info[0]
        try:
            extracted = fitz.Pixmap(page.parent, xref)
            if extracted.n > 4:
                extracted = fitz.Pixmap(fitz.csRGB, extracted)
            image = Image.frombytes("RGB", [extracted.width, extracted.height], extracted.samples)
            ocr_result = ocr_image(image, engine=engine, languages=languages)
            if ocr_result.text:
                image_documents.append(
                    Document(
                        page_content=ocr_result.text,
                        metadata=_base_metadata(
                            pdf_path,
                            source_type="embedded_image",
                            page=page_number,
                            image_index=image_index,
                            extraction_method=ocr_result.engine,
                            ocr_confidence=ocr_result.confidence,
                        ),
                    )
                )
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("Failed to extract embedded image %s from %s: %s", image_index, pdf_path.name, exc)

    return image_documents


def _extract_pdf_documents(pdf_path: Path, engine: str | None, languages: str | None, stats: ExtractionStats) -> list[Document]:
    documents: list[Document] = []

    if fitz is None:
        raise RuntimeError("PyMuPDF is required for PDF extraction. Install pymupdf.")

    with fitz.open(str(pdf_path)) as pdf_document:
        for page_index, page in enumerate(pdf_document, start=1):
            stats.pages_processed += 1

            page_text = page.get_text("text") or ""
            page_metadata = {
                "source_type": "pdf_page",
                "page": page_index,
                "extraction_method": "native_text" if page_text.strip() else "ocr",
            }

            native_document = _document_from_text(page_text, pdf_path, **page_metadata)
            if native_document is not None:
                documents.append(native_document)
                stats.documents_created += 1

            if len(page_text.strip()) < OCR_TEXT_THRESHOLD:
                ocr_result = ocr_image(_render_pdf_page(page), engine=engine, languages=languages)
                stats.ocr_pages += 1
                stats.record_ocr_confidence(ocr_result.confidence)
                if ocr_result.text:
                    documents.append(
                        Document(
                            page_content=ocr_result.text,
                            metadata=_base_metadata(
                                pdf_path,
                                source_type="pdf_page_ocr",
                                page=page_index,
                                extraction_method=ocr_result.engine,
                                ocr_confidence=ocr_result.confidence,
                            ),
                        )
                    )
                    stats.documents_created += 1

            table_documents = _extract_tables_from_page(pdf_path, page, page_index)
            if table_documents:
                documents.extend(table_documents)
                stats.tables_extracted += len(table_documents)
                stats.documents_created += len(table_documents)

            embedded_images = _extract_embedded_images(pdf_path, page, page_index, engine, languages)
            if embedded_images:
                documents.extend(embedded_images)
                stats.images_extracted += len(embedded_images)
                stats.documents_created += len(embedded_images)

    return documents


def _extract_image_document(image_path: Path, engine: str | None, languages: str | None, stats: ExtractionStats) -> list[Document]:
    with Image.open(image_path) as image:
        ocr_result = ocr_image(image, engine=engine, languages=languages)

    stats.record_ocr_confidence(ocr_result.confidence)
    if not ocr_result.text:
        return []

    stats.documents_created += 1
    stats.ocr_pages += 1

    return [
        Document(
            page_content=ocr_result.text,
            metadata=_base_metadata(
                image_path,
                source_type="image",
                extraction_method=ocr_result.engine,
                ocr_confidence=ocr_result.confidence,
            ),
        )
    ]


def load_multimodal_documents(data_folder: str | Path | None = None, engine: str | None = None, languages: str | None = None) -> LoadedDocuments:
    folder = Path(data_folder) if data_folder is not None else Path(__file__).resolve().parents[2] / "data"
    loaded = LoadedDocuments()

    if not folder.exists():
        return loaded

    for file_path in sorted(path for path in folder.rglob("*") if path.is_file() and _supported_file(path)):
        loaded.stats.files_processed += 1

        try:
            if file_path.suffix.lower() == ".pdf":
                loaded.documents.extend(_extract_pdf_documents(file_path, engine, languages, loaded.stats))
            else:
                loaded.documents.extend(_extract_image_document(file_path, engine, languages, loaded.stats))
        except Exception as exc:
            logger.warning("Skipping %s because extraction failed: %s", file_path.name, exc)

    return loaded


def load_supported_documents(data_folder: str | Path | None = None, engine: str | None = None, languages: str | None = None) -> list[Document]:
    return load_multimodal_documents(data_folder=data_folder, engine=engine, languages=languages).documents
