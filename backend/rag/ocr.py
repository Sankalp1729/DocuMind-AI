from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from statistics import mean

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

logger = logging.getLogger(__name__)

try:
    import pytesseract
    from pytesseract import Output
except ImportError:  # pragma: no cover - optional dependency
    pytesseract = None
    Output = None

try:
    from paddleocr import PaddleOCR
except ImportError:  # pragma: no cover - optional dependency
    PaddleOCR = None


@dataclass(slots=True)
class OCRResult:
    text: str
    confidence: float | None
    engine: str


def _normalize_engine_name(engine_name: str | None) -> str:
    value = (engine_name or os.getenv("DOCUMIND_OCR_ENGINE", "tesseract")).strip().lower()
    if value in {"auto", "default"}:
        return "tesseract"
    return value


def preprocess_image_for_ocr(image: Image.Image) -> Image.Image:
    normalized = ImageOps.exif_transpose(image).convert("RGB")
    grayscale = ImageOps.grayscale(normalized)
    contrast = ImageEnhance.Contrast(grayscale).enhance(1.8)
    sharpened = contrast.filter(ImageFilter.SHARPEN)
    return ImageOps.autocontrast(sharpened)


def _set_tesseract_command() -> None:
    command = os.getenv("DOCUMIND_TESSERACT_CMD")
    if command and pytesseract is not None:
        pytesseract.pytesseract.tesseract_cmd = command


def _parse_tesseract_confidence(data: dict[str, list[str]]) -> float | None:
    confidences: list[float] = []

    for raw_conf in data.get("conf", []):
        try:
            confidence = float(raw_conf)
        except (TypeError, ValueError):
            continue
        if confidence >= 0:
            confidences.append(confidence)

    if not confidences:
        return None

    return round(mean(confidences), 2)


def _ocr_with_tesseract(image: Image.Image, languages: str | None = None) -> OCRResult:
    if pytesseract is None or Output is None:
        raise RuntimeError("pytesseract is not installed")

    _set_tesseract_command()
    lang_value = languages or os.getenv("DOCUMIND_OCR_LANGS", "eng")
    config = os.getenv("DOCUMIND_TESSERACT_CONFIG", "--oem 1 --psm 6")

    data = pytesseract.image_to_data(image, lang=lang_value, config=config, output_type=Output.DICT)
    text = pytesseract.image_to_string(image, lang=lang_value, config=config).strip()
    confidence = _parse_tesseract_confidence(data)

    return OCRResult(text=text, confidence=confidence, engine="tesseract")


class _PaddleOCRBackend:
    def __init__(self, languages: str | None = None) -> None:
        if PaddleOCR is None:
            raise RuntimeError("paddleocr is not installed")

        lang_value = (languages or os.getenv("DOCUMIND_OCR_LANGS", "en")).split(",")[0].strip() or "en"
        self.reader = PaddleOCR(use_angle_cls=True, lang=lang_value, show_log=False)

    def recognize(self, image: Image.Image) -> OCRResult:
        result = self.reader.ocr(np.array(image), cls=True)
        lines: list[str] = []
        confidences: list[float] = []

        for page_result in result or []:
            for entry in page_result or []:
                if len(entry) < 2:
                    continue
                text_block = entry[1]
                if isinstance(text_block, (list, tuple)) and text_block:
                    text = str(text_block[0]).strip()
                    if text:
                        lines.append(text)
                    if len(text_block) > 1:
                        try:
                            confidences.append(float(text_block[1]) * 100)
                        except (TypeError, ValueError):
                            continue

        confidence = round(mean(confidences), 2) if confidences else None
        return OCRResult(text="\n".join(lines).strip(), confidence=confidence, engine="paddleocr")


def ocr_image(image: Image.Image, engine: str | None = None, languages: str | None = None) -> OCRResult:
    normalized_engine = _normalize_engine_name(engine)
    prepared_image = preprocess_image_for_ocr(image)

    if normalized_engine == "paddleocr":
        try:
            return _PaddleOCRBackend(languages=languages).recognize(prepared_image)
        except Exception as exc:  # pragma: no cover - optional fallback
            logger.warning("PaddleOCR failed, falling back to Tesseract: %s", exc)

    if pytesseract is None:
        raise RuntimeError("No OCR backend is available. Install pytesseract or paddleocr.")

    return _ocr_with_tesseract(prepared_image, languages=languages)
