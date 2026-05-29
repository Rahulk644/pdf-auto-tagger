"""Stage 1 — Scanned-page text extraction via PP-OCRv4 (CPU, MIT-clean).

Closes the one honest scope boundary of the CPU layout backend
([[project-cpu-layout-backend]]): pdfplumber returns nothing on image-only pages
because there is no text layer, so without OCR the pipeline produced empty
extraction for scanned PDFs (the historical MinerU OCR path was never actually
implemented — this module was a stub).

Approach: rapidocr-onnxruntime (Apache-2.0) ships PP-OCRv4 weights as ONNX and
runs them via onnxruntime — same models the user's intel pointed at, without the
PaddlePaddle dependency that would balloon the install. RapidOCR's `__call__`
already does detection + direction-classification + recognition in one pass on
the page image, so for a v1 we don't pre-crop by Heron regions; the Stage-3 CPU
layout detector then groups the extracted lines like any other PageElement list.

Self-gating: missing rapidocr-onnxruntime -> log + return empty per scanned page
(the pipeline still ships a /UA-1 valid output, just with no text on those pages).

Coordinates: rendering at STANDARD_DPI means image pixel coords == the 150-DPI
standard space all downstream stages use — no transform needed.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from tagger.config import STANDARD_DPI
from tagger.models.data_types import PageClassification, PageElement, PageType

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

_ocr = None
_load_failed = False


def _load_ocr():
    """Lazy, one-time RapidOCR singleton load."""
    global _ocr, _load_failed
    if _ocr is not None or _load_failed:
        return _ocr is not None
    try:
        from rapidocr_onnxruntime import RapidOCR
        _ocr = RapidOCR()
        logger.info("RapidOCR (PP-OCRv4 ONNX) loaded")
        return True
    except Exception as e:
        _load_failed = True
        logger.warning("rapidocr-onnxruntime unavailable (%s) — scanned pages "
                       "will yield no text", e)
        return False


def _polygon_to_bbox(box) -> tuple[float, float, float, float]:
    """RapidOCR's per-line box is a 4-corner polygon; downstream wants (x0,y0,x1,y1)."""
    xs = [float(p[0]) for p in box]
    ys = [float(p[1]) for p in box]
    return (min(xs), min(ys), max(xs), max(ys))


def extract_scanned_pages(
    pdf_path: "str | Path",
    classifications: list[PageClassification],
) -> dict[int, list[PageElement]]:
    """OCR text on scanned (and mixed) pages — see module docstring."""
    scanned_pages = sorted(
        c.page_num for c in classifications
        if c.page_type in (PageType.SCANNED, PageType.MIXED)
    )
    if not scanned_pages:
        return {}
    if not _load_ocr():
        return {p: [] for p in scanned_pages}

    try:
        import fitz
        import numpy as np
        from PIL import Image
    except ImportError as e:
        logger.warning("Scanned extraction needs fitz/numpy/PIL (%s)", e)
        return {p: [] for p in scanned_pages}

    out: dict[int, list[PageElement]] = {}
    with fitz.open(str(pdf_path)) as doc:
        for page_num in scanned_pages:
            idx = page_num - 1
            if idx >= len(doc):
                continue
            pix = doc[idx].get_pixmap(dpi=STANDARD_DPI)
            img = np.array(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
            try:
                result, _elapse = _ocr(img)
            except Exception as e:
                logger.warning("OCR failed on page %d: %s", page_num, e)
                out[page_num] = []
                continue

            elements: list[PageElement] = []
            for i, item in enumerate(result or []):
                if not item or len(item) < 3:
                    continue
                box, text, conf_str = item[0], item[1], item[2]
                if not text or not text.strip():
                    continue
                try:
                    conf = float(conf_str)
                except (TypeError, ValueError):
                    conf = 0.0
                bbox = _polygon_to_bbox(box)
                elements.append(PageElement(
                    element_id=f"p{page_num}_o{i}",
                    page_num=page_num,
                    text=text,
                    bbox=bbox,
                    source="rapidocr",
                    confidence=conf,
                ))
            out[page_num] = elements
            logger.info("OCR page %d: %d text lines", page_num, len(elements))

    return out
