"""
Stage 0 — Page-level classifier.

Classifies each page in a PDF as native, scanned, mixed, or corrupt
using pure pdfplumber heuristics.  No ML models — this is free and fast.

Signals used:
  - Character count from pdfplumber
  - Unicode validity of extracted characters
  - Image XObject coverage (fraction of page area)
  - Character density (chars per square inch)

This classification drives the Stage 1 extraction path:
  - native  → pdfplumber char extraction
  - scanned → MinerU2.5 OCR
  - mixed   → both paths, merged
  - corrupt → skip or flag for manual handling
"""

from __future__ import annotations

import logging
import unicodedata
from typing import TYPE_CHECKING

import pdfplumber

from tagger.config import PAGE_CLASSIFIER, PDF_NATIVE_DPI
from tagger.models.data_types import PageClassification, PageType

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def classify_pages(pdf_path: str | Path) -> list[PageClassification]:
    """
    Classify every page in a PDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of PageClassification, one per page, in page order.
    """
    results: list[PageClassification] = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            page_num = page_idx + 1
            try:
                classification = _classify_single_page(page, page_num)
            except Exception as e:
                logger.warning(
                    "Page %d: classification failed (%s), marking as corrupt",
                    page_num, e,
                )
                classification = PageClassification(
                    page_num=page_num,
                    page_type=PageType.CORRUPT,
                    char_count=0,
                    image_coverage=0.0,
                    unicode_validity=0.0,
                    char_density=0.0,
                    confidence=0.9,
                    page_width_pt=float(page.width),
                    page_height_pt=float(page.height),
                )
            results.append(classification)

    logger.info(
        "Classified %d pages: %s",
        len(results),
        {pt.value: sum(1 for r in results if r.page_type == pt) for pt in PageType},
    )
    return results


def _classify_single_page(
    page: pdfplumber.page.Page,
    page_num: int,
) -> PageClassification:
    """Classify a single pdfplumber page."""
    cfg = PAGE_CLASSIFIER

    page_width_pt = float(page.width)
    page_height_pt = float(page.height)

    # -- Extract characters --------------------------------------------------
    chars = page.chars or []
    char_count = len(chars)

    # -- Unicode validity -----------------------------------------------------
    unicode_validity = _compute_unicode_validity(chars)

    # -- Image coverage -------------------------------------------------------
    image_coverage = _compute_image_coverage(page, page_width_pt, page_height_pt)

    # -- Character density (chars per square inch) ----------------------------
    page_area_sq_in = (page_width_pt / PDF_NATIVE_DPI) * (page_height_pt / PDF_NATIVE_DPI)
    char_density = char_count / page_area_sq_in if page_area_sq_in > 0 else 0.0

    # -- Decision logic -------------------------------------------------------
    page_type, confidence = _decide_page_type(
        char_count=char_count,
        unicode_validity=unicode_validity,
        image_coverage=image_coverage,
        char_density=char_density,
    )

    logger.debug(
        "Page %d: type=%s conf=%.2f chars=%d img_cov=%.2f uni_val=%.2f density=%.4f",
        page_num, page_type.value, confidence, char_count,
        image_coverage, unicode_validity, char_density,
    )

    return PageClassification(
        page_num=page_num,
        page_type=page_type,
        char_count=char_count,
        image_coverage=image_coverage,
        unicode_validity=unicode_validity,
        char_density=char_density,
        confidence=confidence,
        page_width_pt=page_width_pt,
        page_height_pt=page_height_pt,
    )


def _compute_unicode_validity(chars: list[dict]) -> float:
    """
    Fraction of characters that are valid, printable Unicode.

    Catches garbled OCR layers, CMAP mapping failures, and corrupted
    content streams that produce replacement characters or control codes.
    """
    if not chars:
        return 1.0  # No chars → nothing to validate

    valid_count = 0
    for ch in chars:
        text = ch.get("text", "")
        if not text:
            continue
        # Check each codepoint
        for cp in text:
            cat = unicodedata.category(cp)
            # Valid: letters, numbers, punctuation, symbols, separators, marks
            # Invalid: Cc (control), Cs (surrogate), Co (private use area, sometimes OK),
            #          Cn (unassigned) — except common whitespace
            if cat.startswith(("L", "N", "P", "S", "Z", "M")):
                valid_count += 1
            elif cp in ("\n", "\r", "\t", " "):
                valid_count += 1
            # else: invalid codepoint, don't count

    total_codepoints = sum(len(ch.get("text", "")) for ch in chars)
    if total_codepoints == 0:
        return 1.0

    return valid_count / total_codepoints


def _compute_image_coverage(
    page: pdfplumber.page.Page,
    page_width_pt: float,
    page_height_pt: float,
) -> float:
    """
    Fraction of the page area covered by image XObjects.

    Uses pdfplumber's `.images` which gives bounding boxes for each
    embedded image.  For scanned PDFs, this will typically be ~1.0
    (one large image covering the entire page).
    """
    page_area = page_width_pt * page_height_pt
    if page_area <= 0:
        return 0.0

    images = page.images or []
    if not images:
        return 0.0

    # Sum image areas (may overlap — this is an approximation)
    total_image_area = 0.0
    for img in images:
        x0 = float(img.get("x0", 0))
        y0 = float(img.get("top", 0))
        x1 = float(img.get("x1", 0))
        y1 = float(img.get("bottom", 0))
        w = max(0, x1 - x0)
        h = max(0, y1 - y0)
        total_image_area += w * h

    # Cap at 1.0 (overlapping images can exceed page area)
    return min(1.0, total_image_area / page_area)


def _decide_page_type(
    char_count: int,
    unicode_validity: float,
    image_coverage: float,
    char_density: float,
) -> tuple[PageType, float]:
    """
    Apply classification heuristics and return (PageType, confidence).

    Decision tree:
    1. If Unicode validity is very low → CORRUPT
    2. If no chars AND high image coverage → SCANNED
    3. If no chars AND low image coverage → CORRUPT (blank page)
    4. If many chars AND low image coverage → NATIVE
    5. If some chars AND high image coverage → MIXED
    6. Edge cases → MIXED with lower confidence
    """
    cfg = PAGE_CLASSIFIER

    # 1. Corruption check
    if char_count > 0 and unicode_validity < cfg.min_unicode_validity:
        return PageType.CORRUPT, 0.85

    # 2. Pure scanned (no text, mostly image)
    if char_count == 0 and image_coverage >= cfg.scanned_image_coverage:
        return PageType.SCANNED, 0.95

    # 3. Blank / broken page (no text, no image)
    if char_count == 0 and image_coverage < cfg.scanned_image_coverage:
        # Could be a blank page or a page with only vector graphics
        if image_coverage < 0.05:
            return PageType.CORRUPT, 0.70  # Truly blank
        return PageType.SCANNED, 0.60  # Some vector graphics, treat as scan

    # 4. Pure native (lots of text, minimal images)
    if char_count >= cfg.min_native_chars and image_coverage < cfg.native_image_coverage:
        return PageType.NATIVE, 0.95

    # 5. Native with images (lots of text + significant images)
    if char_count >= cfg.min_native_chars and image_coverage >= cfg.native_image_coverage:
        if image_coverage >= cfg.scanned_image_coverage:
            # High image + text → could be an OCR'd scan with text layer
            return PageType.MIXED, 0.80
        # Sparse-text override: chars exist but density is tiny — body is in the
        # image, only a header (or other small caption text) is in the text layer.
        # Without this an image-of-text PDF with a tagged title would be NATIVE
        # and OCR would never fire.
        if (char_density < cfg.sparse_text_density
                and image_coverage >= cfg.sparse_text_image_coverage):
            return PageType.MIXED, 0.85
        return PageType.NATIVE, 0.85

    # 6. Few characters (1–50) — likely mixed or OCR with partial text
    if 0 < char_count < cfg.min_native_chars:
        if image_coverage >= cfg.scanned_image_coverage:
            return PageType.MIXED, 0.75
        if char_density < cfg.min_char_density:
            return PageType.MIXED, 0.65
        return PageType.NATIVE, 0.60

    # Fallback
    return PageType.MIXED, 0.50
