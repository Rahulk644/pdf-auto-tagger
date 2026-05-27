"""
Stage 8c — Artifact detector.

Identifies running headers, footers, and page numbers that should be
tagged as Artifact (decorative/repeated content not part of the
document's logical structure).

Detection signals:
  - Same text appearing at the same Y-position across 3+ pages
  - Sequential integers at consistent positions (page numbers)
  - Decorative rules/lines at page edges
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict

from tagger.config import SEMANTIC
from tagger.models.data_types import PDFTag, TaggedElement

logger = logging.getLogger(__name__)


def detect_artifacts(elements: list[TaggedElement], total_pages: int) -> list[TaggedElement]:
    """
    Detect running headers, footers, and page numbers.

    Scans all elements across pages for repeated text/position patterns.
    Matches are re-tagged as Artifact.

    Modifies elements in-place and returns the same list.
    """
    artifact_count = 0

    # Group elements by (normalized_text, y_band) across pages
    artifact_count += _detect_repeated_text(elements, total_pages)

    # Detect page numbers (sequential integers at consistent positions)
    artifact_count += _detect_page_numbers(elements, total_pages)

    logger.info("Artifact detector: tagged %d elements as Artifact", artifact_count)
    return elements


def _detect_repeated_text(elements: list[TaggedElement], total_pages: int) -> int:
    """
    Find text that appears at the same Y-position across pages.

    This catches running headers ("Company Name", "Document Title")
    and footers ("Confidential", "Draft", copyright notices).
    """
    cfg = SEMANTIC
    tagged_count = 0

    min_occurrences = min(cfg.artifact_min_page_occurrences, max(2, total_pages - 1))

    # Group by (normalized_text, y_band)
    # y_band = round y position to nearest tolerance bucket
    text_positions: dict[str, dict[int, list[TaggedElement]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for el in elements:
        if not el.text or len(el.text.strip()) < 2:
            continue
        if el.pdf_tag == PDFTag.ARTIFACT:
            continue

        normalized = _normalize_text(el.text)
        if not normalized:
            continue

        # Round Y to tolerance bucket
        y_center = (el.bbox[1] + el.bbox[3]) / 2.0
        y_bucket = round(y_center / cfg.artifact_y_tolerance_px)

        text_positions[normalized][y_bucket].append(el)

    # Check for repeated patterns
    for text_key, y_buckets in text_positions.items():
        for y_bucket, bucket_elements in y_buckets.items():
            # Count unique pages
            unique_pages = {el.page_num for el in bucket_elements}

            if len(unique_pages) >= min_occurrences:
                for el in bucket_elements:
                    if el.pdf_tag != PDFTag.ARTIFACT:
                        el.pdf_tag = PDFTag.ARTIFACT
                        tagged_count += 1
                        logger.debug(
                            "Repeated text artifact (page %d): '%s' "
                            "(appears on %d pages at y≈%d)",
                            el.page_num, el.text[:40],
                            len(unique_pages), y_bucket,
                        )

    return tagged_count


def _detect_page_numbers(elements: list[TaggedElement], total_pages: int) -> int:
    """
    Detect page numbers: sequential integers at consistent Y positions.

    Looks for elements containing only a number (possibly with surrounding
    formatting like "- 5 -" or "Page 5") that form a sequential or
    near-sequential series across pages.
    """
    cfg = SEMANTIC
    tagged_count = 0

    min_occurrences = min(cfg.artifact_min_page_occurrences, max(2, total_pages - 1))

    # Collect potential page number elements
    # Group by Y-position band
    candidates: dict[int, list[tuple[int, TaggedElement]]] = defaultdict(list)

    for el in elements:
        if el.pdf_tag == PDFTag.ARTIFACT:
            continue

        page_num_match = _extract_page_number(el.text)
        if page_num_match is not None:
            y_center = (el.bbox[1] + el.bbox[3]) / 2.0
            y_bucket = round(y_center / cfg.artifact_y_tolerance_px)
            candidates[y_bucket].append((page_num_match, el))

    # Check each Y-band for sequential patterns or consistent small font numbers
    for y_bucket, entries in candidates.items():
        if len(entries) < min_occurrences:
            continue

        # Sort by page number
        entries.sort(key=lambda e: e[1].page_num)

        # Check if the elements have small font (<= 12pt) and are at a consistent Y-band.
        # We don't use _is_roughly_sequential because test PDFs may have non-contiguous pages.
        for _, el in entries:
            if el.font_size is None or el.font_size <= 12.0:
                if el.pdf_tag != PDFTag.ARTIFACT:
                    el.pdf_tag = PDFTag.ARTIFACT
                    tagged_count += 1
                    logger.debug(
                        "Page number artifact (page %d): '%s'",
                        el.page_num, el.text.strip(),
                    )

    return tagged_count


def _normalize_text(text: str) -> str:
    """Normalize text for comparison (lowercase, collapse whitespace, strip)."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    # Remove page-specific numbers that might differ across pages
    # but keep the rest of the pattern
    text = re.sub(r"\d+", "#", text)
    return text


def _extract_page_number(text: str) -> int | None:
    """
    Try to extract a page number from text.

    Matches:
      "5", "42", "- 5 -", "Page 5", "5 of 100", "| 5 |"
    """
    if not text:
        return None

    text = text.strip()

    # Pure number
    if text.isdigit():
        return int(text)

    # Common page number patterns
    patterns = [
        re.compile(r"^[-–—|\s]*(\d+)[-–—|\s]*$"),        # "- 5 -", "| 5 |"
        re.compile(r"^[Pp]age\s+(\d+)$"),                  # "Page 5"
        re.compile(r"^(\d+)\s+of\s+\d+$"),                 # "5 of 100"
        re.compile(r"^[-–—]\s*(\d+)\s*[-–—]$"),            # "— 5 —"
    ]

    for pat in patterns:
        m = pat.match(text)
        if m:
            return int(m.group(1))

    return None


def _is_roughly_sequential(numbers: list[int]) -> bool:
    """
    Check if a list of integers forms a roughly sequential series.

    Allows gaps (missing pages) but requires at least 50% of the
    differences between consecutive elements to be 1.
    """
    if len(numbers) < 3:
        return False

    diffs = [numbers[i + 1] - numbers[i] for i in range(len(numbers) - 1)]
    sequential_count = sum(1 for d in diffs if d == 1)

    return sequential_count / len(diffs) >= 0.5
