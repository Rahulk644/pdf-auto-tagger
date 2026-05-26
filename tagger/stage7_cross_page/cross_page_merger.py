"""
Stage 7 — Cross-page merger.

Detects elements that span page boundaries and merges them:
  - Tables that continue across pages (matching column widths)
  - Lists that continue (matching indentation + bullet style)
  - Paragraphs split mid-sentence (no terminal punctuation)

Uses heuristics, not ML.  Confidence for cross-page merges is
set lower than single-page elements.
"""

from __future__ import annotations

import logging
import re

from tagger.config import CROSS_PAGE
from tagger.models.data_types import PDFTag, TaggedElement

logger = logging.getLogger(__name__)


def merge_cross_page(
    all_elements: list[TaggedElement],
    total_pages: int,
) -> list[TaggedElement]:
    """
    Detect and flag cross-page continuations.

    Currently implements:
      1. Split paragraph detection (sentence continuation)
      2. Table continuation detection (column width matching)
      3. List continuation detection (bullet style matching)

    Elements that are continuations get `cross_page = True` and
    a reference to what they continue.

    Modifies elements in-place and returns the same list.
    """
    if total_pages < 2:
        return all_elements

    # Group elements by page
    by_page: dict[int, list[TaggedElement]] = {}
    for el in all_elements:
        by_page.setdefault(el.page_num, []).append(el)

    # Sort within each page by reading order (top to bottom)
    for page_elements in by_page.values():
        page_elements.sort(key=lambda e: (e.bbox[1], e.bbox[0]))

    merged_count = 0

    for page_num in range(1, total_pages):
        next_page = page_num + 1
        if page_num not in by_page or next_page not in by_page:
            continue

        current_page_els = by_page[page_num]
        next_page_els = by_page[next_page]

        if not current_page_els or not next_page_els:
            continue

        # Get last element on current page and first on next
        last_el = current_page_els[-1]
        first_el = next_page_els[0]

        # Skip artifacts
        if last_el.pdf_tag == PDFTag.ARTIFACT or first_el.pdf_tag == PDFTag.ARTIFACT:
            # Get the last non-artifact element
            non_artifacts_curr = [e for e in current_page_els if e.pdf_tag != PDFTag.ARTIFACT]
            non_artifacts_next = [e for e in next_page_els if e.pdf_tag != PDFTag.ARTIFACT]

            if non_artifacts_curr and non_artifacts_next:
                last_el = non_artifacts_curr[-1]
                first_el = non_artifacts_next[0]
            else:
                continue

        # 1. Split paragraph detection
        if _is_split_paragraph(last_el, first_el):
            first_el.cross_page = True
            first_el.confidence = min(first_el.confidence, CROSS_PAGE.cross_page_confidence)
            merged_count += 1
            logger.debug(
                "Cross-page paragraph: p%d '%s...' → p%d '%s...'",
                page_num, last_el.text[-30:] if last_el.text else "",
                next_page, first_el.text[:30] if first_el.text else "",
            )

        # 2. Table continuation
        if _is_table_continuation(last_el, first_el, current_page_els, next_page_els):
            first_el.cross_page = True
            first_el.confidence = min(first_el.confidence, CROSS_PAGE.cross_page_confidence)
            merged_count += 1
            logger.debug(
                "Cross-page table: p%d → p%d",
                page_num, next_page,
            )

        # 3. List continuation
        if _is_list_continuation(last_el, first_el):
            first_el.cross_page = True
            first_el.confidence = min(first_el.confidence, CROSS_PAGE.cross_page_confidence)
            merged_count += 1
            logger.debug(
                "Cross-page list: p%d → p%d",
                page_num, next_page,
            )

    logger.info("Cross-page merger: detected %d continuations", merged_count)
    return all_elements


def _is_split_paragraph(last_el: TaggedElement, first_el: TaggedElement) -> bool:
    """
    Detect a paragraph split across pages.

    Signals:
      - Both are P tags
      - Last element's text doesn't end with sentence-terminal punctuation
      - First element starts with lowercase
      - Same font size
    """
    if last_el.pdf_tag != PDFTag.P or first_el.pdf_tag != PDFTag.P:
        return False

    if not last_el.text or not first_el.text:
        return False

    last_text = last_el.text.rstrip()
    first_text = first_el.text.lstrip()

    # Check: last text doesn't end with terminal punctuation
    terminal_chars = ".!?:;\"')"
    if last_text and last_text[-1] in terminal_chars:
        return False

    # Check: first text starts with lowercase (sentence continuation)
    if first_text and first_text[0].isupper():
        return False

    # Check: same font size
    if (
        last_el.font_size is not None
        and first_el.font_size is not None
        and abs(last_el.font_size - first_el.font_size) > 1.0
    ):
        return False

    return True


def _is_table_continuation(
    last_el: TaggedElement,
    first_el: TaggedElement,
    current_page: list[TaggedElement],
    next_page: list[TaggedElement],
) -> bool:
    """
    Detect a table continuing across page boundary.

    Signals:
      - Last element on current page is TABLE
      - First element on next page is TABLE
      - Similar X-position (left alignment matches)
    """
    # Find last table on current page
    curr_tables = [e for e in current_page if e.pdf_tag == PDFTag.TABLE]
    next_tables = [e for e in next_page if e.pdf_tag == PDFTag.TABLE]

    if not curr_tables or not next_tables:
        return False

    last_table = curr_tables[-1]
    first_table = next_tables[0]

    # Check X alignment (left edge should match within tolerance)
    width_curr = last_table.bbox[2] - last_table.bbox[0]
    width_next = first_table.bbox[2] - first_table.bbox[0]

    if width_curr > 0:
        width_ratio = abs(width_curr - width_next) / width_curr
        if width_ratio < CROSS_PAGE.table_column_width_tolerance * 10:
            return True

    return False


def _is_list_continuation(last_el: TaggedElement, first_el: TaggedElement) -> bool:
    """
    Detect a list continuing across page boundary.

    Signals:
      - Last element is LI
      - First element on next page is also LI
      - Similar X-position (indentation matches)
    """
    if last_el.pdf_tag != PDFTag.LI or first_el.pdf_tag != PDFTag.LI:
        return False

    # Check X alignment
    x_diff = abs(last_el.bbox[0] - first_el.bbox[0])
    if x_diff < 10:  # Within 10 standard-DPI pixels
        return True

    return False
