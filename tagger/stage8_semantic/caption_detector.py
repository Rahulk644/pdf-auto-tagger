"""
Stage 8d — Caption detector.

Pattern matching for figure and table captions:
  - Text immediately before/after a Figure or Table in reading order
  - Starts with "Figure", "Table", "Fig.", "Exhibit", etc.
  - Tags matched elements as Caption
"""

from __future__ import annotations

import logging
import re

from tagger.config import SEMANTIC
from tagger.models.data_types import PDFTag, TaggedElement

logger = logging.getLogger(__name__)

# Pre-compile caption patterns
_CAPTION_REGEXES = [re.compile(pat, re.IGNORECASE) for pat in SEMANTIC.caption_patterns]


def detect_captions(elements: list[TaggedElement]) -> list[TaggedElement]:
    """
    Detect figure and table captions.

    Looks for text elements immediately adjacent (in reading order)
    to Figure or Table elements that match caption patterns.

    Modifies elements in-place and returns the same list.
    """
    caption_count = 0

    # Group elements by page
    by_page: dict[int, list[TaggedElement]] = {}
    for el in elements:
        by_page.setdefault(el.page_num, []).append(el)

    for page_num, page_elements in by_page.items():
        # Sort by reading order (top-to-bottom, left-to-right)
        page_elements.sort(key=lambda e: (e.bbox[1], e.bbox[0]))

        for i, el in enumerate(page_elements):
            if el.pdf_tag not in (PDFTag.FIGURE, PDFTag.TABLE):
                continue

            # Check the element immediately AFTER the figure/table
            if i + 1 < len(page_elements):
                next_el = page_elements[i + 1]
                if _is_caption_text(next_el):
                    next_el.pdf_tag = PDFTag.CAPTION
                    caption_count += 1
                    logger.debug(
                        "Caption after %s (page %d): '%s'",
                        el.pdf_tag.value, page_num, next_el.text[:50],
                    )
                    continue

            # Check the element immediately BEFORE the figure/table
            if i - 1 >= 0:
                prev_el = page_elements[i - 1]
                if _is_caption_text(prev_el):
                    prev_el.pdf_tag = PDFTag.CAPTION
                    caption_count += 1
                    logger.debug(
                        "Caption before %s (page %d): '%s'",
                        el.pdf_tag.value, page_num, prev_el.text[:50],
                    )

    logger.info("Caption detector: tagged %d captions", caption_count)
    return elements


def _is_caption_text(element: TaggedElement) -> bool:
    """
    Check if an element looks like a caption.

    Requirements:
      - Currently tagged as P (paragraph) — not already a heading, etc.
      - Text matches one of the caption patterns
      - Text is relatively short (< 500 chars — captions aren't essays)
    """
    if element.pdf_tag != PDFTag.P:
        return False

    text = element.text.strip()
    if not text or len(text) > 500:
        return False

    return any(regex.match(text) for regex in _CAPTION_REGEXES)
