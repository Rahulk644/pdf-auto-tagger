"""
Stage 8b — TOC (Table of Contents) detector.

Pattern matching for TOC entries:
  - Text ending in page numbers with dot leaders
  - Clustered in the first 10% of document pages
  - Matching heading text found elsewhere in document

Tags detected entries as TOCI (Table of Contents Item).
"""

from __future__ import annotations

import logging
import re

from tagger.config import SEMANTIC
from tagger.models.data_types import PDFTag, TaggedElement

logger = logging.getLogger(__name__)

# Patterns for TOC entries
# Matches: "Chapter 1 .............. 5" or "Introduction 12" or "1.2.3 Methods....42"
_TOC_PATTERNS = [
    re.compile(r"^.+?\.{3,}\s*\d+\s*$"),             # dot leaders + page num
    re.compile(r"^.+?\s{3,}\d+\s*$"),                  # space leaders + page num
    re.compile(r"^\d+(\.\d+)*\s+.+?\s+\d+\s*$"),       # "1.2.3 Title 42"
    re.compile(r"^(Chapter|Section|Part)\s+\d+.*\d+$", re.IGNORECASE),
    re.compile(r"^\d{1,3}\s*[A-Za-z].*\.{3,}"),        # page-num prefix + title + dot leaders
]

# TOC entries are often mis-tagged as headings upstream (MinerU classifies
# dot-leader lines as Title/Section-header); the dot-leader pattern is the
# structural signal, so consider headings as candidates too.
_TOC_CANDIDATE_TAGS = {
    PDFTag.P, PDFTag.H1, PDFTag.H2, PDFTag.H3, PDFTag.H4, PDFTag.H5, PDFTag.H6,
}


def detect_toc_entries(
    elements: list[TaggedElement],
    total_pages: int,
) -> list[TaggedElement]:
    """
    Detect and re-tag TOC entries.

    Only considers elements in the first N% of the document (configured by
    SEMANTIC.toc_page_fraction) to avoid false positives on content pages
    that happen to end with numbers.

    Modifies elements in-place and returns the same list.
    """
    if total_pages == 0:
        return elements

    # Only look in the first N% of pages, but always at least the first 3 —
    # the fraction alone misses TOCs on short docs (e.g. page 2 of a 15-page doc).
    max_toc_page = max(int(total_pages * SEMANTIC.toc_page_fraction), 3)

    # Collect candidate elements (P or heading-tagged — see _TOC_CANDIDATE_TAGS)
    candidates = [
        el for el in elements
        if el.page_num <= max_toc_page
        and el.pdf_tag in _TOC_CANDIDATE_TAGS
        and el.text
    ]

    if not candidates:
        return elements

    # Check each candidate against TOC patterns
    toc_count = 0
    for el in candidates:
        text = el.text.strip()
        if any(pat.match(text) for pat in _TOC_PATTERNS):
            el.pdf_tag = PDFTag.TOCI
            toc_count += 1
            logger.debug(
                "TOC entry detected (page %d): '%s'",
                el.page_num, text[:50],
            )

    # Heuristic: if we found TOC entries, check if there's a cluster
    # (≥3 entries on the same page = likely a real TOC page)
    if toc_count > 0:
        toc_pages = {}
        for el in elements:
            if el.pdf_tag == PDFTag.TOCI:
                toc_pages[el.page_num] = toc_pages.get(el.page_num, 0) + 1

        # If a page has only 1-2 potential TOC entries, it might be
        # a false positive — lower confidence
        for el in elements:
            if el.pdf_tag == PDFTag.TOCI:
                page_toc_count = toc_pages.get(el.page_num, 0)
                if page_toc_count < 3:
                    el.confidence = min(el.confidence, 0.65)

    logger.info(
        "TOC detector: found %d entries in first %d pages (of %d total)",
        toc_count, max_toc_page, total_pages,
    )
    return elements
