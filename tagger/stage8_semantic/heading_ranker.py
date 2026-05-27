"""
Stage 8a — Heading level ranker.

Assigns heading levels (H1–H6) using document-wide font rarity and full text
style (size + weight + color), not raw font-size ranking alone.

No LLM needed — pure algorithmic approach using pdfplumber font metadata.

Algorithm:
  1. Build a document-wide frequency map of font sizes across ALL elements.
     A size occurring in more than `heading_body_frequency_fraction` of
     elements is "body-common" — it does not anchor a distinct heading level.
  2. Group heading elements by TextStyle tuple (size bucket + weight + color).
  3. Order the rare ("structural") styles by size desc, bold-first, and assign
     H1, H2, … sequentially (clamped to H6).
  4. Body-common heading styles fold into the deepest structural level so they
     do not inflate the hierarchy (the failure mode on multi-size documents).
"""

from __future__ import annotations

import logging
from collections import Counter

from tagger.config import SEMANTIC
from tagger.models.data_types import PDFTag, TaggedElement

logger = logging.getLogger(__name__)

_HEADING_TAGS = (PDFTag.H1, PDFTag.H2, PDFTag.H3, PDFTag.H4, PDFTag.H5, PDFTag.H6)


def assign_heading_levels(elements: list[TaggedElement]) -> list[TaggedElement]:
    """
    Assign H1–H6 tags using font rarity + text style.

    Modifies elements in-place and returns the same list.
    """
    heading_set = set(_HEADING_TAGS)
    headings = [
        el for el in elements
        if el.pdf_tag in heading_set
        and el.font_size is not None
        and el.font_size > 0
    ]

    if not headings:
        logger.debug("No headings with font size data to rank")
        return elements

    tol = SEMANTIC.heading_size_tolerance_pt

    # 1. Document-wide font-size frequency (rarity signal)
    size_freq: Counter[float] = Counter()
    for el in elements:
        if el.font_size is not None and el.font_size > 0:
            size_freq[_bucket(el.font_size, tol)] += 1
    total_sized = sum(size_freq.values())
    body_threshold = total_sized * SEMANTIC.heading_body_frequency_fraction

    # 2. Group headings by TextStyle tuple
    styles: dict[tuple, float] = {}  # style key -> representative size bucket
    for h in headings:
        key = _style_key(h, tol)
        styles.setdefault(key, key[0])

    # 3. Rare ("structural") styles anchor the hierarchy
    structural = sorted(
        (k for k in styles if size_freq[k[0]] <= body_threshold),
        key=lambda k: (-k[0], 0 if k[1] == "bold" else 1, k[2]),
    )

    level_map: dict[tuple, PDFTag] = {}
    if structural:
        for idx, key in enumerate(structural):
            level_map[key] = _HEADING_TAGS[min(idx, SEMANTIC.max_heading_levels - 1)]
        deepest = level_map[structural[-1]]
        # 4. Body-common heading styles fold into the deepest structural level
        for key in styles:
            level_map.setdefault(key, deepest)
    else:
        # Fallback: every heading size is body-common — rank distinct buckets by size
        ordered = sorted(styles, key=lambda k: (-k[0], 0 if k[1] == "bold" else 1, k[2]))
        for idx, key in enumerate(ordered):
            level_map[key] = _HEADING_TAGS[min(idx, SEMANTIC.max_heading_levels - 1)]

    # Assign
    assigned_count = 0
    for el in headings:
        new_tag = level_map.get(_style_key(el, tol), PDFTag.H6)
        if el.pdf_tag != new_tag:
            logger.debug(
                "Heading '%s...' (%.1fpt %s): %s → %s",
                el.text[:30], el.font_size, el.font_weight or "normal",
                el.pdf_tag.value, new_tag.value,
            )
            el.pdf_tag = new_tag
            assigned_count += 1

    logger.info(
        "Heading ranker: %d styles (%d structural) → reassigned %d/%d headings",
        len(styles), len(structural), assigned_count, len(headings),
    )
    return elements


def _bucket(size: float, tol: float) -> float:
    """Round a font size to the nearest tolerance bucket so near-equal sizes merge."""
    if tol <= 0:
        return round(size, 1)
    return round(size / tol) * tol


def _style_key(el: TaggedElement, tol: float) -> tuple[float, str, str]:
    """TextStyle tuple: (size bucket, weight, color).

    font_color is not currently propagated onto TaggedElement, so it degrades
    to "" today; read defensively so the tuple upgrades automatically if color
    is added upstream later.
    """
    return (
        _bucket(el.font_size, tol),
        (el.font_weight or "normal").lower(),
        (getattr(el, "font_color", None) or "").lower(),
    )
