"""
Stage 8a — Heading level ranker.

Assigns heading levels (H1–H6) based on font size ranking.
No LLM needed — pure algorithmic approach using pdfplumber font metadata.

Algorithm:
  1. Collect all elements classified as headings (Title, Section-header)
  2. Extract unique font sizes
  3. Merge sizes within 1pt tolerance
  4. Rank: largest → H1, second → H2, etc. (max H6)
"""

from __future__ import annotations

import logging
from collections import defaultdict

from tagger.config import SEMANTIC
from tagger.models.data_types import PDFTag, TaggedElement

logger = logging.getLogger(__name__)


def assign_heading_levels(elements: list[TaggedElement]) -> list[TaggedElement]:
    """
    Assign H1–H6 tags based on font size ranking.

    Elements that have layout_category in ("Title", "Section-header")
    and currently have a generic heading tag are re-assigned.

    Modifies elements in-place and returns the same list.
    """
    # Collect heading elements
    heading_tags = {PDFTag.H1, PDFTag.H2, PDFTag.H3, PDFTag.H4, PDFTag.H5, PDFTag.H6}
    headings = [
        el for el in elements
        if el.pdf_tag in heading_tags
        and el.font_size is not None
        and el.font_size > 0
    ]

    if not headings:
        logger.debug("No headings with font size data to rank")
        return elements

    # Collect unique font sizes
    raw_sizes = sorted(
        set(h.font_size for h in headings if h.font_size is not None),
        reverse=True,
    )

    if not raw_sizes:
        return elements

    # Merge sizes within tolerance
    merged_sizes = _merge_similar_sizes(raw_sizes, SEMANTIC.heading_size_tolerance_pt)

    # Build level map: largest → H1, second → H2, etc.
    level_map: dict[float, PDFTag] = {}
    heading_tags_ordered = [PDFTag.H1, PDFTag.H2, PDFTag.H3, PDFTag.H4, PDFTag.H5, PDFTag.H6]

    for idx, size in enumerate(merged_sizes):
        level_idx = min(idx, SEMANTIC.max_heading_levels - 1)
        level_map[size] = heading_tags_ordered[level_idx]

    # Assign levels
    assigned_count = 0
    for el in headings:
        closest_size = _find_closest(el.font_size, merged_sizes)
        new_tag = level_map.get(closest_size, PDFTag.H6)

        if el.pdf_tag != new_tag:
            logger.debug(
                "Heading '%s...' (%.1fpt): %s → %s",
                el.text[:30], el.font_size, el.pdf_tag.value, new_tag.value,
            )
            el.pdf_tag = new_tag
            assigned_count += 1

    logger.info(
        "Heading ranker: %d unique sizes → %d levels, reassigned %d/%d headings",
        len(raw_sizes), len(merged_sizes), assigned_count, len(headings),
    )
    return elements


def _merge_similar_sizes(
    sizes: list[float],
    tolerance: float,
) -> list[float]:
    """
    Merge font sizes that are within `tolerance` points of each other.

    Input must be sorted descending.
    Returns merged sizes (descending), each representing a group.
    """
    if not sizes:
        return []

    merged: list[float] = [sizes[0]]

    for size in sizes[1:]:
        if merged[-1] - size <= tolerance:
            # Within tolerance — merge (keep the larger size as representative)
            continue
        merged.append(size)

    return merged


def _find_closest(value: float, candidates: list[float]) -> float:
    """Find the closest value in candidates to the given value."""
    if not candidates:
        return value
    return min(candidates, key=lambda c: abs(c - value))
