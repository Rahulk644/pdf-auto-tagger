"""
Stage 5a — Text handler.

Converts text-typed layout regions into properly tagged elements.
Maps layout categories to PDF tags and preserves font metadata
for downstream heading ranking.

This handler covers: Title, Section-header, Text, List-item,
Caption, Footnote.
"""

from __future__ import annotations

import logging
import re

from tagger.models.data_types import (
    LayoutCategory,
    LayoutRegion,
    PageElement,
    PDFTag,
    TaggedElement,
)
from tagger.stage1_extraction.coord_transformer import bbox_contains

logger = logging.getLogger(__name__)

# Layout category → initial PDF tag mapping
_CATEGORY_TAG_MAP: dict[LayoutCategory, PDFTag] = {
    LayoutCategory.TITLE:          PDFTag.H1,
    LayoutCategory.SECTION_HEADER: PDFTag.H2,
    LayoutCategory.TEXT:           PDFTag.P,
    LayoutCategory.LIST_ITEM:      PDFTag.LI,
    LayoutCategory.CAPTION:        PDFTag.CAPTION,
    LayoutCategory.FOOTNOTE:       PDFTag.NOTE,
}

# Bullet/number patterns for list detection
_LIST_BULLET_PATTERN = re.compile(
    r"^[\u2022\u2023\u25E6\u2043\u2219•‣◦⁃∙◆■□▪▸►▻–—-]\s",
)
_LIST_NUMBER_PATTERN = re.compile(
    r"^(\d{1,3}[\.\)]\s|[a-zA-Z][\.\)]\s|[ivxlcdm]+[\.\)]\s|[IVXLCDM]+[\.\)]\s)",
)


def handle_text_regions(
    regions: list[LayoutRegion],
    elements: list[PageElement],
    page_num: int,
) -> list[TaggedElement]:
    """
    Convert text-category layout regions into TaggedElements.

    For each region, finds overlapping PageElements and creates
    TaggedElement(s) with the appropriate PDF tag.

    Args:
        regions: Layout regions with text-like categories.
        elements: Merged PageElements from Stage 2 for this page.
        page_num: 1-indexed page number.

    Returns:
        List of TaggedElement ready for validation.
    """
    tagged: list[TaggedElement] = []

    for region in regions:
        if region.category not in _CATEGORY_TAG_MAP:
            continue

        # Find elements within this region
        matched = _find_matching_elements(region, elements)

        if not matched:
            # Region with no text content — skip
            logger.debug(
                "Page %d: text region %s has no matching elements",
                page_num, region.region_id,
            )
            continue

        base_tag = _CATEGORY_TAG_MAP[region.category]

        for el in matched:
            # Refine tag based on text content
            refined_tag = _refine_tag(el, base_tag)

            tagged.append(TaggedElement(
                element_id=el.element_id,
                page_num=page_num,
                pdf_tag=refined_tag,
                text=el.text,
                bbox=el.bbox,
                confidence=region.confidence,
                original_mcid=el.mcid,
                font_name=el.font_name,
                font_size=el.font_size,
                font_weight=el.font_weight,
                merged_from=el.merged_from,
                layout_category=region.category.value,
            ))

    logger.debug("Page %d: text handler produced %d tagged elements", page_num, len(tagged))
    return tagged


def _find_matching_elements(
    region: LayoutRegion,
    elements: list[PageElement],
) -> list[PageElement]:
    """Find PageElements that fall within a layout region."""
    # First try pre-matched element IDs
    if region.matched_elements:
        element_map = {el.element_id: el for el in elements}
        return [
            element_map[eid] for eid in region.matched_elements
            if eid in element_map
        ]

    # Fall back to spatial matching
    return [
        el for el in elements
        if bbox_contains(region.bbox, el.bbox, tolerance=5.0)
    ]


def _refine_tag(element: PageElement, base_tag: PDFTag) -> PDFTag:
    """
    Refine a tag based on text content patterns.

    - Detect list items from bullet/number patterns
    - Keep base tag otherwise
    """
    if not element.text:
        return base_tag

    text = element.text.strip()

    # If tagged as P but starts with bullet → LI
    if base_tag == PDFTag.P:
        if _LIST_BULLET_PATTERN.match(text) or _LIST_NUMBER_PATTERN.match(text):
            return PDFTag.LI

    return base_tag
