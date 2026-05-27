"""
Stage 4 — Content Router (Region-First Rewrite)

THE FIX: MinerU regions are now the source of truth.
We iterate over MinerU regions, collect all PageElements
that fall inside each region's bbox, and merge them into a
single cohesive TaggedElement.
"""

from __future__ import annotations

import logging
from typing import Any

from tagger.models.data_types import (
    LayoutCategory,
    LayoutRegion,
    PageElement,
    PDFTag,
    TaggedElement,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Coordinate containment
# ---------------------------------------------------------------------------

def _containment_ratio(elem_bbox: tuple[float, float, float, float], region_bbox: tuple[float, float, float, float]) -> float:
    """
    What fraction of the element's area falls inside the region bbox?
    Returns 0.0–1.0. Used to decide if an element belongs to a region.
    """
    cx0, cy0, cx1, cy1 = elem_bbox
    rx0, ry0, rx1, ry1 = region_bbox

    elem_area = (cx1 - cx0) * (cy1 - cy0)
    if elem_area <= 0:
        return 0.0

    ix0 = max(cx0, rx0)
    iy0 = max(cy0, ry0)
    ix1 = min(cx1, rx1)
    iy1 = min(cy1, ry1)

    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0

    intersection = (ix1 - ix0) * (iy1 - iy0)
    return intersection / elem_area


def element_belongs_to_region(
    element: PageElement,
    region_bbox: tuple[float, float, float, float],
    threshold: float = 0.5,
) -> bool:
    """
    An element belongs to a region if at least `threshold` of its area
    is inside the region bbox.
    """
    return _containment_ratio(element.bbox, region_bbox) >= threshold


# ---------------------------------------------------------------------------
# Text merging
# ---------------------------------------------------------------------------

def _merge_elements_to_text(elements: list[PageElement]) -> str:
    """
    Merge a list of PageElements into a clean text string.
    Note: We PRESERVE the input list order, as Stage 2 already
    sorted them into reading order. We do NOT re-sort by coordinates.
    """
    if not elements:
        return ""

    result = []
    prev = None

    for elem in elements:
        text = elem.text
        if not text or text.isspace():
            continue

        if prev is None:
            result.append(text)
            prev = elem
            continue

        prev_top    = prev.bbox[1]
        curr_top    = elem.bbox[1]
        prev_size   = prev.font_size or 12.0

        # New line detection: vertical gap between tops exceeds ~half the font size
        vertical_gap = curr_top - prev_top
        if vertical_gap > prev_size * 0.5:
            result.append("\n")
        else:
            # Same line — check for word gap
            prev_x1   = prev.bbox[2]
            curr_x0   = elem.bbox[0]
            avg_width = prev_size * 0.4  # approximate char width
            if curr_x0 - prev_x1 > avg_width:
                result.append(" ")

        result.append(text)
        prev = elem

    return "".join(result).strip()


# ---------------------------------------------------------------------------
# Font metadata aggregation
# ---------------------------------------------------------------------------

def _aggregate_font_metadata(elements: list[PageElement]) -> dict:
    """
    Aggregate font properties across all elements in a region.
    Returns the dominant (most common) values, plus the max font size.
    """
    if not elements:
        return {"font_size": 12.0, "font_size_max": 12.0, "font_weight": "normal", "is_italic": False, "fontname": ""}

    sizes = [e.font_size for e in elements if e.font_size]
    names = [e.font_name for e in elements if e.font_name]
    weights = [e.font_weight for e in elements if e.font_weight]
    italics = [e.is_italic for e in elements]

    max_size  = max(sizes) if sizes else 12.0
    avg_size  = sum(sizes) / len(sizes) if sizes else 12.0
    dominant_font = max(set(names), key=names.count) if names else ""
    dominant_weight = max(set(weights), key=weights.count) if weights else "normal"

    is_italic = any(italics)

    return {
        "font_size":      avg_size,
        "font_size_max":  max_size,
        "font_weight":    dominant_weight,
        "is_italic":      is_italic,
        "fontname":       dominant_font,
    }


# ---------------------------------------------------------------------------
# Category → PDF tag mapping
# ---------------------------------------------------------------------------

CATEGORY_TO_TAG: dict[str, PDFTag] = {
    LayoutCategory.TITLE.value:          PDFTag.H1,
    LayoutCategory.SECTION_HEADER.value: PDFTag.H2,
    LayoutCategory.TEXT.value:           PDFTag.P,
    LayoutCategory.LIST_ITEM.value:      PDFTag.LI,
    LayoutCategory.TABLE.value:          PDFTag.TABLE,
    LayoutCategory.FORMULA.value:        PDFTag.FORMULA,
    LayoutCategory.PICTURE.value:        PDFTag.FIGURE,
    LayoutCategory.CAPTION.value:        PDFTag.CAPTION,
    LayoutCategory.FOOTNOTE.value:       PDFTag.NOTE,
    LayoutCategory.PAGE_HEADER.value:    PDFTag.ARTIFACT,
    LayoutCategory.PAGE_FOOTER.value:    PDFTag.ARTIFACT,
}


def _category_to_tag(category: str) -> PDFTag:
    return CATEGORY_TO_TAG.get(category, PDFTag.P)


# ---------------------------------------------------------------------------
# Unmatched elements handler
# ---------------------------------------------------------------------------

def _handle_unmatched_elements(
    unmatched: list[PageElement],
    page_num: int,
    start_reading_order: int,
) -> list[TaggedElement]:
    """
    Elements that fell outside every MinerU region are grouped by proximity
    and tagged as low-confidence P elements. These are MinerU's blind spots.
    """
    if not unmatched:
        return []

    # Elements are already in reading order from Stage 2.
    # Group by Y-position proximity (within 5px = same line cluster)
    groups: list[list[PageElement]] = []
    current: list[PageElement] = [unmatched[0]]

    for elem in unmatched[1:]:
        prev_top = current[-1].bbox[1]
        curr_top = elem.bbox[1]
        if abs(curr_top - prev_top) < 5:
            current.append(elem)
        else:
            groups.append(current)
            current = [elem]
    groups.append(current)

    tagged_elements = []
    for i, group in enumerate(groups):
        text = _merge_elements_to_text(group)
        if not text.strip():
            continue

        x0 = min(e.bbox[0] for e in group)
        y0 = min(e.bbox[1] for e in group)
        x1 = max(e.bbox[2] for e in group)
        y1 = max(e.bbox[3] for e in group)
        bbox = (x0, y0, x1, y1)

        font_meta = _aggregate_font_metadata(group)

        tagged_elements.append(TaggedElement(
            element_id=f"p{page_num}_unmatched_{i}",
            page_num=page_num,
            pdf_tag=PDFTag.P,
            text=text,
            bbox=bbox,
            confidence=0.4,  # Low — MinerU missed this
            font_name=font_meta["fontname"],
            font_size=font_meta["font_size"],
            font_weight=font_meta["font_weight"],
            layout_category="Text",
            review_reason="unmatched_by_mineru",
            merged_from=[src_id for e in group for src_id in e.merged_from],
        ))

    return tagged_elements


# ---------------------------------------------------------------------------
# Category priority for element-centric routing tie-breaking
# ---------------------------------------------------------------------------

# Lower number = higher priority. TABLE/FIGURE/FORMULA boundaries are strict
# structural regions that must always win over loose text regions.
_REGION_PRIORITY: dict[str, int] = {
    LayoutCategory.TABLE.value:   0,
    LayoutCategory.PICTURE.value: 1,
    LayoutCategory.FORMULA.value: 1,
}
_DEFAULT_PRIORITY = 2


# ---------------------------------------------------------------------------
# Core Stage 4: Element-centric router
# ---------------------------------------------------------------------------

def route_page(
    page_num: int,
    mineru_regions: list[LayoutRegion],
    page_elements: list[PageElement],
    containment_threshold: float = 0.5,
) -> list[TaggedElement]:
    """
    Element-centric routing: for each PageElement, evaluate containment ratio
    against ALL regions, then assign the element to the single best match.

    Best match is ranked by:
      1. Category priority  — TABLE/FIGURE/FORMULA beat TEXT in ties
      2. Containment ratio  — highest overlap wins among equal-priority regions

    This eliminates the greedy first-claimed-wins bug of the old region-first
    loop, where a text region numbered before a table region in reading order
    would permanently steal table cells before the table region was evaluated.

    Output contract is unchanged: one TaggedElement per MinerU region, plus
    residual P elements for any PageElements that matched no region.

    Args:
        page_num: 1-indexed page number
        mineru_regions: MinerU layout regions for this page
        page_elements: extracted text elements (already in reading order)
        containment_threshold: minimum fraction of element area that must
                               overlap a region for that region to be a
                               candidate match

    Returns:
        List of TaggedElement, one per MinerU region (plus unmatched residuals)
    """
    # Sort regions by reading order once — used for final TaggedElement ordering
    sorted_regions = sorted(
        mineru_regions,
        key=lambda r: getattr(r, "reading_order", 0),
    )

    # Pre-compute category string and priority for each region (avoid repeated lookups)
    region_meta: dict[str, tuple[str, int]] = {}
    for region in sorted_regions:
        cat = region.category.value if isinstance(region.category, LayoutCategory) else str(region.category)
        priority = _REGION_PRIORITY.get(cat, _DEFAULT_PRIORITY)
        region_meta[region.region_id] = (cat, priority)

    # Build per-region element buckets via element-centric assignment
    region_buckets: dict[str, list[PageElement]] = {r.region_id: [] for r in sorted_regions}
    unmatched: list[PageElement] = []

    for elem in page_elements:
        # Evaluate this element against every region
        candidates: list[tuple[int, float, LayoutRegion]] = []
        for region in sorted_regions:
            cat, priority = region_meta[region.region_id]
            r_bbox = region.bbox
            
            # Boundary Leakage Fix: MinerU's table bounding boxes often crop out
            # row headers on the left or overflowing numeric values on the right.
            # Dilate table boundaries horizontally (100px) and vertically (10px).
            if cat == LayoutCategory.TABLE.value:
                r_bbox = (r_bbox[0] - 100, r_bbox[1] - 10, r_bbox[2] + 100, r_bbox[3] + 10)

            ratio = _containment_ratio(elem.bbox, r_bbox)
            if ratio >= containment_threshold:
                candidates.append((priority, ratio, region))

        if not candidates:
            # Debug: log why the element was unmatched
            best_ratio = 0.0
            best_region = None
            for region in sorted_regions:
                r = _containment_ratio(elem.bbox, region.bbox)
                if r > best_ratio:
                    best_ratio = r
                    best_region = region
            if best_region:
                cat, _ = region_meta[best_region.region_id]
                logger.debug(
                    "Stage4 unmatched: elem=%s bbox=%s | best_region=%s (%s) bbox=%s ratio=%.3f < threshold=%.2f",
                    elem.element_id, tuple(round(v, 1) for v in elem.bbox),
                    best_region.region_id, cat,
                    tuple(round(v, 1) for v in best_region.bbox),
                    best_ratio, containment_threshold,
                )
            unmatched.append(elem)
            continue

        # Rank: primary = lower priority number (TABLE first), secondary = higher ratio
        candidates.sort(key=lambda c: (c[0], -c[1]))
        winner_priority, winner_ratio, winning_region = candidates[0]

        # Diagnostic: log when priority override changed the outcome
        # (i.e., the highest-ratio region was NOT the winner)
        if len(candidates) > 1:
            max_ratio_candidate = max(candidates, key=lambda c: c[1])
            if max_ratio_candidate[2].region_id != winning_region.region_id:
                bypassed_cat, _ = region_meta[max_ratio_candidate[2].region_id]
                winner_cat, _ = region_meta[winning_region.region_id]
                logger.debug(
                    "Stage4 priority save: elem=%s (text=%s...) "
                    "→ %s(%s) ratio=%.2f over %s(%s) ratio=%.2f",
                    elem.element_id,
                    (elem.text or "")[:20],
                    winning_region.region_id, winner_cat, winner_ratio,
                    max_ratio_candidate[2].region_id, bypassed_cat, max_ratio_candidate[1],
                )

        region_buckets[winning_region.region_id].append(elem)


    # Build TaggedElements in reading order, one per region
    tagged_elements: list[TaggedElement] = []

    for region in sorted_regions:
        region_bbox = region.bbox
        category, _ = region_meta[region.region_id]
        confidence = region.confidence
        region_elements = region_buckets[region.region_id]

        merged_text = _merge_elements_to_text(region_elements)
        font_meta = _aggregate_font_metadata(region_elements)

        needs_review = False
        review_reason = None

        # Low-confidence if region has a category but no chars were found
        if not region_elements and category not in ("Picture", "Formula"):
            needs_review = True
            review_reason = "no_chars_found"
            if confidence > 0.5:
                confidence = 0.5

        if confidence < 0.6:
            needs_review = True
            review_reason = "low_confidence"

        tagged_elements.append(TaggedElement(
            element_id=region.region_id,
            page_num=page_num,
            pdf_tag=_category_to_tag(category),
            text=merged_text,
            bbox=region_bbox,
            confidence=confidence,
            font_name=font_meta["fontname"],
            font_size=font_meta["font_size"],
            font_weight=font_meta["font_weight"],
            layout_category=category,
            needs_review=needs_review,
            review_reason=review_reason,
            original_mcid=region_elements[0].mcid if region_elements else None,
            merged_from=[src_id for e in region_elements for src_id in e.merged_from],
        ))

    # Residual elements that matched no region → low-confidence P fallbacks
    residuals = _handle_unmatched_elements(
        unmatched,
        page_num=page_num,
        start_reading_order=len(tagged_elements),
    )
    tagged_elements.extend(residuals)

    return tagged_elements

def diagnose_page(page_elements: list[TaggedElement]) -> dict:
    """
    Returns a summary you can log to verify the fix worked.
    A healthy page should show 1 Table element (not 50+).
    """
    from collections import Counter
    tag_counts = Counter(e.pdf_tag.value if hasattr(e.pdf_tag, 'value') else str(e.pdf_tag) for e in page_elements)
    flagged = [e for e in page_elements if e.needs_review]

    return {
        "total_elements": len(page_elements),
        "tag_distribution": dict(tag_counts),
        "flagged_count": len(flagged),
    }
