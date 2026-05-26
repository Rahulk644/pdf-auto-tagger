"""
Stage 5d — Formula extractor.

Stub for UniMERNet formula-to-LaTeX extraction.
Deferred to quality-upgrade phase — requires Conda + heavy deps.

For now, formula regions are tagged as FORMULA with the raw text
content from pdfplumber (which may be garbled for math).
"""

from __future__ import annotations

import logging

from tagger.models.data_types import FormulaResult, LayoutRegion, PageElement

logger = logging.getLogger(__name__)


def extract_formula(
    region: LayoutRegion,
    matched_elements: list[PageElement],
) -> FormulaResult:
    """
    Extract formula content from a region.

    Currently uses raw text from pdfplumber. UniMERNet integration
    will replace this with proper LaTeX extraction.

    Args:
        region: Layout region classified as FORMULA.
        matched_elements: PageElements that overlap this region.

    Returns:
        FormulaResult with LaTeX string.
    """
    # For now, concatenate raw text from matched elements
    raw_text = " ".join(el.text for el in matched_elements if el.text).strip()

    if not raw_text:
        logger.debug("Region %s: empty formula region", region.region_id)
        return FormulaResult(
            region_id=region.region_id,
            latex="",
            is_inline=_is_inline_formula(region),
            confidence=0.3,
        )

    # Check if the text already looks like LaTeX
    is_latex = any(c in raw_text for c in ("\\", "^", "_", "{", "}"))

    return FormulaResult(
        region_id=region.region_id,
        latex=raw_text if is_latex else f"\\text{{{raw_text}}}",
        is_inline=_is_inline_formula(region),
        confidence=0.4 if is_latex else 0.2,
    )


def _is_inline_formula(region: LayoutRegion) -> bool:
    """
    Heuristic: inline formulas are typically short and narrow.

    Display formulas tend to be wider and on their own line.
    """
    height = region.bbox[3] - region.bbox[1]
    width = region.bbox[2] - region.bbox[0]

    # If the formula region is roughly one line tall → inline
    # Standard line height at 150 DPI ≈ 20-25px for 11pt text
    return height < 35 and width < 200
