"""
Stage 1 — Scanned page extraction via MinerU2.5.

Stub module for the MinerU OCR extraction path.
This handles pages classified as "scanned" by Stage 0.

MinerU2.5 handles OCR internally (PaddleOCR built in), so no
separate OCR library (Surya, Tesseract) is needed.

NOTE: This module requires the `mineru` optional dependency:
    pip install "pdf-auto-tagger[mineru]"

The actual implementation will be filled in Phase 2 (P4) when
we integrate the MinerU layout model. For now, this provides
the interface contract.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from tagger.models.data_types import PageClassification, PageElement, PageType

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def extract_scanned_pages(
    pdf_path: str | Path,
    classifications: list[PageClassification],
) -> dict[int, list[PageElement]]:
    """
    Extract text elements from scanned pages using MinerU2.5 OCR.

    Args:
        pdf_path: Path to the PDF file.
        classifications: Stage 0 output — used to filter to scanned/mixed pages.

    Returns:
        Dict mapping page_num → list of PageElement.
        Only contains entries for scanned pages (mixed pages get both
        native + OCR extraction, merged by the pipeline orchestrator).
    """
    scanned_pages = {
        c.page_num for c in classifications
        if c.page_type in (PageType.SCANNED, PageType.MIXED)
    }

    if not scanned_pages:
        logger.info("No scanned pages to extract")
        return {}

    # Check if MinerU is available
    try:
        from mineru_vl_utils import MinerUClient  # noqa: F401
    except ImportError:
        logger.warning(
            "MinerU not installed. %d scanned page(s) will have no text. "
            "Install with: pip install \"pdf-auto-tagger[mineru]\"",
            len(scanned_pages),
        )
        return {page_num: [] for page_num in scanned_pages}

    # TODO: Implement MinerU2.5 extraction in Phase 2 (P4)
    logger.warning(
        "MinerU extraction not yet implemented. "
        "%d scanned page(s) will have no text.",
        len(scanned_pages),
    )
    return {page_num: [] for page_num in scanned_pages}
