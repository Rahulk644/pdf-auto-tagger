"""
Stage 4 — Content router.

Routes each detected layout region to the appropriate specialist
based on MinerU2.5's category output.

This is pure Python — no models, no cost.  It's a lookup table
that determines which Stage 5 specialist handles each region.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

from tagger.models.data_types import LayoutCategory, LayoutRegion

logger = logging.getLogger(__name__)


class SpecialistType(str, Enum):
    """Which specialist module handles a given region."""

    TEXT_HANDLER = "text_handler"
    TABLE_EXTRACTOR = "table_extractor"
    FORMULA_EXTRACTOR = "formula_extractor"
    FIGURE_HANDLER = "figure_handler"
    ARTIFACT = "artifact"


@dataclass
class RoutedRegion:
    """A layout region with its assigned specialist."""

    region: LayoutRegion
    specialist: SpecialistType


# Routing table: LayoutCategory → SpecialistType
ROUTE_TABLE: dict[LayoutCategory, SpecialistType] = {
    LayoutCategory.TITLE:          SpecialistType.TEXT_HANDLER,
    LayoutCategory.SECTION_HEADER: SpecialistType.TEXT_HANDLER,
    LayoutCategory.TEXT:           SpecialistType.TEXT_HANDLER,
    LayoutCategory.LIST_ITEM:      SpecialistType.TEXT_HANDLER,
    LayoutCategory.TABLE:          SpecialistType.TABLE_EXTRACTOR,
    LayoutCategory.FORMULA:        SpecialistType.FORMULA_EXTRACTOR,
    LayoutCategory.PICTURE:        SpecialistType.FIGURE_HANDLER,
    LayoutCategory.CAPTION:        SpecialistType.TEXT_HANDLER,
    LayoutCategory.FOOTNOTE:       SpecialistType.TEXT_HANDLER,
    LayoutCategory.PAGE_HEADER:    SpecialistType.ARTIFACT,
    LayoutCategory.PAGE_FOOTER:    SpecialistType.ARTIFACT,
}


def route_regions(regions: list[LayoutRegion]) -> list[RoutedRegion]:
    """
    Route each layout region to its specialist.

    Args:
        regions: Layout regions from Stage 3.

    Returns:
        List of RoutedRegion with specialist assignments.
    """
    routed: list[RoutedRegion] = []

    for region in regions:
        specialist = ROUTE_TABLE.get(region.category, SpecialistType.TEXT_HANDLER)
        routed.append(RoutedRegion(region=region, specialist=specialist))

    # Log routing summary
    counts = {}
    for r in routed:
        counts[r.specialist.value] = counts.get(r.specialist.value, 0) + 1
    logger.info("Routed %d regions: %s", len(routed), counts)

    return routed
