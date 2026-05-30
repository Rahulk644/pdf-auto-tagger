"""
Stage 1 — Native text extraction via pdfplumber.

Extracts every character from native PDF pages with full font metadata:
  - text content
  - bounding box (converted to standard 150-DPI coords)
  - font name, size, weight (bold detection), color
  - MCID (if the PDF is already tagged)

This is the extraction path for pages classified as "native" by Stage 0.
Scanned pages use the MinerU2.5 OCR path instead (scanned_extractor.py).

Design note: This borrows patterns from the incumbent-QA-Tool's
extract_pdfplumber_data() but outputs PageElement dataclasses instead
of the raw dict format, and normalizes coordinates immediately.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import pdfplumber

from tagger.config import STANDARD_DPI
from tagger.models.data_types import PageClassification, PageElement, PageType
from tagger.page_cache import open_pdf
from tagger.stage1_extraction.coord_transformer import pdfplumber_to_standard

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# Regex patterns for bold detection in font names
_BOLD_PATTERNS = re.compile(
    r"(Bold|Bd|Black|Heavy|Demi|Semibold|SemiBold)",
    re.IGNORECASE,
)

_ITALIC_PATTERNS = re.compile(
    r"(Italic|Ital|It|Oblique|Slant)",
    re.IGNORECASE,
)


def extract_native_pages(
    pdf_path: str | Path,
    classifications: list[PageClassification],
) -> dict[int, list[PageElement]]:
    """
    Extract text elements from all native (and mixed) pages.

    Args:
        pdf_path: Path to the PDF file.
        classifications: Stage 0 output — used to filter to native/mixed pages.

    Returns:
        Dict mapping page_num → list of PageElement (one per character group).
        Only contains entries for native and mixed pages.
    """
    native_pages = {
        c.page_num for c in classifications
        if c.page_type in (PageType.NATIVE, PageType.MIXED)
    }

    if not native_pages:
        logger.info("No native pages to extract")
        return {}

    results: dict[int, list[PageElement]] = {}

    with open_pdf(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            page_num = page_idx + 1
            if page_num not in native_pages:
                continue

            try:
                elements = _extract_page_chars(page, page_num)
                results[page_num] = elements
                logger.debug(
                    "Page %d: extracted %d character groups",
                    page_num, len(elements),
                )
            except Exception as e:
                logger.error("Page %d: extraction failed: %s", page_num, e)
                results[page_num] = []

    total = sum(len(v) for v in results.values())
    logger.info(
        "Extracted %d elements from %d native/mixed pages",
        total, len(results),
    )
    return results


def _extract_page_chars(
    page: pdfplumber.page.Page,
    page_num: int,
) -> list[PageElement]:
    """
    Extract characters from a single page, grouped by character.

    Each character becomes a PageElement at this stage.
    Stage 2 (text merger) will group them into words → lines → paragraphs.
    """
    chars = page.chars or []
    if not chars:
        return []

    page_height_pt = float(page.height)
    elements: list[PageElement] = []

    for char_idx, ch in enumerate(chars):
        text = ch.get("text", "")
        if not text or text.isspace():
            continue

        # Extract bbox — pdfplumber gives (x0, top, x1, bottom), top-left origin
        x0 = float(ch.get("x0", 0))
        top = float(ch.get("top", 0))
        x1 = float(ch.get("x1", 0))
        bottom = float(ch.get("bottom", 0))

        # Skip degenerate bboxes
        if x1 - x0 < 0.1 or bottom - top < 0.1:
            continue

        # Convert to standard coords
        bbox = pdfplumber_to_standard(
            (x0, top, x1, bottom),
            page_height_pt=page_height_pt,
        )

        # Font metadata
        font_name = ch.get("fontname", None)
        font_size = _safe_float(ch.get("size", None))
        font_weight = "bold" if font_name and _BOLD_PATTERNS.search(font_name) else "normal"
        is_italic = bool(font_name and _ITALIC_PATTERNS.search(font_name))

        # Color — pdfplumber gives stroking_color or non_stroking_color
        # non_stroking_color is the fill color (text color)
        font_color = _extract_color(ch.get("non_stroking_color"))

        # MCID — if the PDF is already tagged
        mcid = ch.get("mcid", None)
        if mcid is not None:
            try:
                mcid = int(mcid)
            except (ValueError, TypeError):
                mcid = None

        element = PageElement(
            element_id=f"p{page_num}_c{char_idx}",
            page_num=page_num,
            text=text,
            bbox=bbox,
            font_name=font_name,
            font_size=font_size,
            font_weight=font_weight,
            font_color=font_color,
            is_italic=is_italic,
            upright=bool(ch.get("upright", True)),
            source="pdfplumber",
            confidence=1.0,
            mcid=mcid,
        )
        elements.append(element)

    return elements


def _safe_float(val) -> float | None:
    """Safely convert a value to float, returning None on failure."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _extract_color(color_val) -> str | None:
    """
    Convert pdfplumber color value to hex string.

    pdfplumber can return:
      - None
      - A single float (grayscale, 0.0 = black, 1.0 = white)
      - A tuple of floats (RGB or CMYK)
    """
    if color_val is None:
        return None

    try:
        if isinstance(color_val, (int, float)):
            # Grayscale: 0 = black, 1 = white
            gray = int(max(0, min(255, float(color_val) * 255)))
            return f"#{gray:02x}{gray:02x}{gray:02x}"

        if isinstance(color_val, (tuple, list)):
            if len(color_val) == 3:
                # RGB (values 0–1)
                r = int(max(0, min(255, float(color_val[0]) * 255)))
                g = int(max(0, min(255, float(color_val[1]) * 255)))
                b = int(max(0, min(255, float(color_val[2]) * 255)))
                return f"#{r:02x}{g:02x}{b:02x}"

            if len(color_val) == 4:
                # CMYK → RGB conversion
                c, m, y, k = (float(v) for v in color_val)
                r = int(255 * (1 - c) * (1 - k))
                g = int(255 * (1 - m) * (1 - k))
                b = int(255 * (1 - y) * (1 - k))
                return f"#{r:02x}{g:02x}{b:02x}"

            if len(color_val) == 1:
                # Single-element tuple (grayscale)
                gray = int(max(0, min(255, float(color_val[0]) * 255)))
                return f"#{gray:02x}{gray:02x}{gray:02x}"

    except (ValueError, TypeError, IndexError):
        pass

    return None
