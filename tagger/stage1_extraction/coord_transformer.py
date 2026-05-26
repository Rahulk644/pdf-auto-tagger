"""
Coordinate transformer — single source of truth for bbox normalization.

PDF native coordinates:
  - 1 unit = 1/72 inch (72 DPI)
  - Origin at BOTTOM-LEFT of the page
  - Y increases upward

Standardized pipeline coordinates:
  - 150 DPI (configurable via config.STANDARD_DPI)
  - Origin at TOP-LEFT of the page
  - Y increases downward (screen convention)

Every bbox in the pipeline is expressed in standardized coords after
Stage 1 extraction.  Stage 10 writeback uses the inverse transform.
"""

from __future__ import annotations

from tagger.config import STANDARD_DPI, PDF_NATIVE_DPI


def pdf_to_standard(
    bbox: tuple[float, float, float, float],
    page_height_pt: float,
    target_dpi: int = STANDARD_DPI,
) -> tuple[float, float, float, float]:
    """
    Convert a PDF-native bbox to standardized display coordinates.

    Args:
        bbox: (x0, y0_pdf, x1, y1_pdf) in PDF points (72 DPI, bottom-left origin).
        page_height_pt: Total page height in PDF points.
        target_dpi: Target DPI for output coordinates.

    Returns:
        (x0, y0, x1, y1) in target DPI, top-left origin.
        Guaranteed: x0 <= x1, y0 <= y1.
    """
    scale = target_dpi / PDF_NATIVE_DPI
    x0_pdf, y0_pdf, x1_pdf, y1_pdf = bbox

    # Scale to target DPI
    x0 = x0_pdf * scale
    x1 = x1_pdf * scale

    # Flip Y-axis: PDF origin is bottom-left, standard is top-left
    # In PDF coords, y0_pdf < y1_pdf means y0 is lower on the page
    y0 = (page_height_pt - y1_pdf) * scale
    y1 = (page_height_pt - y0_pdf) * scale

    # Ensure consistent ordering
    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0

    return (x0, y0, x1, y1)


def standard_to_pdf(
    bbox: tuple[float, float, float, float],
    page_height_pt: float,
    source_dpi: int = STANDARD_DPI,
) -> tuple[float, float, float, float]:
    """
    Inverse transform: standardized coords → PDF-native coords.

    Used by Stage 10 writeback to convert back to PDF coordinate space.

    Args:
        bbox: (x0, y0, x1, y1) in standard DPI, top-left origin.
        page_height_pt: Total page height in PDF points.
        source_dpi: DPI of the input coordinates.

    Returns:
        (x0, y0_pdf, x1, y1_pdf) in PDF points (72 DPI, bottom-left origin).
    """
    scale = PDF_NATIVE_DPI / source_dpi
    x0_std, y0_std, x1_std, y1_std = bbox

    x0 = x0_std * scale
    x1 = x1_std * scale

    # Reverse Y-axis flip
    y0_pdf = page_height_pt - (y1_std * scale)
    y1_pdf = page_height_pt - (y0_std * scale)

    if x0 > x1:
        x0, x1 = x1, x0
    if y0_pdf > y1_pdf:
        y0_pdf, y1_pdf = y1_pdf, y0_pdf

    return (x0, y0_pdf, x1, y1_pdf)


def pdfplumber_to_standard(
    bbox: tuple[float, float, float, float],
    page_height_pt: float,
    target_dpi: int = STANDARD_DPI,
) -> tuple[float, float, float, float]:
    """
    Convert a pdfplumber bbox to standardized coordinates.

    pdfplumber uses a DIFFERENT convention than raw PDF:
      - Origin at TOP-LEFT (already flipped from PDF native)
      - Units are still PDF points (72 DPI)
      - bbox format: (x0, top, x1, bottom) where top < bottom

    So we only need to scale, not flip Y.

    Args:
        bbox: (x0, top, x1, bottom) from pdfplumber, in PDF points.
        page_height_pt: Total page height in PDF points (unused here but
                        kept for API consistency).
        target_dpi: Target DPI for output coordinates.

    Returns:
        (x0, y0, x1, y1) in target DPI, top-left origin.
    """
    scale = target_dpi / PDF_NATIVE_DPI
    x0, top, x1, bottom = bbox

    return (
        x0 * scale,
        top * scale,
        x1 * scale,
        bottom * scale,
    )


def compute_iou(
    bbox_a: tuple[float, float, float, float],
    bbox_b: tuple[float, float, float, float],
) -> float:
    """
    Compute Intersection-over-Union between two bboxes.

    Both bboxes must be in the same coordinate system.
    Returns 0.0 if no overlap, 1.0 if identical.
    """
    x0 = max(bbox_a[0], bbox_b[0])
    y0 = max(bbox_a[1], bbox_b[1])
    x1 = min(bbox_a[2], bbox_b[2])
    y1 = min(bbox_a[3], bbox_b[3])

    if x1 <= x0 or y1 <= y0:
        return 0.0

    intersection = (x1 - x0) * (y1 - y0)
    area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
    area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def bbox_contains(
    outer: tuple[float, float, float, float],
    inner: tuple[float, float, float, float],
    tolerance: float = 2.0,
) -> bool:
    """
    Check if `outer` bbox fully contains `inner` bbox.

    Args:
        outer: The containing bbox.
        inner: The bbox to check containment of.
        tolerance: Pixels of slack allowed (for floating-point imprecision).

    Returns:
        True if inner is fully within outer (with tolerance).
    """
    return (
        inner[0] >= outer[0] - tolerance
        and inner[1] >= outer[1] - tolerance
        and inner[2] <= outer[2] + tolerance
        and inner[3] <= outer[3] + tolerance
    )
