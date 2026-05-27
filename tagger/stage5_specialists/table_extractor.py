"""
Stage 5b — Table extractor.

Two extraction paths:
  - Native PDFs: pdfplumber's table detection (find_tables + extract_tables)
  - Scanned PDFs: MinerU's built-in table extraction

Outputs TableStructure with HTML representation for struct tree writeback.
StructEqTable deferred to quality-upgrade phase.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pdfplumber

from tagger.config import TABLE
from tagger.models.data_types import (
    LayoutRegion,
    PageClassification,
    PageType,
    TableStructure,
)

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def extract_table_native(
    pdf_path: str | Path,
    page_num: int,
    region: LayoutRegion,
    classification: PageClassification,
) -> TableStructure | None:
    """
    Extract table structure from a native PDF page using pdfplumber.

    Uses pdfplumber's find_tables() within the region bbox, then
    extract_table() to get cell contents.

    Args:
        pdf_path: Path to the PDF file.
        page_num: 1-indexed page number.
        region: The layout region classified as TABLE.
        classification: Page classification (for native vs scanned routing).

    Returns:
        TableStructure if extraction succeeds, None otherwise.
    """
    if classification.page_type == PageType.SCANNED:
        logger.debug("Page %d: skipping pdfplumber table on scanned page", page_num)
        return None

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            if page_num > len(pdf.pages):
                return None

            page = pdf.pages[page_num - 1]

            # Crop page to region bbox (convert from standard DPI to PDF points)
            from tagger.stage1_extraction.coord_transformer import standard_to_pdf
            from tagger.config import STANDARD_DPI

            pdf_bbox = standard_to_pdf(
                region.bbox,
                page_height_pt=float(page.height),
                source_dpi=STANDARD_DPI,
            )

            # Clamp bbox to page bounds
            x0 = max(0, pdf_bbox[0])
            y0 = max(0, pdf_bbox[1])
            x1 = min(float(page.width), pdf_bbox[2])
            y1 = min(float(page.height), pdf_bbox[3])

            if x1 <= x0 or y1 <= y0:
                return None

            cropped = page.within_bbox((x0, y0, x1, y1))

            # Find tables in the cropped region
            tables = cropped.find_tables()
            if not tables:
                # Try the full page and find table closest to our region
                tables = page.find_tables()
                if not tables:
                    return None

                # Find best overlapping table
                best_table = None
                best_overlap = 0
                for t in tables:
                    t_bbox = t.bbox  # (x0, top, x1, bottom) in page coords
                    overlap = _compute_overlap_area(
                        (x0, y0, x1, y1),
                        t_bbox,
                    )
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_table = t

                if best_table is None:
                    return None
                tables = [best_table]

            # Extract the first (best) table
            table = tables[0]
            rows = table.extract()

            if not rows:
                return TableStructure(
                    region_id=region.region_id,
                    html="<table></table>",
                    num_rows=0,
                    num_cols=0,
                    has_header=False,
                    confidence=0.3,
                )

            # Build HTML
            num_rows = len(rows)
            num_cols = max(len(row) for row in rows) if rows else 0

            if num_rows == 0 or num_cols == 0:
                return None

            # Check minimum cells threshold
            total_cells = sum(
                1 for row in rows for cell in row if cell is not None and str(cell).strip()
            )
            if total_cells < TABLE.min_cells:
                logger.debug(
                    "Page %d: table has only %d non-empty cells (min=%d), skipping",
                    page_num, total_cells, TABLE.min_cells,
                )
                return None

            # Heuristic: first row is header if it has different formatting
            # (simple: check if first row has no None cells)
            has_header = all(
                cell is not None and str(cell).strip()
                for cell in rows[0]
            ) if rows else False

            html = _build_html(rows, has_header)

            return TableStructure(
                region_id=region.region_id,
                html=html,
                num_rows=num_rows,
                num_cols=num_cols,
                has_header=has_header,
                confidence=0.75,
            )

    except Exception as e:
        logger.warning(
            "Page %d: pdfplumber table extraction failed: %s",
            page_num, e,
        )
        return None


def _is_numeric_content(text: str) -> bool:
    """Return True if text is empty or contains only numeric/currency content."""
    if not text:
        return True
    # Strip currency, commas, parens (negatives), percent, dashes, spaces
    cleaned = text.strip().lstrip("$").replace(",", "").replace("(", "").replace(")", "").replace("%", "").replace("-", "").strip()
    return not cleaned or cleaned.replace(".", "").isdigit()


def _build_html(rows: list[list], has_header: bool) -> str:
    """Build an HTML table string from extracted rows."""
    parts = ["<table>"]

    for row_idx, row in enumerate(rows):
        parts.append("  <tr>")
        for col_idx, cell in enumerate(row):
            cell_text = str(cell).strip() if cell is not None else ""
            # Escape HTML entities
            cell_text = (
                cell_text
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )

            is_header_row = has_header and row_idx == 0
            is_row_header = (
                col_idx == 0
                and not is_header_row
                and bool(cell_text)  # empty first-col cells are structural gaps, not labels
                and not _is_numeric_content(cell_text)
            )

            if is_header_row:
                parts.append(f'    <th scope="col">{cell_text}</th>')
            elif is_row_header:
                parts.append(f'    <th scope="row">{cell_text}</th>')
            else:
                parts.append(f"    <td>{cell_text}</td>")
        parts.append("  </tr>")

    parts.append("</table>")
    return "\n".join(parts)


def _compute_overlap_area(
    bbox_a: tuple[float, float, float, float],
    bbox_b: tuple[float, float, float, float],
) -> float:
    """Compute overlap area between two bboxes."""
    x0 = max(bbox_a[0], bbox_b[0])
    y0 = max(bbox_a[1], bbox_b[1])
    x1 = min(bbox_a[2], bbox_b[2])
    y1 = min(bbox_a[3], bbox_b[3])

    if x1 <= x0 or y1 <= y0:
        return 0.0

    return (x1 - x0) * (y1 - y0)
