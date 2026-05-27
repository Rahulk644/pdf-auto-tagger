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
    page_elements: list[PageElement] = None,
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
            # Convert region bbox (150 DPI) to pdfplumber coords (72 DPI, Top-Left)
            from tagger.config import STANDARD_DPI, PDF_NATIVE_DPI
            scale = STANDARD_DPI / PDF_NATIVE_DPI
            # region.bbox is (x0, y0, x1, y1) in 150 DPI Top-Left
            x0, y0, x1, y1 = region.bbox
            pad = 5
            
            # pdfplumber expects (x0, top, x1, bottom) in 72 DPI Top-Left
            crop_box = (
                max(0, x0 / scale - pad),
                max(0, y0 / scale - pad),
                min(page.width, x1 / scale + pad),
                min(page.height, y1 / scale + pad),
            )
            
            cropped = page.within_bbox(crop_box)

            # Find tables in the cropped region using text strategy
            # Text strategy is crucial for tables without gridlines
            table_settings = {"vertical_strategy": "text", "horizontal_strategy": "text"}
            tables = cropped.find_tables(table_settings=table_settings)
            if not tables:
                # Try the full page and find table closest to our region
                tables = page.find_tables(table_settings=table_settings)
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

            # Build cells data structure with spatial mapping
            cells_data = []
            if page_elements:
                # Convert region bbox to PDF coords to match table bbox coords if needed
                # Wait, pdfplumber's table.cells are in PDF coordinates!
                # But PageElement bbox is in standard DPI (150).
                # We need to map standard DPI -> PDF coords for intersection!
                # Wait, pdfplumber's cell is in PDF coords.
                pass

            # Map PageElements to their center points in pdfplumber coords (72 DPI, Top-Left)
            from tagger.config import STANDARD_DPI, PDF_NATIVE_DPI
            scale = STANDARD_DPI / PDF_NATIVE_DPI
            
            # Pre-compute elements in pdfplumber coords if available
            element_points = []
            if page_elements:
                for el in page_elements:
                    # el.bbox is (x0, y0, x1, y1) in 150 DPI (Top-Left)
                    cx = ((el.bbox[0] + el.bbox[2]) / 2.0) / scale
                    cy = ((el.bbox[1] + el.bbox[3]) / 2.0) / scale
                    element_points.append((el, cx, cy))

            for row_idx, row in enumerate(table.rows):
                is_header_row = has_header and row_idx == 0
                for col_idx, cell_bbox in enumerate(row.cells):
                    cell_text_val = rows[row_idx][col_idx] if row_idx < len(rows) and col_idx < len(rows[row_idx]) else None
                    cell_text = str(cell_text_val).strip() if cell_text_val is not None else ""
                    
                    is_row_header = (
                        col_idx == 0
                        and not is_header_row
                        and bool(cell_text)
                        and not _is_numeric_content(cell_text)
                    )
                    
                    merged_from = []
                    if cell_bbox is not None and element_points:
                        cx0, cy0, cx1, cy1 = cell_bbox
                        for el, cx, cy in element_points:
                            if cx0 <= cx <= cx1 and cy0 <= cy <= cy1:
                                merged_from.extend(el.merged_from)
                    
                    # Store as dict instead of TableCell object to avoid widespread refactoring
                    cells_data.append({
                        "row_idx": row_idx,
                        "col_idx": col_idx,
                        "is_header": is_header_row,
                        "is_row_header": is_row_header,
                        "text": cell_text,
                        "merged_from": merged_from,
                        "bbox": cell_bbox,
                    })

            html = _build_html(rows, has_header)

            struct = TableStructure(
                region_id=region.region_id,
                html=html,
                num_rows=num_rows,
                num_cols=num_cols,
                has_header=has_header,
                confidence=0.75,
            )
            # Monkey-patch cells onto the returned structure 
            struct.cells = cells_data
            return struct

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
