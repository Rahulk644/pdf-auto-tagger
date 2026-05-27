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

    Uses pdfplumber's find_tables() within the region bbox, then assigns
    characters to cells via center-point containment against pdfplumber's
    native 72-DPI cell bboxes. merged_from IDs match Stage 1's p{n}_c{idx}
    scheme by replicating the same enumerate(page.chars) indexing.

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

            # Build full-page char index in Stage-1-compatible order.
            # Stage 1 assigns element_id = f"p{page_num}_c{char_idx}" where
            # char_idx is the enumerate index into page.chars (raw, unfiltered).
            # Replicate the same skip conditions so IDs match exactly.
            page_char_index: list[tuple[int, dict]] = []
            for char_idx, ch in enumerate(page.chars or []):
                text = ch.get("text", "")
                if not text or text.isspace():
                    continue
                x0 = float(ch.get("x0", 0))
                top = float(ch.get("top", 0))
                x1 = float(ch.get("x1", 0))
                bottom = float(ch.get("bottom", 0))
                if x1 - x0 < 0.1 or bottom - top < 0.1:
                    continue
                page_char_index.append((char_idx, ch))

            # Crop page to region bbox (convert from 150-DPI standard to 72-DPI PDF points)
            from tagger.config import STANDARD_DPI, PDF_NATIVE_DPI
            scale = STANDARD_DPI / PDF_NATIVE_DPI
            x0, y0, x1, y1 = region.bbox
            pad = 5
            crop_box = (
                max(0, x0 / scale - pad),
                max(0, y0 / scale - pad),
                min(page.width, x1 / scale + pad),
                min(page.height, y1 / scale + pad),
            )
            cropped = page.within_bbox(crop_box)

            table_settings = {"vertical_strategy": "text", "horizontal_strategy": "text"}
            tables = cropped.find_tables(table_settings=table_settings)
            if not tables:
                tables = page.find_tables(table_settings=table_settings)
                if not tables:
                    return None

                best_table = None
                best_overlap = 0
                for t in tables:
                    overlap = _compute_overlap_area((x0, y0, x1, y1), t.bbox)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_table = t

                if best_table is None:
                    return None
                tables = [best_table]

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

            num_rows = len(rows)
            num_cols = max(len(row) for row in rows) if rows else 0

            if num_rows == 0 or num_cols == 0:
                return None

            total_cells = sum(
                1 for row in rows for cell in row if cell is not None and str(cell).strip()
            )
            if total_cells < TABLE.min_cells:
                logger.debug(
                    "Page %d: table has only %d non-empty cells (min=%d), skipping",
                    page_num, total_cells, TABLE.min_cells,
                )
                return None

            has_header = all(
                cell is not None and str(cell).strip()
                for cell in rows[0]
            ) if rows else False

            cells_data = []
            for row_idx, row in enumerate(table.rows):
                is_header_row = has_header and row_idx == 0
                for col_idx, cell_bbox in enumerate(row.cells):
                    merged_from: list[str] = []
                    # Use pdfplumber's extracted text — it handles space reconstruction
                    # from PDF positioning commands, which raw char joining cannot do.
                    raw_cell_text = rows[row_idx][col_idx] if row_idx < len(rows) and col_idx < len(rows[row_idx]) else None
                    cell_text = str(raw_cell_text).strip() if raw_cell_text is not None else ""

                    if cell_bbox is not None:
                        cx0, cy0, cx1, cy1 = cell_bbox

                        # Collect chars whose center point falls inside this cell.
                        # Character-level center points are sufficient — individual
                        # glyphs are small enough that they never straddle boundaries.
                        cell_chars: list[tuple[int, dict]] = []
                        for char_idx, ch in page_char_index:
                            char_cx = (float(ch["x0"]) + float(ch["x1"])) / 2.0
                            char_cy = (float(ch["top"]) + float(ch["bottom"])) / 2.0
                            if cx0 <= char_cx <= cx1 and cy0 <= char_cy <= cy1:
                                cell_chars.append((char_idx, ch))

                        # Reading order: top-to-bottom, left-to-right
                        cell_chars.sort(key=lambda x: (float(x[1]["top"]), float(x[1]["x0"])))

                        merged_from = [f"p{page_num}_c{char_idx}" for char_idx, _ in cell_chars]

                    is_numeric = _is_numeric_content(cell_text) if cell_text else (not bool(merged_from))
                    is_row_header = (
                        col_idx == 0
                        and not is_header_row
                        and (bool(cell_text) or bool(merged_from))
                        and not is_numeric
                    )

                    cells_data.append({
                        "row_idx": row_idx,
                        "col_idx": col_idx,
                        "is_header": is_header_row,
                        "is_row_header": is_row_header,
                        "text": cell_text,
                        "merged_from": merged_from,
                        "bbox": cell_bbox,
                    })

            html = _build_html(cells_data, num_rows, num_cols)

            struct = TableStructure(
                region_id=region.region_id,
                html=html,
                num_rows=num_rows,
                num_cols=num_cols,
                has_header=has_header,
                confidence=0.75,
            )
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
    cleaned = (
        text.strip()
        .lstrip("$")
        .replace(",", "")
        .replace("(", "")
        .replace(")", "")
        .replace("%", "")
        .replace("-", "")
        .strip()
    )
    return not cleaned or cleaned.replace(".", "").isdigit()


def _build_html(cells_data: list[dict], num_rows: int, _num_cols: int) -> str:
    """Build an HTML table string from extracted cells_data."""
    parts = ["<table>"]

    rows_map: dict[int, list[dict]] = {}
    for c in cells_data:
        rows_map.setdefault(c["row_idx"], []).append(c)

    for row_idx in range(num_rows):
        if row_idx not in rows_map:
            continue
        parts.append("  <tr>")
        for cell in sorted(rows_map[row_idx], key=lambda x: x["col_idx"]):
            cell_text = (
                cell.get("text", "")
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            if cell.get("is_header"):
                parts.append(f'    <th scope="col">{cell_text}</th>')
            elif cell.get("is_row_header"):
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
