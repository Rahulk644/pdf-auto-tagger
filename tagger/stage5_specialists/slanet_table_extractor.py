"""SLANet (rapid_table, ONNX) table-structure extractor.

Image-to-markup: reads the table crop and emits HTML structure (rows/cols,
header cells) directly, bypassing the grid-coordinate math that breaks TableFormer
on merged / borderless / gridless tables. Clean CPU deps — onnxruntime (the same
runtime as our RapidOCR) + a ~7.4 MB slanet-plus.onnx; no paddle, no torch.

Why it's here (measured): on a dp-bench single-table TEDS A/B against the Docling
TableFormer tier, SLANet scored 0.843 vs 0.750 and RESCUED every doc where
TableFormer collapsed to 0.000 (110, 122) or near-zero (119). Selected via
TABLE.engine == "slanet" (env TAGGER_TABLE_ENGINE) in the extract_table_native
cascade, replacing the TableFormer model tier.

Cells carry SLANet's OCR text with empty `merged_from`, so Stage 10 emits them
with /ActualText only (the canonical no-MCID path, PDF/UA-valid). Header row is
heuristically row 0 (SLANet rarely emits <th>) so the table always has TH cells.
Self-gates to None if rapid_table is unavailable.
"""
from __future__ import annotations

import logging

from tagger.models.data_types import TableStructure
from tagger.stage5_specialists.docling_table_extractor import _build_html

logger = logging.getLogger(__name__)

_engine = None
_failed = False


def _load() -> bool:
    global _engine, _failed
    if _engine is not None or _failed:
        return _engine is not None
    try:
        from rapid_table import RapidTable, RapidTableInput
        _engine = RapidTable(RapidTableInput())
        logger.info("SLANet (rapid_table, ONNX) table extractor loaded")
        return True
    except Exception as e:
        _failed = True
        logger.info("rapid_table/SLANet unavailable (%s) — table engine stays TableFormer", e)
        return False


def _parse_cells(html: str) -> list[dict]:
    """SLANet HTML -> our cells_data shape. row_idx from <tr> order, col_idx from
    cell order within the row. is_header: a <th>, or row 0 (default header row so
    the table always exposes TH cells for AT). Spans are not expanded (col_idx is
    positional) — a refinement; the row/col grid + header is what AT needs."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    cells: list[dict] = []
    for r, tr in enumerate(soup.find_all("tr")):
        for c, cell in enumerate(tr.find_all(["td", "th"])):
            text = cell.get_text(strip=True)
            is_header = (cell.name == "th") or (r == 0)
            cells.append({
                "row_idx": r, "col_idx": c,
                "is_header": is_header,
                "is_row_header": (c == 0 and r > 0 and bool(text)),
                "text": text, "merged_from": [], "bbox": None,
            })
    return cells


def extract_table_slanet(pdf_path, page_num: int, region):
    """Parse a table region with SLANet. Returns a TableStructure (with .cells)
    or None on failure / unavailability (caller falls back)."""
    if not _load():
        return None
    try:
        import numpy as np
        from tagger.page_cache import render_page

        img = render_page(pdf_path, page_num)
        if img is None:
            return None
        x0, y0, x1, y1 = (int(v) for v in region.bbox)
        crop = img.crop((x0, y0, x1, y1))
        res = _engine(np.array(crop))
        htmls = getattr(res, "pred_htmls", None)
        if not htmls or not htmls[0]:
            return None

        cells = [c for c in _parse_cells(htmls[0]) if c["text"]]
        if len(cells) < 2:
            return None
        num_rows = 1 + max(c["row_idx"] for c in cells)
        num_cols = 1 + max(c["col_idx"] for c in cells)
        struct = TableStructure(
            region_id=region.region_id, html=_build_html(cells, num_rows),
            num_rows=num_rows, num_cols=num_cols,
            has_header=any(c["is_header"] for c in cells), confidence=0.7,
        )
        struct.cells = cells
        return struct
    except Exception as e:
        logger.warning("SLANet table extraction failed on p%d: %s", page_num, e)
        return None
