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


# TABLE.engine value -> rapid_table ModelType name. slanet/ppstructure are ONNX
# (light, CPU-fast); unitable is the torch backend (~480MB, accurate but heavy —
# load it in its own process to avoid OOM on 8GB).
_ENGINE_TO_MODELTYPE = {
    "slanet": "SLANETPLUS",
    "ppstructure": "PPSTRUCTURE_EN",
    "unitable": "UNITABLE",
}


def _load() -> bool:
    global _engine, _failed
    if _engine is not None or _failed:
        return _engine is not None
    try:
        from rapid_table import RapidTable, RapidTableInput, ModelType
        from tagger.config import TABLE
        mt_name = _ENGINE_TO_MODELTYPE.get(TABLE.engine, "SLANETPLUS")
        # NB: rapid_table needs use_ocr=True even though we override cell text
        # with native chars — its HTML grid is built FROM the OCR cell positions
        # (use_ocr=False yields cell_bboxes but no pred_htmls). The OCR cost is
        # intrinsic; SLANet (1.37s) / TableFormerV2 (0.82s) are the faster engines
        # if speed outweighs PP-Structure's +0.01 TEDS-S.
        _engine = RapidTable(RapidTableInput(model_type=getattr(ModelType, mt_name)))
        logger.info("rapid_table table extractor loaded (%s)", mt_name)
        return True
    except Exception as e:
        _failed = True
        logger.info("rapid_table unavailable (%s) — table engine stays TableFormer", e)
        return False


def _parse_cells_in_order(html: str) -> list[dict]:
    """SLANet/PP-Structure HTML -> cells in document (row-major) order, ALIGNED
    with rapid_table's cell_bboxes. row_idx from <tr>, col_idx positional within
    the row; is_header = a <th> or row 0 (so the table always exposes TH for AT).
    Spans not expanded (positional col_idx) — refinement; grid + header is what
    AT needs. Text is left empty here and filled from NATIVE pdfplumber chars."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    cells: list[dict] = []
    for r, tr in enumerate(soup.find_all("tr")):
        for c, cell in enumerate(tr.find_all(["td", "th"])):
            cells.append({
                "row_idx": r, "col_idx": c,
                "is_header": (cell.name == "th") or (r == 0),
                "is_row_header": (c == 0 and r > 0),
                "ocr_text": cell.get_text(strip=True),  # fallback if no bbox/native
                "text": "", "merged_from": [], "bbox": None,
            })
    return cells


def _native_chars(pdf_path, page_num):
    """pdfplumber chars as (char_idx, cx_150, cy_150, text) in 150-DPI page space.
    char_idx is the Stage-1-compatible enumerate index (for merged_from / MCID)."""
    from tagger.config import PDF_NATIVE_DPI, STANDARD_DPI
    from tagger.page_cache import open_pdf
    scale = STANDARD_DPI / PDF_NATIVE_DPI
    out = []
    with open_pdf(pdf_path) as pdf:
        if page_num - 1 >= len(pdf.pages):
            return out
        for ci, ch in enumerate(pdf.pages[page_num - 1].chars or []):
            t = ch.get("text", "")
            if not t or t.isspace():
                continue
            x0, top, x1, bottom = ch["x0"], ch["top"], ch["x1"], ch["bottom"]
            if x1 - x0 < 0.1 or bottom - top < 0.1:
                continue
            out.append((ci, (x0 + x1) / 2 * scale, (top + bottom) / 2 * scale, t))
    return out


def extract_table_slanet(pdf_path, page_num: int, region):
    """Parse a table region with the active rapid_table model. Fills cell TEXT
    from NATIVE pdfplumber chars (exact, content-stream-mapped via merged_from) —
    not the engine's OCR — so the resulting /TD/TH carry accurate text + MCIDs.
    Returns a TableStructure (with .cells) or None (caller falls back)."""
    if not _load():
        return None
    try:
        import numpy as np
        from tagger.page_cache import render_page

        img = render_page(pdf_path, page_num)
        if img is None:
            return None
        rx0, ry0, rx1, ry1 = (int(v) for v in region.bbox)
        crop = img.crop((rx0, ry0, rx1, ry1))
        res = _engine(np.array(crop))
        htmls = getattr(res, "pred_htmls", None)
        if not htmls or not htmls[0]:
            return None

        cells = _parse_cells_in_order(htmls[0])
        cbboxes = getattr(res, "cell_bboxes", None)
        # cell_bboxes is batched per-image ([array(N,4|8)]); unwrap to this image.
        if cbboxes is not None and len(cbboxes) == 1 and hasattr(cbboxes[0], "__len__") \
                and len(cbboxes[0]) == len(cells):
            cbboxes = cbboxes[0]
        chars = _native_chars(pdf_path, page_num)

        # Native-text fill: cell_bboxes are crop-local 150-DPI; offset by the
        # region top-left to page 150-DPI, then assign the native chars whose
        # centre falls inside each cell (in reading order -> text + merged_from).
        if cbboxes is not None and len(cbboxes) == len(cells) and chars:
            for cell, bb in zip(cells, cbboxes):
                xs = [float(v) for v in bb[0::2]]; ys = [float(v) for v in bb[1::2]]
                bx0, by0 = min(xs) + rx0, min(ys) + ry0
                bx1, by1 = max(xs) + rx0, max(ys) + ry0
                inside = [(ci, t) for ci, cx, cy, t in chars
                          if bx0 <= cx <= bx1 and by0 <= cy <= by1]
                inside.sort(key=lambda z: z[0])
                cell["text"] = "".join(t for _, t in inside).strip()
                cell["merged_from"] = [f"p{page_num}_c{ci}" for ci, _ in inside]
        else:  # no usable bboxes -> fall back to the engine's OCR text
            for cell in cells:
                cell["text"] = cell["ocr_text"]

        cells = [c for c in cells if c["text"]]
        if len(cells) < 2:
            return None
        num_rows = 1 + max(c["row_idx"] for c in cells)
        num_cols = 1 + max(c["col_idx"] for c in cells)
        struct = TableStructure(
            region_id=region.region_id, html=_build_html(cells, num_rows),
            num_rows=num_rows, num_cols=num_cols,
            has_header=any(c["is_header"] for c in cells), confidence=0.75,
        )
        struct.cells = cells
        return struct
    except Exception as e:
        logger.warning("rapid_table extraction failed on p%d: %s", page_num, e)
        return None
