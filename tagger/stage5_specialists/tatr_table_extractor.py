"""Stage 5c — Borderless/complex table structure via TATR (Table Transformer).

pdfplumber lattice only sees RULED grids; ~half of real tables are borderless
(no ruling lines) and lattice/text-strategy mangle them. TATR
(microsoft/table-transformer-structure-recognition, ~28M, DETR, MIT) predicts the
row/column structure from the table IMAGE, which works with or without ruling lines.
It runs on CPU (small model on a single cropped region — the decoupled
"small-specialist-on-a-crop" pattern), so it fits the GPU-free backend.

This module is OPTIONAL and self-gating: if torch/transformers/the model are
unavailable, or inference fails, `extract_table_tatr` returns None and Stage 5 falls
back to its existing pdfplumber path — never breaking the pipeline.

Pipeline contract (same as extract_table_native): returns a TableStructure whose
`cells` carry row_idx/col_idx/text/merged_from/is_header so Stage 10 builds TR>TH/TD
and injects BDC markers. Chars are assigned to TATR cells by center-point containment,
exactly like the lattice path.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from tagger.config import PDF_NATIVE_DPI, STANDARD_DPI
from tagger.models.data_types import (
    LayoutRegion,
    PageClassification,
    PageType,
    TableStructure,
)

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

_STRUCT_MODEL = "microsoft/table-transformer-structure-recognition"
_DET_MODEL = "microsoft/table-transformer-detection"
_SCALE = STANDARD_DPI / PDF_NATIVE_DPI
_STRUCT_THRESH = 0.5  # row/column boxes inside a confirmed table crop — keep liberal
_DET_THRESH = 0.8     # PAGE-level table detection — keep conservative: TATR detection
# at 0.5 over-fires on text-heavy layouts (dense prose, multi-column) and classifies
# body text as <table>, cratering NID/MHS on non-table docs. 0.8 keeps the borderless
# wins while filtering the false positives. Calibration, not metric-tuning.
_PAD = 12  # px padding around the region crop — TATR was trained with table margins

# TATR structure-model labels.
_L_ROW = "table row"
_L_COL = "table column"
_L_COL_HEADER = "table column header"

_model = None
_processor = None
_det_model = None
_det_processor = None
_load_failed = False
_det_load_failed = False


def _load():
    """Lazy, one-time STRUCTURE model load. Sets _load_failed so we don't retry."""
    global _model, _processor, _load_failed
    if _model is not None or _load_failed:
        return _model is not None
    try:
        import torch  # noqa: F401
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
        _processor = AutoImageProcessor.from_pretrained(_STRUCT_MODEL)
        _model = AutoModelForObjectDetection.from_pretrained(_STRUCT_MODEL)
        _model.eval()
        logger.info("TATR structure model loaded (%s)", _STRUCT_MODEL)
        return True
    except Exception as e:
        _load_failed = True
        logger.info("TATR structure unavailable (%s) — falls back to pdfplumber", e)
        return False


def _load_det():
    """Lazy, one-time DETECTION model load."""
    global _det_model, _det_processor, _det_load_failed
    if _det_model is not None or _det_load_failed:
        return _det_model is not None
    try:
        import torch  # noqa: F401
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
        _det_processor = AutoImageProcessor.from_pretrained(_DET_MODEL)
        _det_model = AutoModelForObjectDetection.from_pretrained(_DET_MODEL)
        _det_model.eval()
        logger.info("TATR detection model loaded (%s)", _DET_MODEL)
        return True
    except Exception as e:
        _det_load_failed = True
        logger.info("TATR detection unavailable (%s) — borderless tables undetected", e)
        return False


def detect_tables(pdf_path: "str | Path", page_num: int) -> list[tuple]:
    """Detect table regions on a page (born-digital, no GPU) — the borderless-aware
    counterpart to pdfplumber lattice. Returns bboxes in 150-DPI standard coords.
    Empty list when TATR is unavailable or the model finds nothing."""
    if not _load_det():
        return []
    try:
        import fitz
        import torch
        from PIL import Image
        with fitz.open(str(pdf_path)) as doc:
            if page_num - 1 >= len(doc):
                return []
            pix = doc[page_num - 1].get_pixmap(dpi=STANDARD_DPI)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        inputs = _det_processor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = _det_model(**inputs)
        res = _det_processor.post_process_object_detection(
            outputs, threshold=_DET_THRESH,
            target_sizes=torch.tensor([img.size[::-1]]))[0]
        id2label = _det_model.config.id2label
        # Detection model emits 'table' and 'table rotated'; we want 'table'.
        out = []
        for score, label, box in zip(res["scores"], res["labels"], res["boxes"]):
            if id2label[int(label)] == "table":
                out.append(tuple(float(v) for v in box))
        return out
    except Exception as e:
        logger.warning("TATR detection failed on page %d: %s", page_num, e)
        return []


def _detect_structure(crop_img):
    """Run TATR on a PIL crop -> list of (label, (x0,y0,x1,y1)) in crop-pixel coords."""
    import torch
    inputs = _processor(images=crop_img, return_tensors="pt")
    with torch.no_grad():
        outputs = _model(**inputs)
    target = torch.tensor([crop_img.size[::-1]])  # (height, width)
    res = _processor.post_process_object_detection(
        outputs, threshold=_STRUCT_THRESH, target_sizes=target)[0]
    id2label = _model.config.id2label
    out = []
    for score, label, box in zip(res["scores"], res["labels"], res["boxes"]):
        out.append((id2label[int(label)], [float(v) for v in box]))
    return out


def _grid_from_rows_cols(rows, cols):
    """Build an ordered R×C cell-bbox grid (no span merging in v1) from TATR row/column
    bands. rows/cols are bbox lists; a cell is the intersection of a row band and a
    column band."""
    rows = sorted(rows, key=lambda b: (b[1] + b[3]) / 2)
    cols = sorted(cols, key=lambda b: (b[0] + b[2]) / 2)
    grid = []
    for r in rows:
        row_cells = []
        for c in cols:
            # cell = column's x-extent × row's y-extent
            row_cells.append((c[0], r[1], c[2], r[3]))
        grid.append(row_cells)
    return grid


def extract_table_tatr(
    pdf_path: "str | Path",
    page_num: int,
    region: LayoutRegion,
    classification: PageClassification,
) -> TableStructure | None:
    """Extract a (borderless) table's structure with TATR. Returns None on any failure
    so Stage 5 can fall back to pdfplumber."""
    if classification.page_type == PageType.SCANNED:
        return None
    if not _load():
        return None
    try:
        import fitz  # PyMuPDF
        from PIL import Image

        rx0, ry0, rx1, ry1 = region.bbox  # 150-DPI standard, origin top-left
        with fitz.open(str(pdf_path)) as doc:
            if page_num - 1 >= len(doc):
                return None
            pix = doc[page_num - 1].get_pixmap(dpi=STANDARD_DPI)  # px == standard-150
            page_img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        cx0 = max(0, int(rx0) - _PAD)
        cy0 = max(0, int(ry0) - _PAD)
        cx1 = min(page_img.width, int(rx1) + _PAD)
        cy1 = min(page_img.height, int(ry1) + _PAD)
        crop = page_img.crop((cx0, cy0, cx1, cy1))
        if crop.width < 8 or crop.height < 8:
            return None

        dets = _detect_structure(crop)
        rows = [b for lbl, b in dets if lbl == _L_ROW]
        cols = [b for lbl, b in dets if lbl == _L_COL]
        headers = [b for lbl, b in dets if lbl == _L_COL_HEADER]
        if len(rows) < 2 or len(cols) < 2:
            return None  # not a real grid — let pdfplumber/text handle it

        # crop-pixel -> standard-150 coords (add crop origin)
        def to_std(b):
            return (b[0] + cx0, b[1] + cy0, b[2] + cx0, b[3] + cy0)
        grid = _grid_from_rows_cols([to_std(b) for b in rows], [to_std(b) for b in cols])
        header_bands = [to_std(b) for b in headers]

        cells_data = _assign_chars(pdf_path, page_num, grid, header_bands)
        if cells_data is None:
            return None
        num_rows = len(grid)
        num_cols = len(grid[0]) if grid else 0
        nonempty = sum(1 for c in cells_data if c["text"] or c["merged_from"])
        if nonempty < 4:
            return None

        html = _build_html(cells_data, num_rows)
        struct = TableStructure(
            region_id=region.region_id, html=html, num_rows=num_rows,
            num_cols=num_cols, has_header=bool(header_bands), confidence=0.7,
        )
        struct.cells = cells_data
        return struct
    except Exception as e:
        logger.warning("Page %d: TATR extraction failed (%s)", page_num, e)
        return None


def _assign_chars(pdf_path, page_num, grid, header_bands):
    """Assign pdfplumber chars to TATR cells by center-point containment (standard
    coords), replicating Stage-1's p{n}_c{idx} indexing for merged_from."""
    import pdfplumber

    def _in(box, x, y):
        return box[0] <= x <= box[2] and box[1] <= y <= box[3]

    cells_data = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        if page_num - 1 >= len(pdf.pages):
            return None
        page = pdf.pages[page_num - 1]
        chars = []  # (char_idx, cx_std, cy_std)
        for char_idx, ch in enumerate(page.chars or []):
            t = ch.get("text", "")
            if not t or t.isspace():
                continue
            x0, top, x1, bottom = ch["x0"], ch["top"], ch["x1"], ch["bottom"]
            if x1 - x0 < 0.1 or bottom - top < 0.1:
                continue
            cx = (x0 + x1) / 2 * _SCALE
            cy = (top + bottom) / 2 * _SCALE
            chars.append((char_idx, cx, cy, t))

        for r, row in enumerate(grid):
            is_header = any(_overlap_y(row[0], hb) for hb in header_bands) if header_bands else False
            for c, cell in enumerate(row):
                inside = [(ci, t) for ci, cx, cy, t in chars if _in(cell, cx, cy)]
                inside.sort(key=lambda z: z[0])
                text = "".join(t for _, t in inside).strip()
                merged_from = [f"p{page_num}_c{ci}" for ci, _ in inside]
                is_row_header = (c == 0 and not is_header and bool(text)
                                 and not _is_numeric(text))
                cells_data.append({
                    "row_idx": r, "col_idx": c, "is_header": is_header,
                    "is_row_header": is_row_header, "text": text,
                    "merged_from": merged_from, "bbox": cell,
                })
    return cells_data


def _overlap_y(a, b) -> bool:
    return not (a[3] < b[1] or a[1] > b[3])


def _is_numeric(text: str) -> bool:
    cleaned = text.strip().lstrip("$").replace(",", "").replace("(", "").replace(
        ")", "").replace("%", "").replace("-", "").strip()
    return not cleaned or cleaned.replace(".", "").isdigit()


def _build_html(cells_data, num_rows) -> str:
    rows_map: dict = {}
    for c in cells_data:
        rows_map.setdefault(c["row_idx"], []).append(c)
    parts = ["<table>"]
    for r in range(num_rows):
        if r not in rows_map:
            continue
        parts.append("  <tr>")
        for cell in sorted(rows_map[r], key=lambda x: x["col_idx"]):
            txt = cell["text"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            if cell["is_header"]:
                parts.append(f'    <th scope="col">{txt}</th>')
            elif cell["is_row_header"]:
                parts.append(f'    <th scope="row">{txt}</th>')
            else:
                parts.append(f"    <td>{txt}</td>")
        parts.append("  </tr>")
    parts.append("</table>")
    return "\n".join(parts)
