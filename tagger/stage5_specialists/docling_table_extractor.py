"""Stage 5d — Borderless/complex table extraction via Docling (IBM, MIT).

Replaces the TATR experiment with the right architecture:

- **Docling layout (heron, RT-DETR)** has 17 distinct categories — Caption, Footnote,
  Formula, List-item, Page-footer/header, Picture, Section-header, Table, Text,
  Title, ... — so a heading is classified `Section-header`, a paragraph is `Text`;
  table is its OWN class and won't absorb prose. (TATR's binary 'table/not-table'
  DETR head fundamentally over-fired on dense text+heading layouts -> docs 001/002/
  003 MHS 0.98 -> 0.0 even at threshold 0.8. The architectural fix is "more classes",
  not "more threshold".)
- **TableFormer (transformer encoder-decoder)** infers SEMANTIC table structure, so
  it works on borderless tables where TATR/DETR's visual-line dependency breaks.

Self-gating: if torch / docling_ibm_models / the model artifacts are missing, every
public function returns None / [] and the pipeline falls back to pdfplumber.
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

_LAYOUT_REPO = "ds4sd/docling-layout-heron"
_TF_REPO = "ds4sd/docling-models"
_TF_SUBDIR = "model_artifacts/tableformer/accurate"
_SCALE = STANDARD_DPI / PDF_NATIVE_DPI
_LAYOUT_THRESH = 0.5  # base_threshold; we keep our own per-class filtering below

_layout = None  # docling LayoutPredictor singleton
_tf = None      # docling TFPredictor singleton
_layout_failed = False
_tf_failed = False


def _load_layout():
    global _layout, _layout_failed
    if _layout is not None or _layout_failed:
        return _layout is not None
    try:
        from huggingface_hub import snapshot_download
        from docling_ibm_models.layoutmodel.layout_predictor import LayoutPredictor
        path = snapshot_download(_LAYOUT_REPO)
        _layout = LayoutPredictor(path, device="cpu", num_threads=4,
                                  base_threshold=_LAYOUT_THRESH)
        logger.info("Docling layout (%s) loaded", _LAYOUT_REPO)
        return True
    except Exception as e:
        _layout_failed = True
        logger.info("Docling layout unavailable (%s)", e)
        return False


def _load_tf():
    global _tf, _tf_failed
    if _tf is not None or _tf_failed:
        return _tf is not None
    try:
        import json
        import os
        from huggingface_hub import snapshot_download
        from docling_ibm_models.tableformer.data_management.tf_predictor import TFPredictor
        root = snapshot_download(_TF_REPO, allow_patterns=[f"{_TF_SUBDIR}/*"])
        tf_dir = os.path.join(root, _TF_SUBDIR)
        cfg_path = os.path.join(tf_dir, "tm_config.json")
        with open(cfg_path) as fh:
            cfg = json.load(fh)
        # The config has relative weight paths; point them at the cached snapshot.
        cfg.setdefault("model", {})["save_dir"] = tf_dir
        _tf = TFPredictor(cfg, device="cpu", num_threads=4)
        logger.info("Docling TableFormer (accurate) loaded")
        return True
    except Exception as e:
        _tf_failed = True
        logger.info("Docling TableFormer unavailable (%s)", e)
        return False


def detect_tables(pdf_path: "str | Path", page_num: int) -> list[tuple]:
    """Detect TABLE regions on a page via Docling's layout model. Returns bboxes in
    150-DPI standard coords. Filters to the 'Table' class — other regions (Text,
    Section-header, ...) are NOT returned, so dense-prose pages don't get table
    false-positives the way TATR's binary detection produced."""
    out = []
    for bbox, label in detect_all_regions(pdf_path, page_num):
        if label == "Table":
            out.append(bbox)
    return out


def detect_all_regions(pdf_path: "str | Path", page_num: int) -> list[tuple]:
    """All Heron-detected regions on a page: [(bbox_150dpi, label_string), ...].
    Labels are from Docling's 17-class set (Caption, Footnote, Formula, List-item,
    Page-footer, Page-header, Picture, Section-header, Table, Text, Title, ...).
    Used by the CPU layout detector on MIXED/SCANNED pages where the pdfplumber
    text-line path can't see anything — Heron operates on the page image and
    categorises directly."""
    if not _load_layout():
        return []
    try:
        from tagger.page_cache import render_page
        img = render_page(pdf_path, page_num)
        if img is None:
            return []
        out = []
        for det in _layout.predict(img):
            bbox = (float(det["l"]), float(det["t"]),
                    float(det["r"]), float(det["b"]))
            out.append((bbox, det.get("label")))
        return out
    except Exception as e:
        logger.warning("Docling layout failed on page %d: %s", page_num, e)
        return []


def extract_table(pdf_path: "str | Path", page_num: int,
                  region: LayoutRegion,
                  classification: PageClassification) -> TableStructure | None:
    """Parse a table region with Docling TableFormer (semantic, handles borderless).
    Returns None on any failure so Stage 5 falls back to pdfplumber."""
    if classification.page_type == PageType.SCANNED:
        return None
    if not _load_tf():
        return None
    try:
        rx0, ry0, rx1, ry1 = region.bbox
        from tagger.page_cache import render_page
        page_img = render_page(pdf_path, page_num)  # shared with Heron (same page)
        if page_img is None:
            return None

        # TFPredictor.multi_table_predict expects an `iocr_page` dict (with `image`
        # = numpy ndarray, and `tokens` = OCR word cells) plus a list of table_bboxes.
        # We have native PDF chars (no OCR needed); pdfplumber word boxes in 150-DPI
        # image space act as the OCR cells and the post-processor matches them to
        # TableFormer's predicted cells.
        import numpy as np
        ocr_cells = _native_word_cells(pdf_path, page_num)
        img_arr = np.array(page_img)
        iocr_page = {
            "image": img_arr,
            "width": img_arr.shape[1],
            "height": img_arr.shape[0],
            "tokens": ocr_cells,
        }
        bbox_pix = [rx0, ry0, rx1, ry1]
        results = _tf.multi_table_predict(iocr_page, [bbox_pix], do_matching=True)
        if not results:
            return None
        td = results[0]  # one table
        cells_data = _build_cells_from_tf(td, pdf_path, page_num)
        if not cells_data:
            return None
        num_rows = 1 + max(c["row_idx"] for c in cells_data)
        num_cols = 1 + max(c["col_idx"] for c in cells_data)
        html = _build_html(cells_data, num_rows)
        struct = TableStructure(
            region_id=region.region_id, html=html, num_rows=num_rows,
            num_cols=num_cols,
            has_header=any(c.get("is_header") for c in cells_data),
            confidence=0.75,
        )
        struct.cells = cells_data
        return struct
    except Exception as e:
        logger.warning("Page %d: Docling TableFormer failed (%s)", page_num, e)
        return None


def _native_word_cells(pdf_path, page_num):
    """pdfplumber word boxes in 150-DPI standard coords, as Docling 'OCR cells'.
    Each token needs an `id`; CellMatcher rejects the dict without it."""
    from tagger.page_cache import open_pdf
    out = []
    with open_pdf(pdf_path) as pdf:
        if page_num - 1 >= len(pdf.pages):
            return out
        page = pdf.pages[page_num - 1]
        for i, w in enumerate(page.extract_words()):
            out.append({
                "id": i,
                "bbox": [w["x0"] * _SCALE, w["top"] * _SCALE,
                         w["x1"] * _SCALE, w["bottom"] * _SCALE],
                "text": w.get("text", ""),
            })
    return out


def _build_cells_from_tf(td, pdf_path, page_num):
    """Convert TableFormer's tf_responses into our cells_data shape, populating
    text + merged_from by char-center matching against pdfplumber (TableFormer
    outputs cell bboxes + row/column indices but NOT text)."""
    from tagger.page_cache import open_pdf
    table_cells = td.get("tf_responses") or []
    if not table_cells:
        return []

    chars = []  # (char_idx, cx_std, cy_std, text)
    with open_pdf(pdf_path) as pdf:
        if page_num - 1 >= len(pdf.pages):
            return []
        page = pdf.pages[page_num - 1]
        for ci, ch in enumerate(page.chars or []):
            t = ch.get("text", "")
            if not t or t.isspace():
                continue
            x0, top, x1, bottom = ch["x0"], ch["top"], ch["x1"], ch["bottom"]
            if x1 - x0 < 0.1 or bottom - top < 0.1:
                continue
            cx = (x0 + x1) / 2 * _SCALE
            cy = (top + bottom) / 2 * _SCALE
            chars.append((ci, cx, cy, t))

    def _bbox_of(cell):
        b = cell.get("bbox")
        if isinstance(b, dict):
            return (float(b.get("l", 0)), float(b.get("t", 0)),
                    float(b.get("r", 0)), float(b.get("b", 0)))
        if isinstance(b, (list, tuple)) and len(b) >= 4:
            return tuple(float(v) for v in b[:4])
        return (0.0, 0.0, 0.0, 0.0)

    cells_data = []
    for c in table_cells:
        r = int(c.get("row_id", c.get("start_row_offset_idx", 0)))
        col = int(c.get("column_id", c.get("start_col_offset_idx", 0)))
        # cell_class 2 = column header per TableFormer's label vocabulary
        is_header = c.get("cell_class") == 2 or str(c.get("label", "")).lower() == "ched"
        bbox = _bbox_of(c)
        inside = [(ci, t) for ci, cx, cy, t in chars
                  if bbox[0] <= cx <= bbox[2] and bbox[1] <= cy <= bbox[3]]
        inside.sort(key=lambda z: z[0])
        text = "".join(t for _, t in inside).strip()
        merged_from = [f"p{page_num}_c{ci}" for ci, _ in inside]
        is_row_header = (col == 0 and not is_header and bool(text)
                         and not _is_numeric(text))
        cells_data.append({
            "row_idx": r, "col_idx": col,
            "is_header": is_header, "is_row_header": is_row_header,
            "text": text, "merged_from": merged_from, "bbox": bbox,
        })
    return cells_data


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
