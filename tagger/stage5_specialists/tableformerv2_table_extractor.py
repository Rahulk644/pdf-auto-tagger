"""TableFormerV2 (docling-ibm-models, in-env) as a pipeline table engine.

Generative OTSL structure model: image -> OTSL token sequence + one predicted
bbox per cell token (normalized to the crop). We parse OTSL -> grid (row/col +
header), de-normalize the per-cell bbox to page 150-DPI, and fill cell TEXT from
NATIVE pdfplumber chars (exact + merged_from MCID mapping) — same accessibility
wiring as the rapid_table path. Selected via TABLE.engine == "tableformerv2".

Loads from docling-ibm-models (no new deps, no isolated venv). Self-gates to None
on any failure so the cascade falls back to TableFormer v1.
"""
from __future__ import annotations

import logging

from tagger.models.data_types import TableStructure
from tagger.stage5_specialists.docling_table_extractor import _build_html
from tagger.stage5_specialists.slanet_table_extractor import _native_chars

logger = logging.getLogger(__name__)

_model = None
_tok = None
_tf = None
_failed = False

_CELL_TOKENS = {"fcel", "ched", "rhed", "srow", "ecel", "lcel", "ucel", "xcel"}


def _load() -> bool:
    global _model, _tok, _tf, _failed
    if _model is not None or _failed:
        return _model is not None
    try:
        import torch  # noqa: F401
        from torchvision import transforms
        from transformers import AutoTokenizer
        from docling_ibm_models.tableformer_v2.model import TableFormerV2
        _model = TableFormerV2.from_pretrained("docling-project/TableFormerV2").eval()
        _tok = AutoTokenizer.from_pretrained("docling-project/TableFormerV2")
        _tf = transforms.Compose([
            transforms.Resize((448, 448)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        logger.info("TableFormerV2 (OTSL) table extractor loaded")
        return True
    except Exception as e:
        _failed = True
        logger.info("TableFormerV2 unavailable (%s) — table engine stays TableFormer", e)
        return False


def extract_table_tableformerv2(pdf_path, page_num: int, region):
    if not _load():
        return None
    try:
        import torch
        from tagger.page_cache import render_page

        img = render_page(pdf_path, page_num)
        if img is None:
            return None
        rx0, ry0, rx1, ry1 = (int(v) for v in region.bbox)
        crop = img.crop((rx0, ry0, rx1, ry1))
        cw, ch = max(1, rx1 - rx0), max(1, ry1 - ry0)
        with torch.no_grad():
            out = _model.generate(_tf(crop).unsqueeze(0), _tok, max_length=512)
        toks = _tok.convert_ids_to_tokens(out["generated_ids"][0].tolist())
        bboxes = out["predicted_bboxes"][0]

        chars = _native_chars(pdf_path, page_num)
        cells = []
        row = col = bi = 0
        for t in toks:
            s = t.strip("<>")
            if s == "nl":
                row += 1; col = 0; continue
            if s not in _CELL_TOKENS:
                continue
            bb = bboxes[bi] if bi < len(bboxes) else None
            bi += 1
            text, mf = "", []
            if bb is not None and chars:
                nx0, ny0, nx1, ny1 = (float(v) for v in bb[:4])
                bx0, by0 = nx0 * cw + rx0, ny0 * ch + ry0
                bx1, by1 = nx1 * cw + rx0, ny1 * ch + ry0
                inside = [(ci, tx) for ci, cx, cy, tx in chars if bx0 <= cx <= bx1 and by0 <= cy <= by1]
                inside.sort(key=lambda z: z[0])
                text = "".join(tx for _, tx in inside).strip()
                mf = [f"p{page_num}_c{ci}" for ci, _ in inside]
            cells.append({
                "row_idx": row, "col_idx": col,
                "is_header": (s in ("ched", "rhed")) or row == 0,
                "is_row_header": (col == 0 and row > 0),
                "text": text, "merged_from": mf, "bbox": None,
            })
            col += 1

        cells = [c for c in cells if c["text"]]
        if len(cells) < 2:
            return None
        num_rows = 1 + max(c["row_idx"] for c in cells)
        num_cols = 1 + max(c["col_idx"] for c in cells)
        struct = TableStructure(
            region_id=region.region_id, html=_build_html(cells, num_rows),
            num_rows=num_rows, num_cols=num_cols,
            has_header=any(c["is_header"] for c in cells), confidence=0.75)
        struct.cells = cells
        return struct
    except Exception as e:
        logger.warning("TableFormerV2 extraction failed on p%d: %s", page_num, e)
        return None
