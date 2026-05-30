"""PP-DocLayout-V3 region detector — a drop-in alternative to Docling Heron.

This is the `picodet` layout backend's region source. It runs PaddlePaddle's
PP-DocLayout-V3 (an RT-DETR-family object detector, ~33M params) via plain
HuggingFace `transformers` — NO paddlepaddle runtime, NO GPU required — and
returns regions in the *exact same shape and label vocabulary as Heron*
(`docling_table_extractor.detect_all_regions`):

    [(bbox_150dpi, heron_label_string), ...]

so the CPU layout detector consumes it unchanged. The whole point of the
backend swap is an A/B on the layout DETECTOR only: TableFormer (table
structure), the pdfplumber heading/lattice path, alt-text, and Stages 4-10 are
all identical between `cpu` (Heron) and `picodet`. The gate is dp-bench MHS —
Heron-additive headings on native pages are what closed the heading gap to the
GPU pipeline, so this path only ships if MHS holds.

Model: `PaddlePaddle/PP-DocLayoutV3_safetensors` (Apache-2.0). Renders pages at
STANDARD_DPI via PyMuPDF so emitted pixel coords are already 150-DPI standard
space — no transform downstream, identical to Heron's `detect_all_regions`.
On any failure the public functions return [] so Stage 3 degrades gracefully.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# PP-DocLayout-V3's raw class names -> Heron's 17-class label strings, so the
# output is a drop-in for docling_table_extractor.detect_all_regions and the
# existing _HERON_LABEL_TO_CATEGORY / _merge_docling_headings logic applies
# unchanged. Labels with no Heron cousin fall back to "Text".
_PP_TO_HERON = {
    "doc_title": "Title",
    "paragraph_title": "Section-header",
    "figure_title": "Caption",
    "image": "Picture",
    "chart": "Picture",
    "seal": "Picture",
    "table": "Table",
    "formula": "Formula",
    "header": "Page-header",
    "footer": "Page-footer",
    "footnote": "Footnote",
    "vision_footnote": "Footnote",
    # everything textual -> Text
    "text": "Text",
    "abstract": "Text",
    "content": "Text",
    "aside_text": "Text",
    "reference": "Text",
    "reference_content": "Text",
    "number": "Text",
    "formula_number": "Text",
    "algorithm": "Text",
}

_MODEL_NAME = "PaddlePaddle/PP-DocLayoutV3_safetensors"
_CONF_THRESHOLD = 0.5

# Lazy singletons — load once, reuse. None = not yet attempted; False = failed.
_proc = None
_model = None
_id2label: dict[int, str] | None = None


def _load() -> bool:
    """Load the PP-DocLayout-V3 image processor + model once. Returns False (and
    logs) if transformers/torch or the weights are unavailable — the caller then
    falls back to an empty region list."""
    global _proc, _model, _id2label
    if _model is False:
        return False
    if _model is not None:
        return True
    try:
        import torch  # noqa: F401
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        proc = AutoImageProcessor.from_pretrained(_MODEL_NAME)
        model = AutoModelForObjectDetection.from_pretrained(_MODEL_NAME)
        model.eval()
        _proc = proc
        _model = model
        _id2label = model.config.id2label
        return True
    except Exception as e:  # pragma: no cover - exercised only without weights
        logger.warning("PP-DocLayout-V3 unavailable, picodet backend no-op: %s", e)
        _model = False
        return False


def detect_all_regions(pdf_path, page_num: int) -> list[tuple]:
    """All PP-DocLayout-V3 regions on a page as [(bbox_150dpi, heron_label), ...].

    Mirrors docling_table_extractor.detect_all_regions exactly (same coord space,
    same label vocabulary) so the CPU detector is backend-agnostic."""
    if not _load():
        return []
    try:
        import torch
        from tagger.page_cache import render_page

        img = render_page(pdf_path, page_num)
        if img is None:
            return []

        inputs = _proc(images=[img], return_tensors="pt")
        with torch.no_grad():
            outputs = _model(**inputs)
        # post_process expects (height, width)
        results = _proc.post_process_object_detection(
            outputs, target_sizes=[img.size[::-1]], threshold=_CONF_THRESHOLD)[0]

        out = []
        for score, label_id, box in zip(
                results["scores"], results["labels"], results["boxes"]):
            raw = _id2label.get(int(label_id.item()), "text")
            heron_label = _PP_TO_HERON.get(raw, "Text")
            l, t, r, b = (float(v) for v in box.tolist())
            out.append(((l, t, r, b), heron_label))
        return out
    except Exception as e:
        logger.warning("PP-DocLayout-V3 failed on page %d: %s", page_num, e)
        return []


def detect_tables(pdf_path, page_num: int) -> list[tuple]:
    """Table bboxes only (150-DPI) — the filtered subset of detect_all_regions,
    matching docling_table_extractor.detect_tables."""
    return [bbox for bbox, label in detect_all_regions(pdf_path, page_num)
            if label == "Table"]
