"""CPU-native Stage-3 layout detection — a GPU-free drop-in for MinerU.

Derives LayoutRegion[] from data already in the pipeline (Stage-2 PageElements:
text + bbox + font) plus pdfplumber primitives (ruled lines -> tables, images ->
pictures). No VLM, no GPU, no page-image rendering.

SCOPE: born-digital PDFs only. A scanned page has no text layer, so pdfplumber
yields nothing — Stage 0 classifies those, and they still need MinerU/OCR. This
backend resolves the GPU dependency for the dominant born-digital case.

Region typing:
  - Table   : pdfplumber lattice (gated ruled grid >=2 rows / >=4 ruled cells,
              rejecting >=85%-empty over-segmentation — mirrors Stage-5)
  - Picture : page.images above a min area
  - Title / Section-header : font-size tier OR bold-that-starts-a-block (the
              principled rule proven in scratch/cpu_extract.py — precision over recall)
  - Page-header / Page-footer : a short line in the top/bottom margin band
  - Text    : everything else; body lines grouped into blocks by column + vgap
Reading order: XY-cut (recursive largest-clean-gap) over the region boxes.

All bboxes are emitted in 150-DPI standard space (PageElement.bbox is already
150-DPI; pdfplumber 72-DPI lines/images are scaled by STANDARD_DPI/PDF_NATIVE_DPI).
"""
from __future__ import annotations

import re
import statistics

import pdfplumber

from tagger.config import PDF_NATIVE_DPI, STANDARD_DPI
from tagger.models.data_types import LayoutCategory, LayoutRegion, PageElement

_SCALE = STANDARD_DPI / PDF_NATIVE_DPI
_BIG_RATIO = 1.15      # heading: clearly larger than body text
_BOLD_RATIO = 1.00     # heading: bold and >= body size
_TITLE_RATIO = 1.5     # >= this * body => TITLE (H1) vs SECTION_HEADER
_MAX_HEAD_LEN = 100
_MIN_IMAGE_AREA = 2500.0   # 150-DPI px^2; drop hairline rules / tiny glyph images
_MARGIN_FRAC = 0.07        # top/bottom 7% band => candidate header/footer


# ---- XY-cut reading order (largest-clean-gap; handles header + columns) ----
def _gaps(intervals):
    ints = sorted(intervals)
    merged = [list(ints[0])]
    for lo, hi in ints[1:]:
        if lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [(merged[i + 1][0] - merged[i][1], (merged[i][1] + merged[i + 1][0]) / 2.0)
            for i in range(len(merged) - 1)]


def _xycut_order(boxes):
    order: list[int] = []

    def rec(idx):
        if len(idx) <= 1:
            order.extend(idx)
            return
        xg = _gaps([(boxes[i][0], boxes[i][2]) for i in idx])
        yg = _gaps([(boxes[i][1], boxes[i][3]) for i in idx])
        bx = max(xg, default=(0.0, None))
        by = max(yg, default=(0.0, None))
        if bx[0] <= 0 and by[0] <= 0:
            order.extend(sorted(idx, key=lambda i: (boxes[i][1], boxes[i][0])))
            return
        if by[0] >= bx[0]:
            c = by[1]
            rec([i for i in idx if (boxes[i][1] + boxes[i][3]) / 2 < c])
            rec([i for i in idx if (boxes[i][1] + boxes[i][3]) / 2 >= c])
        else:
            c = bx[1]
            rec([i for i in idx if (boxes[i][0] + boxes[i][2]) / 2 < c])
            rec([i for i in idx if (boxes[i][0] + boxes[i][2]) / 2 >= c])

    rec(list(range(len(boxes))))
    return order


def _center_inside(box, others) -> bool:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return any(o[0] <= cx <= o[2] and o[1] <= cy <= o[3] for o in others)


def _body_size(sizes) -> float:
    rounded = [round(s * 2) / 2 for s in sizes if s]
    return statistics.mode(rounded) if rounded else 0.0


def _table_boxes(page) -> list[tuple]:
    out = []
    for t in page.find_tables(table_settings={"vertical_strategy": "lines",
                                              "horizontal_strategy": "lines"}):
        ruled = [c for c in (t.cells or []) if c]
        if not (t.rows and len(t.rows) >= 2 and len(ruled) >= 4):
            continue
        rows = t.extract() or []
        cells = [c for r in rows for c in r]
        if cells and sum(1 for c in cells if not (c and str(c).strip())) / len(cells) >= 0.85:
            continue  # over-segmentation guard (mirrors Stage-5 table_extractor)
        x0, top, x1, bottom = t.bbox
        out.append((x0 * _SCALE, top * _SCALE, x1 * _SCALE, bottom * _SCALE))
    return out


def _image_boxes(page) -> list[tuple]:
    out = []
    for im in (page.images or []):
        x0, top, x1, bottom = (im["x0"] * _SCALE, im["top"] * _SCALE,
                               im["x1"] * _SCALE, im["bottom"] * _SCALE)
        if (x1 - x0) * (bottom - top) >= _MIN_IMAGE_AREA:
            out.append((x0, top, x1, bottom))
    return out


def _classify_margin(el: PageElement, page_h: float) -> LayoutCategory | None:
    """Page-header/footer if a short line sits in the top/bottom margin band, else None.
    Heading detection is done on pdfplumber lines (see _heading_lineboxes), NOT here —
    Stage-2 elements split numbered headings ('7 ' off 'Variants ...') and shift the
    gap-above signal, which lost bold/body-size headings; the line path keeps them whole."""
    txt = (el.text or "").strip()
    if not txt or len(txt) > _MAX_HEAD_LEN:
        return None
    top, bottom = el.bbox[1], el.bbox[3]
    band = page_h * _MARGIN_FRAC
    if bottom <= band:
        return LayoutCategory.PAGE_HEADER
    if top >= page_h - band:
        return LayoutCategory.PAGE_FOOTER
    return None


def _line_size_bold(ln) -> tuple[float, bool]:
    chars = ln.get("chars", [])
    sizes = [c.get("size") for c in chars if c.get("size")]
    sz = statistics.median(sizes) if sizes else 0.0
    fonts = [(c.get("fontname", "") or "").lower() for c in chars]
    bold = bool(fonts) and sum(
        1 for f in fonts if any(k in f for k in ("bold", "black", "heavy", "semibold"))
    ) >= 0.6 * len(fonts)
    return sz, bold


def _heading_lineboxes(page, tboxes, iboxes) -> list[tuple]:
    """Detect headings on pdfplumber's extract_text_lines (cpu_extract's proven path,
    MHS ~= GPU) and return [(bbox_150dpi, category)]. Keeps numbered/bold-body-size
    headings whole, unlike Stage-2 element segmentation."""
    lines = sorted(page.extract_text_lines(), key=lambda l: (l["top"], l["x0"]))
    blockers = tboxes + iboxes
    info = []  # (line, size, bold, center150, bbox150)
    for ln in lines:
        sz, bold = _line_size_bold(ln)
        bbox = (ln["x0"] * _SCALE, ln["top"] * _SCALE, ln["x1"] * _SCALE, ln["bottom"] * _SCALE)
        info.append((ln, sz, bold, ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2), bbox))
    body = _body_size([sz for ln, sz, _, c, _ in info if sz and not _center_inside(
        (c[0], c[1], c[0], c[1]), blockers)])

    out = []
    prev_bottom = None
    for ln, sz, bold, center, bbox in info:
        gap_above = (ln["top"] - prev_bottom) if prev_bottom is not None else 1e9
        prev_bottom = ln["bottom"]
        if _center_inside((center[0], center[1], center[0], center[1]), blockers):
            continue
        txt = ln["text"].strip()
        short = 0 < len(txt) <= _MAX_HEAD_LEN
        no_end = bool(txt) and txt[-1] not in ".,;:"
        not_num = not re.match(r"^[\d\s.,()%/–-]+$", txt)
        # gap_above, ln sizes and body are all pdfplumber points — consistent.
        big = bool(body) and sz >= body * _BIG_RATIO
        bold_block = (bool(body) and bold and sz >= body * _BOLD_RATIO
                      and gap_above >= body * 0.6)
        if (big or bold_block) and short and no_end and not_num:
            cat = (LayoutCategory.TITLE if bool(body) and sz >= body * _TITLE_RATIO
                   else LayoutCategory.SECTION_HEADER)
            out.append((bbox, cat))
    return out


def detect_regions(pdf_path: str, page_num: int,
                   elements: list[PageElement]) -> list[LayoutRegion]:
    """Produce LayoutRegion[] for one born-digital page — the MinerU-free Stage 3."""
    with pdfplumber.open(pdf_path) as pdf:
        if page_num > len(pdf.pages):
            return []
        page = pdf.pages[page_num - 1]
        page_h = page.height * _SCALE
        tboxes = _table_boxes(page)
        iboxes = _image_boxes(page)
        hboxes = _heading_lineboxes(page, tboxes, iboxes)  # [(bbox150, category)]

    # headings come from the pdfplumber-line path; exclude their elements from body
    # blocks (Stage 4 still matches them into the heading region by bbox).
    blockers = tboxes + iboxes + [b for b, _ in hboxes]
    text_els = [el for el in elements
                if (el.text or "").strip() and not _center_inside(el.bbox, blockers)]

    meta: list[tuple[tuple, LayoutCategory]] = []
    for b in tboxes:
        meta.append((b, LayoutCategory.TABLE))
    for b in iboxes:
        meta.append((b, LayoutCategory.PICTURE))
    for b, cat in hboxes:
        meta.append((b, cat))

    # body text: header/footer by margin, else group consecutive lines into Text blocks
    text_els.sort(key=lambda e: (e.bbox[1], e.bbox[0]))
    block: list[PageElement] = []

    def flush():
        if block:
            x0 = min(e.bbox[0] for e in block)
            y0 = min(e.bbox[1] for e in block)
            x1 = max(e.bbox[2] for e in block)
            y1 = max(e.bbox[3] for e in block)
            meta.append(((x0, y0, x1, y1), LayoutCategory.TEXT))
            block.clear()

    prev: PageElement | None = None
    for el in text_els:
        gap_above = (el.bbox[1] - prev.bbox[3]) if prev is not None else 1e9
        cat = _classify_margin(el, page_h)
        if cat is not None:
            flush()
            meta.append((el.bbox, cat))
            prev = None
            continue
        if prev is not None:
            line_h = max(el.bbox[3] - el.bbox[1], 1.0)
            same_col = not (el.bbox[2] < prev.bbox[0] or el.bbox[0] > prev.bbox[2])
            if gap_above > 1.6 * line_h or not same_col:
                flush()
        block.append(el)
        prev = el
    flush()

    if not meta:
        return []
    order = _xycut_order([m[0] for m in meta])
    regions: list[LayoutRegion] = []
    for ro, idx in enumerate(order):
        bbox, cat = meta[idx]
        regions.append(LayoutRegion(
            region_id=f"r{page_num}_{ro}",
            page_num=page_num,
            bbox=bbox,
            category=cat,
            reading_order=ro,
            confidence=0.9,
        ))
    return regions
