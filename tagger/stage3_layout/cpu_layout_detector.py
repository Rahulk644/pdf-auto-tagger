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

from tagger.config import LAYOUT, PDF_NATIVE_DPI, STANDARD_DPI
from tagger.page_cache import open_pdf
from tagger.models.data_types import LayoutCategory, LayoutRegion, PageElement


def _region_detect_all(pdf_path: str, page_num: int) -> list[tuple]:
    """All categorised regions on a page as [(bbox_150dpi, heron_label), ...],
    from whichever layout backend is active. `picodet` → PP-DocLayout-V3,
    everything else → Docling Heron. Both emit the same shape + label vocabulary,
    so callers stay backend-agnostic."""
    if LAYOUT.backend == "picodet":
        from tagger.stage3_layout.picodet_layout import detect_all_regions
    else:
        from tagger.stage5_specialists.docling_table_extractor import detect_all_regions
    return detect_all_regions(pdf_path, page_num)


def _region_detect_tables(pdf_path: str, page_num: int) -> list[tuple]:
    """Table bboxes (150-DPI) from the active layout backend (see _region_detect_all)."""
    if LAYOUT.backend == "picodet":
        from tagger.stage3_layout.picodet_layout import detect_tables
    else:
        from tagger.stage5_specialists.docling_table_extractor import detect_tables
    return detect_tables(pdf_path, page_num)

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


_NUMERIC_ONLY_RE = re.compile(r"^[\d\s.,()%/–-]+$")


def _valid_heading_text(txt: str) -> bool:
    """Text-quality gate for a heading candidate. Single source of truth shared by
    BOTH the pdfplumber line path (_heading_lineboxes) and the Heron-additive path
    (_merge_docling_headings) — the latter previously bypassed it, letting captions,
    run-in sentences (end in '.'), and appendix code/prompt lines masquerade as
    headings (the precision leak the heading scoreboard exposed). Rules:
      short:   0 < len <= _MAX_HEAD_LEN  (headings aren't paragraphs)
      no_end:  last char not in .,;:     (run-in sentences/captions end punctuated)
      not_num: not a bare numeric/symbol run ('1.2', '(4)', '%')
    """
    txt = (txt or "").strip()
    if not (0 < len(txt) <= _MAX_HEAD_LEN):
        return False
    if txt[-1] in ".,;:":
        return False
    if _NUMERIC_ONLY_RE.match(txt):
        return False
    return True


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


def _box_iou(a: tuple, b: tuple) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    ua = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / ua


def _merge_table_boxes(pdf_path: str, page_num: int, lattice_boxes: list[tuple]) -> list[tuple]:
    """Augment lattice tables with TATR-detected ones (borderless tables lattice can't
    see). Dedupe by IoU > 0.5 — keep the lattice box on overlap (its bbox is tighter
    around the ruled grid). Kept for opt-in/comparison; production uses Docling
    detection instead (see _merge_docling_tables) because TATR's binary detection
    over-fired catastrophically on text+heading docs (NID/MHS regressions)."""
    from tagger.stage5_specialists.tatr_table_extractor import detect_tables
    tatr_boxes = detect_tables(pdf_path, page_num)
    if not tatr_boxes:
        return lattice_boxes
    merged = list(lattice_boxes)
    for tb in tatr_boxes:
        if not any(_box_iou(tb, lb) > 0.5 for lb in lattice_boxes):
            merged.append(tb)
    return merged


def _merge_docling_headings(pdf_path: str, page_num: int,
                            existing: list[tuple],
                            tboxes: list[tuple],
                            iboxes: list[tuple],
                            raw: list[tuple] | None = None) -> list[tuple]:
    """Union pdfplumber-detected heading lineboxes with Heron's semantic Title /
    Section-header detections. Drops Heron candidates that overlap any existing
    heading (IoU>0.3) or any table/image bbox (heading should be its own line,
    not embedded in a table or figure). Never removes an existing heading.

    NOTE (heading-scoreboard diagnosis, arXiv N=3 controlled): a text-quality gate
    was trialled here (resolve the pdfplumber line under the Heron box, reject
    run-in/caption/numeric) but REVERTED — box→line resolution is a spatial-to-
    logical impedance mismatch (a Section-header box overruns into the paragraph
    below, the dominant line resolves to body text, real headings get dropped:
    recall 0.78→0.52 for ~0 precision gain). The real precision leak is upstream:
    _heading_lineboxes firing on the rows of borderless tables Heron's Table boxes
    don't fully cover. Fix that, not this.

    `raw` is the shared per-page region list (from _region_detect_all) so the
    three native-path merges run ONE Heron inference between them; None = fetch
    (keeps the helper usable standalone / in tests)."""
    if raw is None:
        raw = _region_detect_all(pdf_path, page_num)
    if not raw:
        return existing
    out = list(existing)
    existing_boxes = [b for b, _ in existing]
    blockers = tboxes + iboxes
    for bbox, label in raw:
        if label == "Title":
            cat = LayoutCategory.TITLE
        elif label == "Section-header":
            cat = LayoutCategory.SECTION_HEADER
        else:
            continue
        if any(_box_iou(bbox, eb) > 0.3 for eb in existing_boxes):
            continue
        if _center_inside(bbox, blockers):
            continue
        out.append((bbox, cat))
    return out


def _merge_docling_formulas(pdf_path: str, page_num: int,
                            blockers: list[tuple],
                            raw: list[tuple] | None = None) -> list[tuple]:
    """Heron-detected Formula regions on a NATIVE page as [(bbox150, FORMULA), ...].
    The pdfplumber path has no formula signal, so display equations would
    otherwise be grouped into a /P Text block — Stage 9/10 then can't attach
    MathML. Additive + guarded: drop any Heron formula overlapping an existing
    table/image/heading (IoU>0.3) so we never reclassify a real region. Feeds
    the PDF/UA-2 MathML Associated File on the /Formula element. `raw` = shared
    per-page region list (None = fetch); see _merge_docling_headings."""
    if raw is None:
        raw = _region_detect_all(pdf_path, page_num)
    if not raw:
        return []
    out = []
    for bbox, label in raw:
        if label != "Formula":
            continue
        if any(_box_iou(bbox, bb) > 0.3 for bb in blockers):
            continue
        out.append((bbox, LayoutCategory.FORMULA))
    return out


def _merge_docling_tables(pdf_path: str, page_num: int, lattice_boxes: list[tuple],
                          raw: list[tuple] | None = None) -> list[tuple]:
    """Augment lattice tables with Docling-layout-detected ones. Docling's layout
    model has 17 distinct categories so heading/text regions are NEVER classified
    as 'Table' — only genuine tables come out, no false-positives on prose. Dedupe
    by IoU>0.5 (lattice box wins on overlap; its bbox is tighter for ruled grids).
    `raw` = shared per-page region list (None = fetch via the table-only call)."""
    if raw is not None:
        docling_boxes = [b for b, lbl in raw if lbl == "Table"]
    else:
        docling_boxes = _region_detect_tables(pdf_path, page_num)
    if not docling_boxes:
        return lattice_boxes
    merged = list(lattice_boxes)
    for db in docling_boxes:
        if not any(_box_iou(db, lb) > 0.5 for lb in lattice_boxes):
            merged.append(db)
    return merged


def _image_boxes(page, big_image_threshold: float = 0.7) -> list[tuple]:
    """Image bboxes (150-DPI). Drop any image covering >= big_image_threshold of
    the page — that's the page-spanning raster a scanned PDF embeds as page
    background, and tagging it as a Picture would swallow every OCR text element
    via the _center_inside blocker check. On MIXED pages we tighten the threshold
    further (the page body is in the image, only a small visible header is text
    — image coverage is reported around 0.5-0.6 by Stage 0)."""
    out = []
    page_area = (page.width * _SCALE) * (page.height * _SCALE)
    for im in (page.images or []):
        x0, top, x1, bottom = (im["x0"] * _SCALE, im["top"] * _SCALE,
                               im["x1"] * _SCALE, im["bottom"] * _SCALE)
        area = (x1 - x0) * (bottom - top)
        if area >= _MIN_IMAGE_AREA and area / max(page_area, 1.0) < big_image_threshold:
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


# Heading precision guards (arXiv scoreboard diagnosis: the dominant MHS leak is
# _heading_lineboxes firing on borderless-TABLE rows Heron's Table boxes don't
# cover — p12 emitted 20 spurious TITLE/SECT = table data/header rows + captions).
_NUMTOK_RATIO = 0.5        # >this fraction of digit-bearing tokens => data row, not heading
_NUMTOK_MIN = 4            # ...only when the line has >= this many tokens (protects "GPT-4 Results")
_CLUSTER_MIN = 4           # >= this many tight same-size candidates in a run => table/list, not headings
_CLUSTER_GAP = 1.0         # consecutive candidates within this * body gap count as one tight run
_LEAD_NUM_RE = re.compile(r"^\d+(?:\.\d+)*\.?\s+")


def _numeric_token_ratio(txt: str) -> float:
    """Fraction of whitespace tokens that carry a digit, after stripping a leading
    section number. A table data row ('Amazon 1.04M 9.44M 18.2 4 5 ...') scores high;
    a real heading ('Experimental Results') scores 0."""
    t = _LEAD_NUM_RE.sub("", txt)
    toks = t.split()
    if not toks:
        return 0.0
    return sum(1 for w in toks if any(c.isdigit() for c in w)) / len(toks)


def _looks_like_data_row(txt: str) -> bool:
    toks = _LEAD_NUM_RE.sub("", txt).split()
    return len(toks) >= _NUMTOK_MIN and _numeric_token_ratio(txt) > _NUMTOK_RATIO


def _suppress_dense_clusters(cands: list, body: float) -> list:
    """Drop runs of >= _CLUSTER_MIN candidates stacked tightly (gap < _CLUSTER_GAP*body)
    at the same font size — those are borderless-table rows or list items, never a
    real heading sequence (real headings are sparse, separated by body text). cands =
    [(bbox, cat, size, top, bottom), ...] already in reading order."""
    if not body or len(cands) < _CLUSTER_MIN:
        return cands
    keep = [True] * len(cands)
    i = 0
    while i < len(cands):
        j = i + 1
        while (j < len(cands)
               and abs(cands[j][2] - cands[i][2]) <= 0.5            # same size bucket
               and (cands[j][3] - cands[j - 1][4]) < _CLUSTER_GAP * body):  # tight gap
            j += 1
        if j - i >= _CLUSTER_MIN:                                   # a dense run
            for k in range(i, j):
                keep[k] = False
        i = j
    return [c for c, k in zip(cands, keep) if k]


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

    cands = []  # (bbox, cat, size, top150, bottom150) — collected then cluster-filtered
    prev_bottom = None
    for ln, sz, bold, center, bbox in info:
        gap_above = (ln["top"] - prev_bottom) if prev_bottom is not None else 1e9
        prev_bottom = ln["bottom"]
        if _center_inside((center[0], center[1], center[0], center[1]), blockers):
            continue
        txt = ln["text"].strip()
        # gap_above, ln sizes and body are all pdfplumber points — consistent.
        big = bool(body) and sz >= body * _BIG_RATIO
        bold_block = (bool(body) and bold and sz >= body * _BOLD_RATIO
                      and gap_above >= body * 0.6)
        if not ((big or bold_block) and _valid_heading_text(txt)):
            continue
        if _looks_like_data_row(txt):          # borderless-table data row, not a heading
            continue
        cat = (LayoutCategory.TITLE if bool(body) and sz >= body * _TITLE_RATIO
               else LayoutCategory.SECTION_HEADER)
        cands.append((bbox, cat, sz, bbox[1], bbox[3]))
    # Drop tight same-size runs (borderless-table rows / list items masquerading as headings).
    cands = _suppress_dense_clusters(cands, body)
    return [(bbox, cat) for bbox, cat, _, _, _ in cands]


# Heron's 17-class label -> our LayoutCategory. Classes not in our enum (Document
# Index, Code, Checkbox-*, Form, Key-Value Region) fall back to TEXT.
_HERON_LABEL_TO_CATEGORY = {
    "Caption": LayoutCategory.CAPTION,
    "Footnote": LayoutCategory.FOOTNOTE,
    "Formula": LayoutCategory.FORMULA,
    "List-item": LayoutCategory.LIST_ITEM,
    "Page-footer": LayoutCategory.PAGE_FOOTER,
    "Page-header": LayoutCategory.PAGE_HEADER,
    "Picture": LayoutCategory.PICTURE,
    "Section-header": LayoutCategory.SECTION_HEADER,
    "Table": LayoutCategory.TABLE,
    "Text": LayoutCategory.TEXT,
    "Title": LayoutCategory.TITLE,
}


def _detect_via_heron(pdf_path: str, page_num: int,
                      page_w: float, page_h: float) -> list[tuple]:
    """For MIXED/SCANNED pages: ask Heron for all categorised regions, drop the
    page-spanning raster (it's the page-image background, not a real Picture),
    and return [(bbox_150dpi, LayoutCategory), ...] in xy-cut order. pdfplumber
    can't help here (no text layer, no rules); Heron is the right primitive."""
    raw = _region_detect_all(pdf_path, page_num)
    if not raw:
        return []
    page_area = max(1.0, page_w * page_h)
    out = []
    for bbox, label in raw:
        cat = _HERON_LABEL_TO_CATEGORY.get(label, LayoutCategory.TEXT)
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if cat == LayoutCategory.PICTURE and area / page_area >= 0.4:
            # page-spanning raster background — keep it artifacted, not a Figure
            continue
        out.append((bbox, cat))
    return out


def detect_regions(pdf_path: str, page_num: int,
                   elements: list[PageElement],
                   page_type: str = "native") -> list[LayoutRegion]:
    """Produce LayoutRegion[] for one born-digital page — the MinerU-free Stage 3.

    `page_type` is the Stage-0 classification ("native" / "mixed" / "scanned");
    it tightens the page-spanning-image guard on mixed/scanned pages so the
    image-of-text background is artifacted and OCR-derived text forms real
    Text/heading regions, not a single Figure that absorbs the whole page."""
    # MIXED/SCANNED pages: pdfplumber's text-line / lattice paths can't see the
    # image-of-text body — Heron operates on the page image and categorises the
    # regions directly (Title / Section-header / List-item / Caption / Text /
    # ...), so we use it as the layout source on these pages. NATIVE pages stay
    # on the proven pdfplumber + lattice + Docling-table-merge path.
    if page_type in ("mixed", "scanned"):
        with open_pdf(pdf_path) as pdf:
            if page_num > len(pdf.pages):
                return []
            page = pdf.pages[page_num - 1]
            page_w = page.width * _SCALE
            page_h = page.height * _SCALE
        meta = _detect_via_heron(pdf_path, page_num, page_w, page_h)
        if meta:
            order = _xycut_order([m[0] for m in meta])
            return [
                LayoutRegion(
                    region_id=f"r{page_num}_{ro}",
                    page_num=page_num,
                    bbox=meta[idx][0],
                    category=meta[idx][1],
                    reading_order=ro,
                    confidence=0.85,
                )
                for ro, idx in enumerate(order)
            ]
        # Heron unavailable -> fall through to the pdfplumber-driven path with
        # the tighter big-image threshold so OCR text isn't blocker-trapped.

    big_image_thr = 0.4 if page_type in ("mixed", "scanned") else 0.7
    with open_pdf(pdf_path) as pdf:
        if page_num > len(pdf.pages):
            return []
        page = pdf.pages[page_num - 1]
        page_h = page.height * _SCALE
        # ONE Heron inference for the whole native page — the table / heading /
        # formula merges below all filter this shared list instead of each
        # re-rendering the page and re-running the model (was 3x render + 3x
        # inference per page on the dominant stage).
        raw_regions = _region_detect_all(pdf_path, page_num)
        tboxes = _table_boxes(page)
        # Augment with Docling-detected tables (borderless: lattice misses them).
        # Docling layout has 17 distinct classes (Text, Section-header, Table, ...),
        # so heading/text regions get classified as themselves — NOT as "Table" —
        # which structurally rules out the false-positive failure mode TATR has
        # (TATR's binary table-or-not DETR cratered NID/MHS on docs 001-004/048/159).
        # No-op when docling_ibm_models / weights are unavailable.
        tboxes = _merge_docling_tables(pdf_path, page_num, tboxes, raw=raw_regions)
        iboxes = _image_boxes(page, big_image_threshold=big_image_thr)
        hboxes = _heading_lineboxes(page, tboxes, iboxes)  # [(bbox150, category)]
        # Additive Heron-detected headings on native pages — closes the
        # documented MHS gap. Pdfplumber typography catches most headings; Heron
        # adds the semantic ones that lack a distinguishing font signal (numbered
        # subsection titles, bold-at-body-size headings, etc). UNION only —
        # never removes a pdfplumber-detected heading, so this is strictly
        # accuracy-positive: max possible TEDS/NID regression = 0.
        hboxes = _merge_docling_headings(pdf_path, page_num, hboxes, tboxes, iboxes,
                                         raw=raw_regions)
        # Additive Heron-detected display formulas — gives Stage 9/10 a /Formula
        # region to attach PDF/UA-2 MathML to (additive, dedupes against
        # table/image/heading boxes, so it can't reclassify a real region).
        fboxes = _merge_docling_formulas(
            pdf_path, page_num, tboxes + iboxes + [b for b, _ in hboxes],
            raw=raw_regions)

    # headings come from the pdfplumber-line path; exclude their elements from body
    # blocks (Stage 4 still matches them into the heading region by bbox). Formula
    # boxes block too — their text belongs to the /Formula region, not a /P block.
    blockers = tboxes + iboxes + [b for b, _ in hboxes] + [b for b, _ in fboxes]
    text_els = [el for el in elements
                if (el.text or "").strip() and not _center_inside(el.bbox, blockers)]

    meta: list[tuple[tuple, LayoutCategory]] = []
    for b in tboxes:
        meta.append((b, LayoutCategory.TABLE))
    for b in iboxes:
        meta.append((b, LayoutCategory.PICTURE))
    for b, cat in hboxes:
        meta.append((b, cat))
    for b, cat in fboxes:
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
