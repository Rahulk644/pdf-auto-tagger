"""
Stage 10 — Struct tree writeback via pikepdf.

Writes the auto-tagger's output into the PDF's structure tree,
making the document accessible per PDF/UA.

Two modes:
  V1 (re-tag): Modifies existing tagged PDF's struct tree entries
  V2 (full):   Builds struct tree from scratch for untagged PDFs

pikepdf Dictionary() takes string keys with '/' prefix.
Name objects are used only for values (e.g., Name.Document).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pikepdf
from pikepdf import Dictionary, Name, Array, String

from tagger.config import WRITEBACK
from tagger.models.data_types import PDFTag, TaggedElement
from tagger.stage1_extraction.coord_transformer import pdf_to_standard

logger = logging.getLogger(__name__)


def pdf_is_signed(pdf_path) -> bool:
    """True if the PDF carries a digital signature (AcroForm /SigFlags bit 1, a /Sig
    field with a /V value, or a /Perms transform dict).

    Stage 10 rewrites content streams and the struct tree — a destructive change that
    invalidates any cryptographic signature, which for legal/contractual documents is a
    correctness failure, not an accessibility win. So we must NOT modify a signed
    document. Detection only; the caller decides to skip and emit the original unchanged.
    """
    try:
        with pikepdf.open(str(pdf_path)) as pdf:
            root = pdf.Root
            if "/Perms" in root:
                return True
            af = root.get("/AcroForm")
            if af is not None:
                sf = af.get("/SigFlags")
                if sf is not None and (int(sf) & 1):
                    return True
                for f in (af.get("/Fields", []) or []):
                    if str(f.get("/FT", "")) == "/Sig" and f.get("/V") is not None:
                        return True
    except Exception:
        return False
    return False


def _embed_mathml_af(pdf, mathml: str, idx: int):
    """Embed a MathML string as a PDF 2.0 Associated File and return the
    indirect /Filespec. Used to attach machine-readable maths to a /Formula
    structure element (`/AF`, relationship /Supplement) for PDF/UA-2.

    The embedded-file stream's /Subtype is the MIME type application/mathml+xml
    (slash encoded #2F per the Name syntax). The caller links it from the struct
    element's /AF array AND should register it in the catalog /AF array."""
    ef_stream = pdf.make_stream(mathml.encode("utf-8"))
    ef_stream["/Type"] = Name.EmbeddedFile
    ef_stream["/Subtype"] = Name("/application#2Fmathml+xml")
    fname = f"formula_{idx}.mml"
    filespec = pdf.make_indirect(Dictionary({
        "/Type": Name.Filespec,
        "/F": String(fname),
        "/UF": String(fname),
        "/AFRelationship": Name("/Supplement"),
        "/Desc": String("MathML representation of the formula"),
        "/EF": Dictionary({"/F": ef_stream, "/UF": ef_stream}),
    }))
    return filespec


def _is_numeric_content(text: str) -> bool:
    """Return True if text is empty or contains only numeric/currency content."""
    if not text:
        return True
    cleaned = text.strip().lstrip("$").replace(",", "").replace("(", "").replace(")", "").replace("%", "").replace("-", "").strip()
    return not cleaned or cleaned.replace(".", "").isdigit()

def retag_existing_pdf(
    input_path: str | Path,
    output_path: str | Path,
    tagged_elements: list[TaggedElement],
) -> dict:
    """
    V1: Re-tag an existing tagged PDF by modifying struct tree entries.

    Walks the existing structure tree, matches elements by MCID,
    and updates the /S (structure type) entries.
    """
    stats = {
        "total_struct_elems": 0,
        "matched": 0,
        "changed": 0,
        "unmatched": 0,
    }

    # Build MCID → tag mapping from our results
    mcid_to_tag: dict[int, tuple[str, int]] = {}
    for el in tagged_elements:
        if el.original_mcid is not None:
            mcid_to_tag[el.original_mcid] = (el.pdf_tag.value, el.page_num)

    if not mcid_to_tag:
        logger.warning("No elements have MCIDs — cannot re-tag. Copying input as-is.")
        import shutil
        shutil.copy2(str(input_path), str(output_path))
        return stats

    try:
        pdf = pikepdf.open(str(input_path))
        root = pdf.Root

        struct_tree_root = root.get("/StructTreeRoot")
        if struct_tree_root is None:
            logger.warning("PDF has no StructTreeRoot — cannot re-tag")
            pdf.save(str(output_path))
            pdf.close()
            return stats

        _walk_and_retag(struct_tree_root, mcid_to_tag, stats)

        pdf.save(str(output_path))
        pdf.close()

        logger.info(
            "Re-tagged: %d struct elements, %d matched, %d changed, %d unmatched",
            stats["total_struct_elems"], stats["matched"],
            stats["changed"], stats["unmatched"],
        )

    except Exception as e:
        logger.error("Re-tag failed: %s", e)
        import shutil
        shutil.copy2(str(input_path), str(output_path))

    return stats


def _mcid_k(mcids: list[int]):
    """Build a struct element's /K value from its page-local MCIDs.

    One MCID -> the integer itself; several (split content) -> an Array of
    integers, all referring to marked content on the element's /Pg.
    """
    if not mcids:
        return None
    if len(mcids) == 1:
        return mcids[0]
    return Array(list(mcids))


def _intersect_area(a, b) -> float:
    """Absolute intersection area of two (x0, y0, x1, y1) boxes in one space."""
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def _append_k_child(elem, child) -> None:
    """Append a child struct element to elem's /K, normalizing /K to an Array.

    /K may legally mix marked-content MCID integers with child struct elements,
    so a bare int (single MCID) is promoted to an Array before appending.
    """
    k = elem.get("/K")
    if k is None:
        elem["/K"] = Array([child])
    elif isinstance(k, Array):
        k.append(child)
    else:
        elem["/K"] = Array([k, child])


def _link_alt_text(a, owner) -> str:
    """Best alternate description for a Link annotation (clause 7.18.5-2 / 7.18.1-2).

    Prefer the visible text of the block the link sits in (owner /ActualText),
    fall back to the link's URI, then a generic label. Capped so a whole-paragraph
    owner does not yield an unwieldy description.
    """
    if isinstance(owner, Dictionary):
        at = owner.get("/ActualText")
        if at is not None:
            s = str(at).strip()
            if s:
                return s[:300]
    action = a.get("/A")
    if isinstance(action, Dictionary):
        uri = action.get("/URI")
        if uri is not None:
            s = str(uri).strip()
            if s:
                return s
    return "Link"


def _tag_link_annotations(
    pdf,
    page,
    doc_elem,
    owners: list[tuple[tuple, object]],
    page_height_pt: float,
    parent_map_entries: list,
    next_sp: int,
    stats: dict,
) -> int:
    """Bind each Link annotation on the page to a Link struct element (clause 7.18.5-1).

    For every /Link annotation: create a Link struct elem holding an OBJR back to
    the annotation, nest it under the most-overlapping block (or Document), and
    reassign the annotation's /StructParent to a fresh ParentTree key that resolves
    to the Link. Marked content is left untouched (link text stays under its block).
    """
    annots = page.obj.get("/Annots")
    if annots is None:
        return next_sp

    for a in annots:
        if str(a.get("/Subtype")) != "/Link":
            continue
        rect = a.get("/Rect")
        if rect is None:
            continue

        rect_std = pdf_to_standard(
            tuple(float(x) for x in rect), page_height_pt
        )
        owner = doc_elem
        best = 0.0
        for bbox, se in owners:
            area = _intersect_area(rect_std, bbox)
            if area > best:
                best = area
                owner = se

        link_elem = pdf.make_indirect(Dictionary({
            "/Type": Name.StructElem,
            "/S": Name("/Link"),
            "/P": owner,
            "/K": Array([Dictionary({
                "/Type": Name.OBJR,
                "/Obj": a,
                "/Pg": page.obj,
            })]),
        }))
        _append_k_child(owner, link_elem)

        # Alternate description (clause 7.18.5-2 / 7.18.1-2): the OBJR-only Link
        # fix added structure but not the Contents the clause also requires.
        if a.get("/Contents") is None:
            a["/Contents"] = String(_link_alt_text(a, owner))

        a["/StructParent"] = next_sp
        parent_map_entries.append((next_sp, link_elem))
        next_sp += 1
        stats["links_tagged"] = stats.get("links_tagged", 0) + 1

    return next_sp


# A field name is useless as an accessible name if it is empty, too short, or
# basically a generated id ("Text12", "f_01", "X3", "check2").
_CRYPTIC_FIELD_RE = re.compile(r"^(?:text|field|untitled|check|button|fld|f|q)?[\W_]*\d+[\W_\d]*$", re.I)


def _cryptic_field_name(name) -> bool:
    if not name:
        return True
    n = str(name).strip()
    if len(n) < 3:
        return True
    if _CRYPTIC_FIELD_RE.match(n):
        return True
    return sum(c.isalpha() for c in n) < len(n) * 0.5


def _clean_label(s):
    if not s:
        return None
    s = s.strip().rstrip(":").strip()
    if len(s) < 2 or not any(c.isalpha() for c in s):
        return None
    return s[:120]


def _widget_label_from_words(rect, words, ph):
    """Derive a form field's visible label from page words: the text immediately to
    the LEFT on the same row (dominant convention), else the line just ABOVE. rect is
    the widget /Rect in PDF points (bottom-left); words are pdfplumber words (top-left)."""
    x0, y0, x1, y1 = rect
    wtop, wbot = ph - y1, ph - y0
    wh = max(wbot - wtop, 1.0)

    def in_row(w):
        c = (w["top"] + w["bottom"]) / 2
        return wtop - wh * 0.5 <= c <= wbot + wh * 0.5

    left = [w for w in words if w["x1"] <= x0 + 2 and 0 <= (x0 - w["x1"]) < 260 and in_row(w)]
    if left:
        left.sort(key=lambda w: w["x0"])
        return _clean_label(" ".join(w["text"] for w in left))
    above = [w for w in words if w["bottom"] <= wtop + 2 and (wtop - w["bottom"]) < 28
             and not (w["x1"] < x0 - 4 or w["x0"] > x1 + 4)]
    if above:
        nb = max(w["bottom"] for w in above)
        row = sorted((w for w in above if abs(w["bottom"] - nb) < 4), key=lambda w: w["x0"])
        return _clean_label(" ".join(w["text"] for w in row))
    return None


def _tag_widget_annotations(
    pdf,
    page,
    doc_elem,
    owners: list[tuple[tuple, object]],
    page_height_pt: float,
    parent_map_entries: list,
    next_sp: int,
    stats: dict,
    input_path=None,
    page_idx: int = 0,
) -> int:
    """Bind each form-field Widget annotation to a /Form struct element (PDF/UA
    7.18.1 + Matterhorn 11-xx; one widget per /Form, OBJR back to the annotation).

    Mirrors _tag_link_annotations. A widget with no accessible name (/TU) gets one:
    the field name (/T) if it is meaningful, otherwise — for the ~37% of corpus fields
    whose name is a generated id like "Text12" — the adjacent visual label (text to the
    left / above the control). Hidden widgets (/F & 2) and already-tagged ones skipped.
    """
    annots = page.obj.get("/Annots")
    if annots is None:
        return next_sp

    _words_cache = {"v": None}  # extracted lazily, once, only if a label is needed

    def _page_words():
        if _words_cache["v"] is None:
            _words_cache["v"] = []
            if input_path is not None:
                try:
                    from tagger.page_cache import open_pdf
                    with open_pdf(str(input_path)) as plumb:
                        if page_idx < len(plumb.pages):
                            _words_cache["v"] = plumb.pages[page_idx].extract_words(use_text_flow=False)
                except Exception:
                    pass
        return _words_cache["v"]

    for a in annots:
        if str(a.get("/Subtype")) != "/Widget":
            continue
        if a.get("/StructParent") is not None:
            continue  # already in a struct tree
        f = a.get("/F")
        if f is not None and (int(f) & 2):  # Hidden — do not tag
            continue
        rect = a.get("/Rect")
        if rect is None:
            continue

        rect_std = pdf_to_standard(tuple(float(x) for x in rect), page_height_pt)
        owner = doc_elem
        best = 0.0
        for bbox, se in owners:
            area = _intersect_area(rect_std, bbox)
            if area > best:
                best = area
                owner = se

        form_elem = pdf.make_indirect(Dictionary({
            "/Type": Name.StructElem,
            "/S": Name("/Form"),
            "/P": owner,
            "/K": Array([Dictionary({
                "/Type": Name.OBJR,
                "/Obj": a,
                "/Pg": page.obj,
            })]),
        }))
        _append_k_child(owner, form_elem)

        # Accessible name (7.18.1 / WCAG 4.1.2): if the field has no tooltip, give it
        # one. Prefer a MEANINGFUL field name; if the name is a generated id, derive
        # the visible label from adjacent page text instead (else screen readers
        # announce "edit text, Text12").
        if a.get("/TU") is None:
            field_name = a.get("/T")
            if field_name is None:
                parent = a.get("/Parent")
                if parent is not None:
                    field_name = parent.get("/T")
            label = None
            if _cryptic_field_name(field_name):
                label = _widget_label_from_words(
                    tuple(float(x) for x in rect), _page_words(), page_height_pt)
                if label:
                    stats["form_labels_from_layout"] = stats.get("form_labels_from_layout", 0) + 1
            if label is not None:
                a["/TU"] = String(label)
            elif field_name is not None:
                a["/TU"] = String(str(field_name))

        a["/StructParent"] = next_sp
        parent_map_entries.append((next_sp, form_elem))
        next_sp += 1
        stats["forms_tagged"] = stats.get("forms_tagged", 0) + 1

    return next_sp


# URL / email auto-detection — conservative WHOLE-TOKEN match (anchored ^…$) so we
# only link a word that IS a URL/email, never a fragment of prose (precision over
# recall: a missed link is recoverable, a wrong link is noise).
_URL_RE = re.compile(r"^(?:https?://|www\.)\S+$", re.IGNORECASE)
_EMAIL_RE = re.compile(r"^[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}$")
_TRAIL = ".,;:!?)]}>\"'"


def _autodetect_link_annotations(pdf, page, page_idx, input_path, stats) -> None:
    """Synthesize /Link annotations for URL / email TEXT that has no existing link
    annotation (the incumbent auto-links such text; we previously only tagged
    pre-existing annotations — measured the #1 coverage gap on the baseline corpus).

    Each synthesized annotation is FUNCTIONAL (/A /URI, so it is clickable) AND, by
    living on the page /Annots, is picked up by _tag_link_annotations which adds the
    /Link struct element + OBJR + /Contents + /StructParent. So this only produces the
    annotation; the conformant structure is the existing machinery's job.

    Detection is whole-token (pdfplumber word) so it cannot link a fragment of a
    sentence; spans already covered by a link are skipped (no double-linking).
    """
    from tagger.page_cache import open_pdf

    try:
        with open_pdf(str(input_path)) as plumb:
            if page_idx >= len(plumb.pages):
                return
            pl_page = plumb.pages[page_idx]
            words = pl_page.extract_words(use_text_flow=False)
            ph = float(pl_page.height)
    except Exception:
        return

    existing = []
    annots = page.obj.get("/Annots")
    if annots is not None:
        for a in annots:
            if str(a.get("/Subtype")) == "/Link":
                r = a.get("/Rect")
                if r is not None:
                    existing.append(tuple(float(x) for x in r))

    def covered(rect):
        cx, cy = (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2
        return any(e[0] <= cx <= e[2] and min(e[1], e[3]) <= cy <= max(e[1], e[3]) for e in existing)

    new_annots = []
    for w in words:
        core = w["text"].strip().rstrip(_TRAIL)
        if not core:
            continue
        is_email = bool(_EMAIL_RE.match(core))
        is_url = bool(_URL_RE.match(core))
        if not (is_url or is_email):
            continue
        # pdfplumber top/bottom are measured from the page top; PDF /Rect is
        # bottom-left origin → flip with the page height.
        x0, x1 = float(w["x0"]), float(w["x1"])
        rect = (x0, ph - float(w["bottom"]), x1, ph - float(w["top"]))
        if covered(rect):
            continue
        if is_url:
            uri = ("http://" + core) if core.lower().startswith("www.") else core
        else:
            uri = "mailto:" + core
        new_annots.append(pdf.make_indirect(Dictionary({
            "/Type": Name.Annot,
            "/Subtype": Name("/Link"),
            "/Rect": Array([round(v, 2) for v in rect]),
            "/Border": Array([0, 0, 0]),
            "/F": 4,  # Print
            "/A": Dictionary({"/Type": Name("/Action"), "/S": Name("/URI"), "/URI": String(uri)}),
        })))
        stats["links_autodetected"] = stats.get("links_autodetected", 0) + 1

    if not new_annots:
        return
    if annots is None:
        page.obj["/Annots"] = Array(new_annots)
    else:
        for a in new_annots:
            annots.append(a)


def _xmp_packet(title: str) -> bytes:
    """Minimal PDF/UA XMP packet: dc:title + pdfuaid:part=1.

    pdfuaid:part is included so that *adding* a Metadata stream (clause 7.1-8)
    does not surface a separate "missing PDF/UA identifier" failure — a conforming
    file declares its UA part. The title doubles as the displayed document title
    (clause 7.1-10 / DisplayDocTitle).
    """
    from xml.sax.saxutils import escape

    t = escape(title or "Untitled")
    return (
        '<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">\n'
        ' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
        '  <rdf:Description rdf:about=""\n'
        '    xmlns:dc="http://purl.org/dc/elements/1.1/"\n'
        '    xmlns:pdfuaid="http://www.aiim.org/pdfua/ns/id/">\n'
        '   <dc:title><rdf:Alt><rdf:li xml:lang="x-default">'
        + t +
        '</rdf:li></rdf:Alt></dc:title>\n'
        '   <pdfuaid:part>1</pdfuaid:part>\n'
        '  </rdf:Description>\n'
        ' </rdf:RDF>\n'
        '</x:xmpmeta>\n'
        '<?xpacket end="w"?>'
    ).encode("utf-8")


def _add_metadata_and_viewer_prefs(pdf, root, title: str) -> None:
    """Add an XMP /Metadata stream (7.1-8) and DisplayDocTitle=true (7.1-10).

    Both are additive catalog entries the pipeline previously never wrote (the incumbent
    inputs already carried them, masking the gap). Also mirrors the title into
    /Info and dc:title so the title veraPDF expects to display is actually present.
    """
    md = pdf.make_stream(_xmp_packet(title))
    md["/Type"] = Name.Metadata
    md["/Subtype"] = Name.XML
    root["/Metadata"] = md

    vp = root.get("/ViewerPreferences")
    if not isinstance(vp, Dictionary):
        vp = pdf.make_indirect(Dictionary({}))
        root["/ViewerPreferences"] = vp
    vp["/DisplayDocTitle"] = True

    info = pdf.trailer.get("/Info")
    if not isinstance(info, Dictionary):
        info = pdf.make_indirect(Dictionary({}))
        pdf.trailer["/Info"] = info
    info["/Title"] = String(title)


# Annotation subtypes that must NOT be wrapped in an Annot struct element:
# Link gets its own /Link tag (see _tag_link_annotations); Widget/PrinterMark
# are excluded by clause 7.18.1-1; Popup is auxiliary to its parent markup.
_UNTAGGED_ANNOT_SUBTYPES = frozenset({"/Link", "/Widget", "/PrinterMark", "/Popup"})


def _tag_markup_annotations(
    pdf,
    page,
    doc_elem,
    owners: list[tuple[tuple, object]],
    page_height_pt: float,
    parent_map_entries: list,
    next_sp: int,
    stats: dict,
) -> int:
    """Nest each non-Link markup annotation in an Annot struct element (clause 7.18.1-1).

    Mirrors _tag_link_annotations but emits /S /Annot: for every annotation whose
    subtype is not Link/Widget/PrinterMark/Popup (and not hidden), create an Annot
    struct elem holding an OBJR to it, nest under the best-overlapping block (or
    Document), give it a fresh /StructParent, and add /Contents if absent. Purely
    additive — the annotation's appearance and content are untouched.
    """
    annots = page.obj.get("/Annots")
    if annots is None:
        return next_sp

    for a in annots:
        if str(a.get("/Subtype")) in _UNTAGGED_ANNOT_SUBTYPES:
            continue
        if a.get("/StructParent") is not None:
            continue  # already tagged
        f = a.get("/F")
        if f is not None and (int(f) & 2):  # Hidden flag — do not tag
            continue
        rect = a.get("/Rect")
        if rect is None:
            continue

        rect_std = pdf_to_standard(tuple(float(x) for x in rect), page_height_pt)
        owner = doc_elem
        best = 0.0
        for bbox, se in owners:
            area = _intersect_area(rect_std, bbox)
            if area > best:
                best = area
                owner = se

        annot_elem = pdf.make_indirect(Dictionary({
            "/Type": Name.StructElem,
            "/S": Name("/Annot"),
            "/P": owner,
            "/K": Array([Dictionary({
                "/Type": Name.OBJR,
                "/Obj": a,
                "/Pg": page.obj,
            })]),
        }))
        _append_k_child(owner, annot_elem)

        if a.get("/Contents") is None:
            a["/Contents"] = String(_link_alt_text(a, owner))

        a["/StructParent"] = next_sp
        parent_map_entries.append((next_sp, annot_elem))
        next_sp += 1
        stats["annots_tagged"] = stats.get("annots_tagged", 0) + 1

    return next_sp


def _struct_kids(node) -> list:
    k = node.get("/K")
    if isinstance(k, Array):
        return list(k)
    if k is not None:
        return [k]
    return []


def _normalize_table_columns(pdf, root) -> int:
    """Make every TR in each Table span the same number of columns (clause 7.2-42/43).

    veraPDF counts effective columns (sum of col spans). Drop Col/RowSpan so
    effective columns equal cell count, wrap any non-cell child that leaked into a
    TR in its own TD, then pad short rows with empty TD cells to the table max.
    Struct-only (empty TDs carry no marked content) -> rendering-neutral. Returns
    the number of empty TD cells added.
    """
    padded = 0

    def visit(node):
        nonlocal padded
        if not isinstance(node, Dictionary):
            return
        if str(node.get("/S")) == "/Table":
            trs = [c for c in _struct_kids(node)
                   if isinstance(c, Dictionary) and str(c.get("/S")) == "/TR"]
            for tr in trs:
                for c in _struct_kids(tr):
                    if not isinstance(c, Dictionary):
                        continue
                    a = c.get("/A")
                    for ad in (a if isinstance(a, Array) else [a]):
                        if isinstance(ad, Dictionary):
                            for span in ("/ColSpan", "/RowSpan"):
                                if ad.get(span) is not None:
                                    del ad[span]
                    for span in ("/ColSpan", "/RowSpan"):
                        if c.get(span) is not None:
                            del c[span]
            for tr in trs:
                tk = tr.get("/K")
                if not isinstance(tk, Array):
                    tr["/K"] = tk = Array([tk]) if tk is not None else Array([])
                for idx, child in enumerate(list(tk)):
                    if isinstance(child, Dictionary) and str(child.get("/S")) not in ("/TD", "/TH"):
                        td = pdf.make_indirect(Dictionary({
                            "/Type": Name.StructElem, "/S": Name("/TD"),
                            "/P": tr, "/K": Array([child]),
                        }))
                        child["/P"] = td
                        tk[idx] = td
            counts = [sum(1 for c in _struct_kids(tr)
                          if isinstance(c, Dictionary) and str(c.get("/S")) in ("/TD", "/TH"))
                      for tr in trs]
            maxc = max(counts) if counts else 0
            for tr, n in zip(trs, counts):
                if n < maxc:
                    k = tr.get("/K")
                    if not isinstance(k, Array):
                        tr["/K"] = k = Array([k]) if k is not None else Array([])
                    for _ in range(maxc - n):
                        k.append(pdf.make_indirect(Dictionary({
                            "/Type": Name.StructElem, "/S": Name("/TD"), "/P": tr,
                        })))
                        padded += 1
        for c in _struct_kids(node):
            visit(c)

    visit(root)
    return padded


def _normalize_heading_levels(root) -> int:
    """Renumber Hn by nesting depth so the sequence never skips a level (clause 7.4.2-1).

    Walks headings in document order; a stack of ancestor nominal levels yields a
    depth-based level that starts at H1 and never skips. No-op when already
    compliant. Struct-only (/S rewrite) -> rendering-neutral. Returns count changed.
    """
    headings = []

    def visit(node):
        if not isinstance(node, Dictionary):
            return
        s = str(node.get("/S")) if node.get("/S") is not None else ""
        if len(s) == 3 and s.startswith("/H") and s[2].isdigit():
            headings.append(node)
        for c in _struct_kids(node):
            visit(c)

    visit(root)

    changed = 0
    stack = []
    for h in headings:
        L = int(str(h.get("/S"))[2])
        while stack and stack[-1] >= L:
            stack.pop()
        stack.append(L)
        new = len(stack)
        if new != L:
            h["/S"] = Name(f"/H{new}")
            changed += 1
    return changed


def tag_untagged_pdf(
    input_path: str | Path,
    output_path: str | Path,
    tagged_elements: list[TaggedElement],
    total_pages: int,
    repair_mode: str = "auto",
    approved_ids: set[str] | None = None,
) -> dict:
    """
    V2: Build a complete struct tree for an untagged PDF.

    Creates /MarkInfo, /StructTreeRoot, Document element,
    StructElem for each tagged element, and /Tabs /S on pages.
    Also injects BDC/EMC operators into the content streams.
    """
    from tagger.stage10_writeback.content_stream_writer import (
        artifact_wrap_forms,
        detect_cidsets,
        detect_missing_space_refs,
        detect_notdef_refs,
        detect_unembedded_fonts,
        inject_bdc_markers,
    )
    from tagger.stage10_writeback.repair_gate import gate_and_apply, write_report

    stats = {
        "total_elements_written": 0,
        "pages_modified": 0,
        "struct_tree_created": False,
        "links_tagged": 0,
        "links_autodetected": 0,
        "form_labels_from_layout": 0,
        "annots_tagged": 0,
        "repair_mode": repair_mode,
        "repair_findings": {},
        "table_cells_padded": 0,
        "headings_renumbered": 0,
        "formula_mathml_embedded": 0,
    }

    try:
        pdf = pikepdf.open(str(input_path))
        root = pdf.Root
        # PDF 2.0 Associated Files (MathML on /Formula) to register on the
        # catalog /AF array after the struct tree is built.
        catalog_af: list = []

        # MCIDs are assigned page-locally by inject_bdc_markers (the single MCID
        # authority); the struct tree below is built from the maps it returns.

        # 1. Set MarkInfo
        root["/MarkInfo"] = pdf.make_indirect(
            Dictionary({"/Marked": True})
        )

        # 2. Create StructTreeRoot (make indirect so it can be referenced)
        struct_tree_root = pdf.make_indirect(
            Dictionary({"/Type": Name.StructTreeRoot})
        )

        # 3. Create Document element
        doc_elem = pdf.make_indirect(Dictionary({
            "/Type": Name.StructElem,
            "/S": Name("/Document"),
            "/P": struct_tree_root,
            "/K": Array([]),
        }))

        # Group elements by page
        by_page: dict[int, list[TaggedElement]] = {}
        for el in tagged_elements:
            by_page.setdefault(el.page_num, []).append(el)

        # 4. Build struct elements for each page
        parent_map_entries: list[tuple[int, object]] = []
        # Single monotonic counter feeding both page /StructParents and annotation
        # /StructParent keys, so the two never collide in the ParentTree number tree.
        next_sp = 0

        for page_num in sorted(by_page.keys()):
            page_elements = by_page[page_num]
            page_idx = page_num - 1

            if page_idx >= len(pdf.pages):
                continue

            page = pdf.pages[page_idx]
            mb = page.mediabox
            page_height_pt = float(mb[3]) - float(mb[1])

            # (bbox, struct_elem) candidates for nesting Link annotations.
            page_struct_owners: list[tuple[tuple, object]] = []

            # Build the struct tree in the pipeline's reading order. This list
            # is already in reading order (content_router emits TaggedElements per
            # page in MinerU region / reading_order sequence; later stages preserve
            # it). A geometric (top, left) re-sort here interleaved the columns of
            # multi-column documents — e.g. a right-column body line at top=73 was
            # placed before the title at top=81 — corrupting the assistive reading
            # order. MCID allocation is content-stream-driven (inject_bdc_markers),
            # so element order here only affects struct /K order, not MCIDs.

            # 4.1 Inject BDC markers — this allocates the page-local MCIDs.
            element_to_mcids, mcid_to_element, _mcid_tag = inject_bdc_markers(
                pdf, page, page_num, page_elements
            )

            # MCID -> the struct element that owns that marked content. The
            # page's ParentTree array is built from this at the end of the page
            # so it is indexed by MCID (correct by construction).
            mcid_to_structelem: dict[int, object] = {}

            # Track consecutive LI elements for grouping into L
            i = 0
            while i < len(page_elements):
                el = page_elements[i]

                # Skip artifacts
                if el.pdf_tag == PDFTag.ARTIFACT:
                    i += 1
                    continue

                # Check for list item run → wrap in L
                if el.pdf_tag == PDFTag.LI:
                    # Find all consecutive LI elements
                    li_run = [el]
                    j = i + 1
                    while j < len(page_elements) and page_elements[j].pdf_tag == PDFTag.LI:
                        li_run.append(page_elements[j])
                        j += 1

                    # Create L (list) container
                    li_struct_elems = Array([])
                    for li_el in li_run:
                        li_mcids = element_to_mcids.get(li_el.element_id, [])
                        if not li_mcids:
                            continue
                        li_struct, lbody = _build_list_item_struct(
                            pdf, li_el, doc_elem, page.obj,
                            li_mcids,
                        )
                        li_struct_elems.append(li_struct)
                        # Nest any overlapping Link under LBody (LI itself should
                        # hold only Lbl/LBody), keeping list structure valid.
                        page_struct_owners.append((li_el.bbox, lbody))
                        # The list item's marked content lives under LBody.
                        for m in li_mcids:
                            mcid_to_structelem[m] = lbody
                        stats["total_elements_written"] += 1

                    if li_struct_elems:
                        list_elem = pdf.make_indirect(Dictionary({
                            "/Type": Name.StructElem,
                            "/S": Name("/L"),
                            "/P": doc_elem,
                            "/K": li_struct_elems,
                        }))
                        # Re-parent each LI to the L container. _build_list_item_struct
                        # sets /P to doc_elem (the L doesn't exist yet); veraPDF reads
                        # /P for parentStandardType, so an unfixed /P fails clause 7.2-17.
                        for li_struct in li_struct_elems:
                            li_struct["/P"] = list_elem
                        doc_elem["/K"].append(list_elem)

                    i = j
                    continue

                # Check for TOC item run → wrap in TOC (mirrors LI → L).
                # PDF/UA clause 7.2-26: every TOCI must be a child of a TOC.
                if el.pdf_tag == PDFTag.TOCI:
                    toci_run = [el]
                    j = i + 1
                    while j < len(page_elements) and page_elements[j].pdf_tag == PDFTag.TOCI:
                        toci_run.append(page_elements[j])
                        j += 1

                    toc_elem = pdf.make_indirect(Dictionary({
                        "/Type": Name.StructElem,
                        "/S": Name("/TOC"),
                        "/P": doc_elem,
                        "/K": Array([]),
                    }))

                    for toci_el in toci_run:
                        toci_mcids = element_to_mcids.get(toci_el.element_id, [])
                        if not toci_mcids:
                            continue
                        toci_dict = {
                            "/Type": Name.StructElem,
                            "/S": Name("/TOCI"),
                            "/P": toc_elem,
                            "/K": _mcid_k(toci_mcids),
                            "/Pg": page.obj,
                        }
                        if toci_el.text:
                            toci_dict["/ActualText"] = String(toci_el.text)
                        toci_struct = pdf.make_indirect(Dictionary(toci_dict))
                        toc_elem["/K"].append(toci_struct)
                        page_struct_owners.append((toci_el.bbox, toci_struct))
                        for m in toci_mcids:
                            mcid_to_structelem[m] = toci_struct
                        stats["total_elements_written"] += 1

                    if len(toc_elem["/K"]) > 0:
                        doc_elem["/K"].append(toc_elem)
                        stats["total_elements_written"] += 1

                    i = j
                    continue

                # TABLE nested structure
                if el.pdf_tag == PDFTag.TABLE and el.specialist_data.get("cells"):
                    # Construct nested TR/TH/TD
                    table_struct_elem = pdf.make_indirect(Dictionary({
                        "/Type": Name.StructElem,
                        "/S": Name("/Table"),
                        "/P": doc_elem,
                        "/K": Array([]),
                    }))
                    doc_elem["/K"].append(table_struct_elem)
                    # Table/TR own no marked content, so they are not entered in
                    # the MCID-indexed ParentTree array — only their TH/TD cells.

                    # Group cells by row
                    rows = {}
                    for cell in el.specialist_data["cells"]:
                        rows.setdefault(cell["row_idx"], []).append(cell)

                    for row_idx in sorted(rows.keys()):
                        tr_elem = pdf.make_indirect(Dictionary({
                            "/Type": Name.StructElem,
                            "/S": Name("/TR"),
                            "/P": table_struct_elem,
                            "/K": Array([]),
                        }))
                        table_struct_elem["/K"].append(tr_elem)

                        for cell in sorted(rows[row_idx], key=lambda c: c["col_idx"]):
                            cell_id = f"{el.element_id}_cell_{cell['row_idx']}_{cell['col_idx']}"
                            cell_mcids = element_to_mcids.get(cell_id, [])

                            is_empty = not cell.get("merged_from")
                            # A cell may carry native chars (merged_from) yet get NO
                            # MCIDs if BDC injection couldn't mark them. DON'T drop it:
                            # dropping shifts every later cell left and _normalize_table_
                            # columns pads an empty TD at the end (the column-shift +
                            # lost-text failure the dp-bench table decomposition exposed
                            # — e.g. doc052 'REGIONS'/9/8/5 vanished). Emit it positionally
                            # with /ActualText instead (PDF/UA-valid, same no-/K path as
                            # OCR'd text). cell_mcids drives /K below; empty -> /ActualText.
                            if not cell_mcids:
                                is_empty = True

                            # Dynamically determine row header status based on text content
                            cell_text = cell.get("text", "")
                            
                            if not cell_text and cell.get("merged_from"):
                                # pdfplumber failed to extract text, but the cell has physical characters.
                                # Assume it is not numeric.
                                is_numeric = False
                            else:
                                is_numeric = _is_numeric_content(cell_text)
                                
                            is_dynamic_row_header = (
                                cell["col_idx"] == 0
                                and not cell.get("is_header")
                                and (bool(cell_text.strip()) or bool(cell.get("merged_from")))
                                and not is_numeric
                            )
                            is_row_header = cell.get("is_row_header") or is_dynamic_row_header

                            td_tag = "TH" if cell.get("is_header") or is_row_header else "TD"

                            td_elem_dict = {
                                "/Type": Name.StructElem,
                                "/S": Name(f"/{td_tag}"),
                                "/P": tr_elem,
                            }
                            
                            if not is_empty:
                                td_elem_dict["/K"] = _mcid_k(cell_mcids)
                                td_elem_dict["/Pg"] = page.obj

                            if td_tag == "TH":
                                if cell.get("is_header"):
                                    td_elem_dict["/A"] = Dictionary({"/O": Name("/Table"), "/Scope": Name("/Column")})
                                elif is_row_header:
                                    td_elem_dict["/A"] = Dictionary({"/O": Name("/Table"), "/Scope": Name("/Row")})

                            if cell.get("text"):
                                td_elem_dict["/ActualText"] = String(cell["text"])

                            td_elem = pdf.make_indirect(Dictionary(td_elem_dict))
                            tr_elem["/K"].append(td_elem)
                            for m in cell_mcids:
                                mcid_to_structelem[m] = td_elem
                            stats["total_elements_written"] += 1

                    stats["total_elements_written"] += 1
                    i += 1
                    continue

                # Regular element
                mcids = element_to_mcids.get(el.element_id, [])
                # Drop only truly empty elements; if the element carries text
                # (e.g. OCR'd lines on a scanned page that have no corresponding
                # content-stream glyphs to MCID against) we still emit the struct
                # element with /ActualText so assistive tech reads it.
                if not mcids and not (el.text or "").strip():
                    i += 1
                    continue

                tag_name = el.pdf_tag.value
                if tag_name not in WRITEBACK.tag_role_map:
                    tag_name = "P"

                struct_elem_dict = {
                    "/Type": Name.StructElem,
                    "/S": Name(f"/{tag_name}"),
                    "/P": doc_elem,
                }
                if mcids:
                    struct_elem_dict["/K"] = _mcid_k(mcids)
                    struct_elem_dict["/Pg"] = page.obj

                if el.text:
                    struct_elem_dict["/ActualText"] = String(el.text)

                if el.pdf_tag == PDFTag.FIGURE and el.alt_text:
                    struct_elem_dict["/Alt"] = String(el.alt_text)

                # /Formula (PDF/UA-2): attach MathML as an Associated File and a
                # text /Alt fallback so AT can read the equation either way.
                formula_af = None
                if el.pdf_tag == PDFTag.FORMULA:
                    sd = el.specialist_data or {}
                    if "/Alt" not in struct_elem_dict:
                        struct_elem_dict["/Alt"] = String(
                            (el.text or "").strip() or "Mathematical formula")
                    mathml = sd.get("mathml")
                    if mathml:
                        formula_af = _embed_mathml_af(
                            pdf, mathml, stats["total_elements_written"])
                        struct_elem_dict["/AF"] = Array([formula_af])

                struct_elem = pdf.make_indirect(Dictionary(struct_elem_dict))
                if formula_af is not None:
                    catalog_af.append(formula_af)
                    stats["formula_mathml_embedded"] += 1

                doc_elem["/K"].append(struct_elem)
                page_struct_owners.append((el.bbox, struct_elem))
                for m in mcids:
                    mcid_to_structelem[m] = struct_elem

                stats["total_elements_written"] += 1
                i += 1

            # Build the page's ParentTree array INDEXED BY MCID: position k holds
            # the struct element that owns page-local MCID k. Every allocated
            # MCID maps to an element that produced a struct element above, so
            # the range 0..n-1 is fully covered.
            n_mcids = len(mcid_to_element)
            missing = [k for k in range(n_mcids) if k not in mcid_to_structelem]
            if missing:
                logger.warning(
                    "Page %d: %d MCIDs without a struct element (%s); using OBJR-free gap.",
                    page_num, len(missing), missing[:10],
                )
            page_struct_parents = Array(
                [mcid_to_structelem.get(k, doc_elem) for k in range(n_mcids)]
            )

            # Set page's StructParents (page-level: array indexed by MCID)
            page_key = next_sp
            next_sp += 1
            page.obj["/StructParents"] = page_key
            parent_map_entries.append((page_key, page_struct_parents))

            # Synthesize /Link annotations for bare URL/email text (incumbent
            # auto-links these; we previously tagged only pre-existing annotations).
            # Runs BEFORE _tag_link_annotations so the new annots get wrapped too.
            _autodetect_link_annotations(pdf, page, page_idx, input_path, stats)

            # Tag Link annotations on this page (object-level StructParent keys).
            next_sp = _tag_link_annotations(
                pdf, page, doc_elem, page_struct_owners,
                page_height_pt, parent_map_entries, next_sp, stats,
            )

            # Tag form-field Widget annotations as /Form (one widget per /Form).
            next_sp = _tag_widget_annotations(
                pdf, page, doc_elem, page_struct_owners,
                page_height_pt, parent_map_entries, next_sp, stats,
                input_path=input_path, page_idx=page_idx,
            )

            # Tag remaining markup annotations (FreeText, etc.) as /Annot.
            next_sp = _tag_markup_annotations(
                pdf, page, doc_elem, page_struct_owners,
                page_height_pt, parent_map_entries, next_sp, stats,
            )

            # Set reading order
            page.obj["/Tabs"] = Name("/S")

            stats["pages_modified"] += 1

        # 5. Build ParentTree number tree
        nums = Array([])
        for idx, parents in parent_map_entries:
            nums.append(idx)
            nums.append(parents)

        parent_tree = pdf.make_indirect(Dictionary({
            "/Nums": nums,
        }))

        # 6. Finalize struct tree
        struct_tree_root["/K"] = doc_elem
        struct_tree_root["/ParentTree"] = parent_tree

        root["/StructTreeRoot"] = struct_tree_root

        # Register MathML Associated Files on the catalog /AF (PDF 2.0 7.11.4 —
        # an associated file must be reachable from the catalog, not only from
        # the struct element).
        if catalog_af:
            existing_af = root.get("/AF")
            af_array = existing_af if isinstance(existing_af, Array) else Array([])
            for fs in catalog_af:
                af_array.append(fs)
            root["/AF"] = af_array
        stats["struct_tree_created"] = True

        # 7. Set language
        if "/Lang" not in root:
            root["/Lang"] = String("en-US")

        # 7b. XMP /Metadata stream (7.1-8) + ViewerPreferences/DisplayDocTitle
        #     (7.1-10). Additive catalog entries; the incumbent inputs already carried
        #     them, so the pipeline never added them — the clean corpus exposed it.
        info = pdf.trailer.get("/Info")
        existing_title = info.get("/Title") if isinstance(info, Dictionary) else None
        title = (str(existing_title).strip() if existing_title else "") or \
            Path(input_path).stem.replace("_", " ").strip().title()
        _add_metadata_and_viewer_prefs(pdf, root, title)

        # 8. Artifact-wrap marks inside Form XObjects (their content streams have
        #    their own marked-content scope, untouched by page-level injection).
        artifact_wrap_forms(pdf)

        # 9. Modifying font repairs (the ONLY gated surface): detect, then apply
        #    per repair_mode. All tagging above is additive and always runs; only
        #    these source-altering repairs (edit fonts/show strings) are gated, to
        #    honour the the incumbent-safe posture. See repair_gate.py for the boundary test.
        findings = []
        findings += detect_cidsets(pdf)
        findings += detect_notdef_refs(pdf)
        findings += detect_missing_space_refs(pdf)
        findings += detect_unembedded_fonts(pdf)
        gate_and_apply(findings, repair_mode=repair_mode, approved_ids=approved_ids)
        report = write_report(
            findings, Path(output_path).with_suffix(".repairs.json"), repair_mode
        )
        stats["repair_findings"] = report["summary"]

        # 10. Normalize struct tree: uniform table columns + non-skipping headings.
        #     (Additive — reshapes our tag tree only; always runs.)
        stats["table_cells_padded"] = _normalize_table_columns(pdf, doc_elem)
        stats["headings_renumbered"] = _normalize_heading_levels(doc_elem)

        # Save
        pdf.save(str(output_path))
        pdf.close()

        logger.info(
            "Tagged PDF written: %d elements across %d pages",
            stats["total_elements_written"], stats["pages_modified"],
        )

    except Exception as e:
        logger.error("Struct tree writeback failed: %s", e)
        import traceback
        traceback.print_exc()
        import shutil
        shutil.copy2(str(input_path), str(output_path))

    return stats


def _build_list_item_struct(
    pdf,
    el: TaggedElement,
    parent,
    page_obj,
    mcids: list[int],
):
    """
    Build a StructElem for a list item with Lbl + LBody children.

    PDF/UA structure: LI > [ Lbl, LBody ]. The list item's marked content is
    referenced from LBody (/K = its page-local MCIDs). Returns (li_elem, lbody)
    so the caller can index the ParentTree array by those MCIDs to LBody.
    """
    label = el.specialist_data.get("list_label", "") if hasattr(el, "specialist_data") and el.specialist_data else ""
    body = el.specialist_data.get("list_body", el.text) if hasattr(el, "specialist_data") and el.specialist_data else el.text

    children = Array([])

    if label:
        lbl = pdf.make_indirect(Dictionary({
            "/Type": Name.StructElem,
            "/S": Name("/Lbl"),
            "/ActualText": String(label),
        }))
        children.append(lbl)

    lbody = pdf.make_indirect(Dictionary({
        "/Type": Name.StructElem,
        "/S": Name("/LBody"),
        "/K": _mcid_k(mcids),
        "/Pg": page_obj,
        "/ActualText": String(body or ""),
    }))
    children.append(lbody)

    li_elem = pdf.make_indirect(Dictionary({
        "/Type": Name.StructElem,
        "/S": Name("/LI"),
        "/P": parent,
        "/K": children if len(children) > 1 else children[0],
        "/Pg": page_obj,
        "/ActualText": String(el.text or ""),
    }))

    # Parent links for the children.
    lbody["/P"] = li_elem
    if label:
        children[0]["/P"] = li_elem

    return li_elem, lbody


def _walk_and_retag(
    node,
    mcid_to_tag: dict[int, tuple[str, int]],
    stats: dict,
) -> None:
    """Recursively walk the struct tree and update /S entries."""
    if not isinstance(node, Dictionary):
        return

    node_type = node.get("/Type")
    if node_type == Name.StructElem:
        stats["total_struct_elems"] += 1

        k_val = node.get("/K")
        mcid = _extract_mcid(k_val)

        if mcid is not None and mcid in mcid_to_tag:
            new_tag, page_num = mcid_to_tag[mcid]
            old_tag = str(node.get("/S", ""))

            if old_tag != f"/{new_tag}":
                node["/S"] = Name(f"/{new_tag}")
                stats["changed"] += 1
                logger.debug(
                    "MCID %d: %s → /%s (page %d)",
                    mcid, old_tag, new_tag, page_num,
                )
            stats["matched"] += 1
        else:
            stats["unmatched"] += 1

    # Recurse into /K children
    k_val = node.get("/K")
    if isinstance(k_val, Array):
        for child in k_val:
            if isinstance(child, Dictionary):
                _walk_and_retag(child, mcid_to_tag, stats)
    elif isinstance(k_val, Dictionary):
        _walk_and_retag(k_val, mcid_to_tag, stats)


def _extract_mcid(k_val) -> int | None:
    """Extract MCID from a struct element's /K value."""
    if isinstance(k_val, int):
        return int(k_val)

    if isinstance(k_val, Dictionary):
        mcid = k_val.get("/MCID")
        if mcid is not None:
            return int(mcid)

    if isinstance(k_val, Array):
        for item in k_val:
            result = _extract_mcid(item)
            if result is not None:
                return result

    return None
