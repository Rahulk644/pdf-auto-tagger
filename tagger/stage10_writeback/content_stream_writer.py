import logging
import pikepdf

from tagger.stage10_writeback.repair_gate import MODIFYING, Finding
from tagger.stage1_extraction.coord_transformer import pdf_to_standard

logger = logging.getLogger(__name__)


def _affine_mul(m1: tuple, m2: tuple) -> tuple:
    """Compose two PDF affine matrices (6-tuples); m1 is applied first.

    A point transforms as [x y 1] · M. Concatenating cm (m1) onto the CTM (m2)
    gives a matrix where the image's unit square is mapped by m1 then m2.
    """
    a1, b1, c1, d1, e1, f1 = m1
    a2, b2, c2, d2, e2, f2 = m2
    return (
        a1 * a2 + b1 * c2,
        a1 * b2 + b1 * d2,
        c1 * a2 + d1 * c2,
        c1 * b2 + d1 * d2,
        e1 * a2 + f1 * c2 + e2,
        e1 * b2 + f1 * d2 + f2,
    )


def _image_rect_std(ctm: tuple, page_height_pt: float) -> tuple:
    """Device-space rect (standard 150-DPI, top-left) an image Do/inline image fills.

    An image is painted by mapping the unit square [0,1]^2 through the CTM, so its
    placement rect is the bbox of the four transformed corners (in PDF points),
    converted to the standard coordinate space the Figure bboxes live in.
    """
    a, b, c, d, e, f = ctm
    xs, ys = [], []
    for x, y in ((0, 0), (1, 0), (0, 1), (1, 1)):
        xs.append(a * x + c * y + e)
        ys.append(b * x + d * y + f)
    return pdf_to_standard((min(xs), min(ys), max(xs), max(ys)), page_height_pt)


def _overlap_area(a: tuple, b: tuple) -> float:
    """Intersection area of two (x0,y0,x1,y1) boxes in one coordinate space."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy

# Text-showing operators (the only ops that consume Stage-1 characters).
_TEXT_OPS = {"Tj", "TJ", "'", '"'}
# Operators that paint visible marks directly on the current content stream.
# (Do is resolved further: /Image marks here, /Form recurses separately.)
# NB: pikepdf names the inline-image op "INLINE IMAGE" (with a space), not
# "INLINE_IMAGE" — the underscore form never matched, leaving inline images
# (e.g. raster fills inside chart Form XObjects) bare and failing clause 7.1-3.
_PAINT_OPS = {"f", "F", "f*", "S", "s", "B", "B*", "b", "b*", "sh", "Do", "INLINE IMAGE"}
# Graphics-state barriers. A marked-content sequence must not straddle these,
# so any open *artifact* sequence is closed before them (tagged sequences are
# left open to avoid splitting one element's MCID across multiple sequences).
_GSTATE_BARRIERS = {"q", "Q", "BX", "EX"}
# Pre-existing marked content is stripped; we re-derive all marking ourselves.
_STRIP_OPS = {"BDC", "BMC", "EMC"}


def _emc():
    return pikepdf.ContentStreamInstruction(
        pikepdf._core._ObjectList([]), pikepdf.Operator("EMC")
    )


def _artifact_bmc():
    return pikepdf.ContentStreamInstruction(
        pikepdf._core._ObjectList([pikepdf.Name("/Artifact")]),
        pikepdf.Operator("BMC"),
    )


def _tag_bdc(mcid: int, tag: str):
    return pikepdf.ContentStreamInstruction(
        pikepdf._core._ObjectList(
            [pikepdf.Name(f"/{tag}"), pikepdf.Dictionary({"/MCID": mcid})]
        ),
        pikepdf.Operator("BDC"),
    )


def _text_len(instruction, op: str, bpc: int = 1) -> int:
    """Number of glyphs drawn by a text-showing operator.

    Counts CHARACTER CODES — raw bytes // bytes-per-code — NOT decoded-string
    length. A composite (Type0) font encodes one glyph in `bpc` bytes (2 for the
    near-universal Identity-H/V), so len(str)/len(bytes) over-counts its glyphs and
    desyncs the positional char<->glyph mapping in _rewrite_stream. On dp-bench 052
    that drift left 1043 Type0 glyphs unmatched (1536 glyphs vs 2579 bytes) — the
    table and most body text fell through to /Artifact (lost to AT and to TEDS).
    """
    def _codes(s) -> int:
        return len(bytes(s)) // bpc
    if op == "TJ":
        return sum(_codes(item) for item in instruction.operands[0]
                   if isinstance(item, pikepdf.String))
    if op == '"':
        # aw ac string " — the string is the last operand.
        return _codes(instruction.operands[-1])
    # Tj and ' both take a single string operand.
    return _codes(instruction.operands[0])


def _font_bytes_per_code(font_obj) -> int:
    """Bytes per character code for a font: composite (Type0) fonts use multi-byte
    codes (Identity-H/V and the common CJK CMaps are 2-byte), simple fonts 1 byte."""
    try:
        if str(font_obj.get("/Subtype")) == "/Type0":
            return 2
    except Exception:
        pass
    return 1


def _build_font_bpc(resources) -> dict:
    """Map each font resource name (e.g. '/C2_0') to its bytes-per-code, so the
    rewriter can count glyphs correctly per the currently-selected (Tf) font."""
    out: dict = {}
    try:
        fonts = resources.get("/Font") if resources is not None else None
        if fonts is not None:
            for name, fobj in fonts.items():
                out[str(name)] = _font_bytes_per_code(fobj)
    except Exception:
        pass
    return out


def _rewrite_stream(instructions, char_to_key, key_to_tag, do_subtype,
                    figure_boxes=None, page_height_pt=None, font_bpc=None):
    """Run the marked-content state machine over a parsed content stream.

    Guarantees that every mark-producing operator ends up inside either a tagged
    BDC (text belonging to a struct element) or an /Artifact BMC sequence, with
    every sequence wholly contained within one text object and — for artifacts —
    one graphics-state level (proper nesting).

    This function is the single MCID authority: it allocates a fresh PAGE-LOCAL
    MCID (0,1,2,… reset per stream) every time it opens a tagged run, so a
    struct element whose glyphs are split across the stream collects several
    distinct MCIDs. ``char_to_key`` maps an absolute Stage-1 char index to a
    struct-element key (element_id, or cell_id for table cells); ``key_to_tag``
    maps that key to its PDF tag. Pass empty dicts for form streams (everything
    becomes an artifact, no MCIDs allocated). ``do_subtype`` maps an XObject
    resource name to its /Subtype so /Form invocations pass through (their inner
    marks are wrapped by the form pass) while /Image invocations are wrapped.

    Returns (new_instructions, element_to_mcids, mcid_to_element, mcid_to_tag):
      element_to_mcids: {key -> [mcid, …]} in content order (for struct /K)
      mcid_to_element:  {mcid -> key}      (for the MCID-indexed ParentTree)
      mcid_to_tag:      {mcid -> tag}
    """
    new_cs = []
    element_to_mcids: dict = {}
    mcid_to_element: dict = {}
    mcid_to_tag: dict = {}
    next_mcid = 0
    current_char_idx = 0
    current_bpc = 1  # bytes-per-code of the current font (Tf); 2 for Type0
    font_bpc = font_bpc or {}
    active = None  # None | ("tag", key) | ("art",)
    ctm = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    ctm_stack: list = []

    def close():
        nonlocal active
        if active is not None:
            new_cs.append(_emc())
            active = None

    def open_tag(key):
        nonlocal active, next_mcid
        mcid = next_mcid
        next_mcid += 1
        tag = key_to_tag[key]
        new_cs.append(_tag_bdc(mcid, tag))
        element_to_mcids.setdefault(key, []).append(mcid)
        mcid_to_element[mcid] = key
        mcid_to_tag[mcid] = tag
        active = ("tag", key)

    def open_art():
        nonlocal active
        new_cs.append(_artifact_bmc())
        active = ("art",)

    def match_figure(cur_ctm):
        """Figure element_id this image best falls within, else None."""
        if not figure_boxes or page_height_pt is None:
            return None
        rect = _image_rect_std(cur_ctm, page_height_pt)
        img_area = max(1e-6, (rect[2] - rect[0]) * (rect[3] - rect[1]))
        best_key, best_cov = None, 0.0
        for bbox, key in figure_boxes:
            cov = _overlap_area(rect, bbox) / img_area
            if cov > best_cov:
                best_cov, best_key = cov, key
        # Require the image to lie mostly inside the figure region (avoids
        # claiming a decorative/background image that merely grazes a figure).
        return best_key if best_cov >= 0.5 else None

    for ins in instructions:
        op = str(ins.operator)

        if op in _STRIP_OPS:
            continue

        if op == "BT":
            if active == ("art",):
                close()
            new_cs.append(ins)
            continue
        if op == "ET":
            # A tagged or artifact text run cannot survive the end of the text
            # object it lives in.
            close()
            new_cs.append(ins)
            continue
        if op in _GSTATE_BARRIERS:
            if active == ("art",):
                close()
            if op == "q":
                ctm_stack.append(ctm)
            elif op == "Q" and ctm_stack:
                ctm = ctm_stack.pop()
            new_cs.append(ins)
            continue
        if op == "cm":
            try:
                m = tuple(float(x) for x in ins.operands)
                if len(m) == 6:
                    ctm = _affine_mul(m, ctm)
            except Exception:
                pass
            new_cs.append(ins)
            continue
        if op == "Tf":
            # Track the selected font's bytes-per-code so glyph counting (and thus
            # the char<->glyph index alignment) is correct for composite fonts.
            if ins.operands:
                current_bpc = font_bpc.get(str(ins.operands[0]), 1)
            new_cs.append(ins)
            continue

        if op in _TEXT_OPS:
            op_len = _text_len(ins, op, current_bpc)
            target = None
            for i in range(current_char_idx, current_char_idx + op_len):
                if i in char_to_key:
                    target = char_to_key[i]
                    break
            if target is not None:
                if active != ("tag", target):
                    close()
                    open_tag(target)
            elif active is None:
                # Untagged text with nothing open (e.g. leading whitespace or an
                # artifact-tagged element's text) becomes an artifact. If a tag
                # is already open, inter-word whitespace stays inside it.
                open_art()
            new_cs.append(ins)
            current_char_idx += op_len
            continue

        if op in _PAINT_OPS:
            if op == "Do":
                name = str(ins.operands[0]) if ins.operands else ""
                if do_subtype(name) == "/Form":
                    # Form invocation produces no marks itself; the form pass
                    # wraps the marks inside its content stream.
                    new_cs.append(ins)
                    continue
            # Image op (image Do or inline image): if it lies within a Figure
            # region, tag it as /Figure marked content so Stage 10 builds a
            # /Figure struct elem (with /Alt) instead of artifact-wrapping it.
            if op in ("Do", "INLINE IMAGE"):
                fig = match_figure(ctm)
                if fig is not None:
                    close()
                    open_tag(fig)
                    new_cs.append(ins)
                    close()  # the Figure MCID wraps exactly this image op
                    continue
            if active != ("art",):
                close()
                open_art()
            new_cs.append(ins)
            continue

        # Path construction, clipping, colour, positioning, state — no marks.
        new_cs.append(ins)

    close()
    return new_cs, element_to_mcids, mcid_to_element, mcid_to_tag


def _xobject_subtype_resolver(obj):
    """Return a callable name -> /Subtype string for an object's /XObject dict."""
    try:
        xobjs = obj.get("/Resources", pikepdf.Dictionary()).get("/XObject", None)
    except Exception:
        xobjs = None

    def resolve(name: str):
        if xobjs is None:
            return None
        try:
            return str(xobjs[name].get("/Subtype"))
        except Exception:
            return None

    return resolve


def inject_bdc_markers(
    pdf: pikepdf.Pdf,
    page: pikepdf.Page,
    page_num: int,
    tagged_elements: list,
) -> tuple[dict, dict, dict]:
    """
    Inject BDC/EMC around tagged text and /Artifact BMC/EMC around every other
    mark-producing operator in the page content stream, so that no content item
    is left outside marked content (PDF/UA-1 clause 7.1-3).

    Allocates PAGE-LOCAL MCIDs (0,1,2,… per page) — this is the single source of
    truth for MCID numbering. The struct tree is then built from the returned
    maps so the page's ParentTree array can be indexed by MCID.

    Returns (element_to_mcids, mcid_to_element, mcid_to_tag) for the page.
    """
    # 1. Map absolute Stage-1 char indices to struct-element keys (element_id,
    #    or cell_id for table cells) and remember each key's PDF tag.
    char_to_key: dict[int, str] = {}
    key_to_tag: dict[str, str] = {}

    def _map_chars(merged_from, key):
        for source_id in merged_from:
            if source_id.startswith(f"p{page_num}_c"):
                try:
                    char_to_key[int(source_id.split("_c")[1])] = key
                except ValueError:
                    pass

    for el in tagged_elements:
        if el.page_num != page_num:
            continue

        if el.pdf_tag.value == "Artifact":
            # Artifacts must NOT receive an MCID — their glyphs fall through to a
            # plain /Artifact BMC in the state machine. Giving them an MCID would
            # emit "/Artifact <</MCID>> BDC" (artifact tagged as real content),
            # which fails PDF/UA clause 7.1-1/7.1-2.
            continue

        if el.pdf_tag.value == "Table" and el.specialist_data.get("cells"):
            for cell in el.specialist_data["cells"]:
                if not cell.get("merged_from"):
                    continue
                cell_id = f"{el.element_id}_cell_{cell['row_idx']}_{cell['col_idx']}"
                key_to_tag[cell_id] = (
                    "TH" if cell.get("is_header") or cell.get("is_row_header") else "TD"
                )
                _map_chars(cell["merged_from"], cell_id)
        else:
            key_to_tag[el.element_id] = el.pdf_tag.value
            _map_chars(el.merged_from, el.element_id)

    # Figure regions (image-only, so they carry no char mappings): image paint
    # ops whose placement falls inside one are tagged /Figure instead of being
    # artifact-wrapped, so Stage 10 can build a /Figure struct elem with /Alt.
    figure_boxes = [
        (el.bbox, el.element_id)
        for el in tagged_elements
        if el.page_num == page_num and el.pdf_tag.value == "Figure"
    ]
    mb = page.obj.get("/MediaBox") or page.mediabox
    page_height_pt = float(mb[3]) - float(mb[1])

    # 2. Parse and rewrite the content stream. Even with no tagged text we still
    #    run the state machine so bare graphics get artifact-wrapped.
    cs = pikepdf.parse_content_stream(page)
    font_bpc = _build_font_bpc(page.obj.get("/Resources"))
    new_cs, element_to_mcids, mcid_to_element, mcid_to_tag = _rewrite_stream(
        cs, char_to_key, key_to_tag, _xobject_subtype_resolver(page.obj),
        figure_boxes, page_height_pt, font_bpc,
    )

    # 3. Write back the new content stream
    page.Contents = pdf.make_stream(pikepdf.unparse_content_stream(new_cs))
    logger.debug(
        "Page %d: allocated %d page-local MCIDs across %d elements.",
        page_num, len(mcid_to_element), len(element_to_mcids),
    )
    return element_to_mcids, mcid_to_element, mcid_to_tag


def artifact_wrap_forms(pdf: pikepdf.Pdf) -> int:
    """Artifact-wrap bare marks inside every Form XObject in the document.

    Form XObjects carry their own content stream with its own marked-content
    scope, so marks drawn inside a form are not covered by page-level wrapping
    and would otherwise fail clause 7.1-3. Each distinct form is rewritten once;
    nested forms are processed recursively. Returns the number of forms rewritten.
    """
    processed: set = set()

    def process_form(form_obj) -> None:
        try:
            oid = form_obj.objgen
        except Exception:
            oid = id(form_obj)
        if oid in processed:
            return
        processed.add(oid)

        try:
            cs = pikepdf.parse_content_stream(form_obj)
        except Exception:
            return
        resolve = _xobject_subtype_resolver(form_obj)
        new_cs, *_ = _rewrite_stream(cs, {}, {}, resolve)
        form_obj.write(pikepdf.unparse_content_stream(new_cs))

        # Recurse into nested forms.
        try:
            xobjs = form_obj.get("/Resources", pikepdf.Dictionary()).get("/XObject", None)
            if xobjs is not None:
                for _name, child in xobjs.items():
                    if str(child.get("/Subtype")) == "/Form":
                        process_form(child)
        except Exception:
            pass

    for page in pdf.pages:
        try:
            xobjs = page.obj.get("/Resources", pikepdf.Dictionary()).get("/XObject", None)
        except Exception:
            xobjs = None
        if xobjs is None:
            continue
        for _name, obj in xobjs.items():
            try:
                if str(obj.get("/Subtype")) == "/Form":
                    process_form(obj)
            except Exception:
                pass

    if processed:
        logger.debug("Artifact-wrapped %d form XObject(s).", len(processed))
    return len(processed)


def detect_cidsets(pdf: pikepdf.Pdf) -> list[Finding]:
    """Detect broken /CIDSet streams on embedded CID FontDescriptors (clause 7.21.4.2-2).

    PDF/UA-1 7.21.4.2 is conditional: IF a CID font's FontDescriptor has a /CIDSet
    it must identify exactly the glyphs present in the embedded program. The source
    PDFs ship subset CID fonts whose inherited /CIDSet is incorrect. UA-1 does not
    REQUIRE /CIDSet, and it is informational only (no effect on rendering), so the
    repair deletes it. One MODIFYING finding per offending FontDescriptor.
    """
    findings: list[Finding] = []
    seen: set = set()
    for page in pdf.pages:
        res = page.obj.get("/Resources")
        if res is None:
            continue
        fonts = res.get("/Font")
        if fonts is None:
            continue
        for _name, f in fonts.items():
            if str(f.get("/Subtype")) != "/Type0":
                continue
            dfs = f.get("/DescendantFonts")
            if dfs is None:
                continue
            for df in dfs:
                fd = df.get("/FontDescriptor")
                if fd is None or fd.objgen in seen:
                    continue
                seen.add(fd.objgen)
                if fd.get("/CIDSet") is None:
                    continue
                base = str(f.get("/BaseFont"))

                def _apply(fd=fd):
                    if fd.get("/CIDSet") is not None:
                        del fd["/CIDSet"]

                findings.append(Finding(
                    clause="7.21.4.2",
                    location=f"font {base} {tuple(fd.objgen)}",
                    defect_description="CIDSet stream does not match the embedded font subset",
                    proposed_repair="Delete the /CIDSet stream from the FontDescriptor",
                    repair_type=MODIFYING,
                    severity="blocks-compliance",
                    auto_safe=True,
                    apply=_apply,
                ))
    return findings


def _strip_notdef_bytes(b: bytes) -> bytes:
    """Drop every 2-byte 0x0000 (CID 0 = .notdef) pair from an Identity-H string."""
    out = bytearray()
    for i in range(0, len(b) - 1, 2):
        if b[i] == 0 and b[i + 1] == 0:
            continue
        out += b[i:i + 2]
    if len(b) % 2:
        out += b[-1:]
    return bytes(out)


def _identity_type0_names(page) -> set:
    """Resource names of Type0 fonts using a 2-byte Identity CMap on this page."""
    res = page.obj.get("/Resources")
    fonts = res.get("/Font") if res is not None else None
    names = set()
    if fonts is None:
        return names
    for n, f in fonts.items():
        if str(f.get("/Subtype")) != "/Type0":
            continue
        if str(f.get("/Encoding")) in ("/Identity-H", "/Identity-V"):
            names.add(str(n))
    return names


def _page_notdef_count(page) -> int:
    """Count .notdef (CID 0) codes in Type0/Identity show ops on a page (no mutation)."""
    idnames = _identity_type0_names(page)
    if not idnames:
        return 0
    cur = None
    count = 0
    for ins in pikepdf.parse_content_stream(page):
        op = str(ins.operator)
        if op == "Tf":
            cur = str(ins.operands[0])
        elif cur in idnames and op in ("Tj", "'", '"'):
            b = bytes(ins.operands[-1])
            count += sum(1 for i in range(0, len(b) - 1, 2) if b[i] == 0 and b[i + 1] == 0)
        elif cur in idnames and op == "TJ":
            for el in ins.operands[0]:
                if isinstance(el, pikepdf.String):
                    b = bytes(el)
                    count += sum(1 for i in range(0, len(b) - 1, 2) if b[i] == 0 and b[i + 1] == 0)
    return count


def _strip_notdef_page(pdf: pikepdf.Pdf, page) -> int:
    """Drop .notdef (CID 0) codes from one page's Type0/Identity show ops."""
    idnames = _identity_type0_names(page)
    if not idnames:
        return 0
    cur = None
    changed = False
    removed = 0
    new_cs = []
    for ins in pikepdf.parse_content_stream(page):
        op = str(ins.operator)
        if op == "Tf":
            cur = str(ins.operands[0])
        elif cur in idnames and op in ("Tj", "'", '"'):
            s = bytes(ins.operands[-1])
            ns = _strip_notdef_bytes(s)
            if ns != s:
                removed += (len(s) - len(ns)) // 2
                ops = list(ins.operands)
                ops[-1] = pikepdf.String(ns)
                ins = pikepdf.ContentStreamInstruction(ops, ins.operator)
                changed = True
        elif cur in idnames and op == "TJ":
            newarr = []
            for el in ins.operands[0]:
                if isinstance(el, pikepdf.String):
                    s = bytes(el)
                    ns = _strip_notdef_bytes(s)
                    if ns != s:
                        removed += (len(s) - len(ns)) // 2
                        changed = True
                    if ns:
                        newarr.append(pikepdf.String(ns))
                else:
                    newarr.append(el)
            ins = pikepdf.ContentStreamInstruction([pikepdf.Array(newarr)], ins.operator)
        new_cs.append(ins)
    if changed:
        page.Contents = pdf.make_stream(pikepdf.unparse_content_stream(new_cs))
    return removed


def detect_notdef_refs(pdf: pikepdf.Pdf) -> list[Finding]:
    """Detect .notdef (CID 0) refs in Type0/Identity show ops (clause 7.21.8-1).

    Source docs show subset Type0 fonts where a real glyph is padded with a
    trailing CID 0 (always .notdef in CFF). UA-1 7.21.8 forbids any .notdef ref in
    a text-showing operator; CID 0 cannot be made non-.notdef, so the repair
    deletes those codes (run-trailing, so rendering-neutral). One MODIFYING finding
    per affected page.
    """
    findings: list[Finding] = []
    for pidx, page in enumerate(pdf.pages):
        n = _page_notdef_count(page)
        if n == 0:
            continue

        def _apply(page=page):
            _strip_notdef_page(pdf, page)

        findings.append(Finding(
            clause="7.21.8",
            location=f"page {pidx + 1}",
            defect_description=f"{n} reference(s) to the .notdef glyph in Type0 show operators",
            proposed_repair="Remove CID-0 (.notdef) codes from the page's show strings",
            repair_type=MODIFYING,
            severity="blocks-compliance",
            auto_safe=True,
            apply=_apply,
        ))
    return findings


def _font_lacks_space(f) -> bool:
    """True if a simple font's embedded program has no glyph for code 32 (space)."""
    fd = f.get("/FontDescriptor")
    if fd is None:
        return False
    prog = fd.get("/FontFile3") or fd.get("/FontFile2") or fd.get("/FontFile")
    if prog is None:
        return False
    try:
        import fitz
        font = fitz.Font(fontbuffer=bytes(prog.read_bytes()))
        return not font.has_glyph(32)
    except Exception:
        return False


def _page_deficient_space(page, cache: dict) -> tuple[set, int]:
    """Return (deficient simple-font resource names, # run-trailing space codes) on a page."""
    res = page.obj.get("/Resources")
    fonts = res.get("/Font") if res is not None else None
    if fonts is None:
        return set(), 0
    deficient = set()
    for n, f in fonts.items():
        if str(f.get("/Subtype")) == "/Type0":
            continue
        key = f.objgen
        if key not in cache:
            cache[key] = _font_lacks_space(f)
        if cache[key]:
            deficient.add(str(n))
    if not deficient:
        return set(), 0
    cur = None
    count = 0
    for ins in pikepdf.parse_content_stream(page):
        op = str(ins.operator)
        if op == "Tf":
            cur = str(ins.operands[0])
        elif cur in deficient and op in ("Tj", "'", '"'):
            s = bytes(ins.operands[-1])
            count += len(s) - len(s.rstrip(b"\x20"))
        elif cur in deficient and op == "TJ":
            arr = list(ins.operands[0])
            for j in range(len(arr) - 1, -1, -1):
                if isinstance(arr[j], pikepdf.String):
                    s = bytes(arr[j])
                    count += len(s) - len(s.rstrip(b"\x20"))
                    break
    return deficient, count


def _strip_space_page(pdf: pikepdf.Pdf, page, deficient: set) -> int:
    """Drop run-trailing space codes from the given deficient fonts on one page."""
    removed = 0
    changed = False
    cur = None
    new_cs = []
    for ins in pikepdf.parse_content_stream(page):
        op = str(ins.operator)
        if op == "Tf":
            cur = str(ins.operands[0])
        elif cur in deficient and op in ("Tj", "'", '"'):
            s = bytes(ins.operands[-1])
            ns = s.rstrip(b"\x20")
            if ns != s:
                removed += len(s) - len(ns)
                ops = list(ins.operands)
                ops[-1] = pikepdf.String(ns)
                ins = pikepdf.ContentStreamInstruction(ops, ins.operator)
                changed = True
        elif cur in deficient and op == "TJ":
            arr = list(ins.operands[0])
            for j in range(len(arr) - 1, -1, -1):
                if isinstance(arr[j], pikepdf.String):
                    s = bytes(arr[j])
                    ns = s.rstrip(b"\x20")
                    if ns != s:
                        removed += len(s) - len(ns)
                        changed = True
                        arr[j] = pikepdf.String(ns)
                    break
            arr = [e for e in arr
                   if not (isinstance(e, pikepdf.String) and len(bytes(e)) == 0)]
            ins = pikepdf.ContentStreamInstruction([pikepdf.Array(arr)], ins.operator)
        new_cs.append(ins)
    if changed:
        page.Contents = pdf.make_stream(pikepdf.unparse_content_stream(new_cs))
    return removed


def detect_unembedded_fonts(pdf: pikepdf.Pdf) -> list[Finding]:
    """Detect fonts whose program is not embedded (clause 7.21.4.1-1).

    PDF/UA-1 requires every font to be embedded. The clean corpus sources use
    system fonts (Arial, Times New Roman) the producer never embedded — an
    INHERITED source defect, not one the pipeline introduces. Embedding a
    substitute is a MODIFYING repair that ALTERS rendering (different glyph
    outlines) and carries licensing concerns, so it is NOT auto-safe and we ship
    no automatic apply: the finding is surfaced (report-only) through the gate
    rather than silently embedding a font. One finding per non-embedded font.
    """
    findings: list[Finding] = []
    seen: set = set()

    def _descriptors(f):
        if str(f.get("/Subtype")) == "/Type0":
            for df in (f.get("/DescendantFonts") or []):
                yield df.get("/FontDescriptor")
        else:
            yield f.get("/FontDescriptor")

    for page in pdf.pages:
        res = page.obj.get("/Resources")
        fonts = res.get("/Font") if res is not None else None
        if fonts is None:
            continue
        for _name, f in fonts.items():
            base = str(f.get("/BaseFont"))
            for fd in _descriptors(f):
                # Key on the descriptor (or the font itself for the no-descriptor
                # standard-14 case) so a font used on several pages counts once.
                key = fd.objgen if fd is not None else f.objgen
                if key in seen:
                    continue
                seen.add(key)
                if fd is not None and any(
                    fd.get(k) is not None
                    for k in ("/FontFile", "/FontFile2", "/FontFile3")
                ):
                    continue  # embedded
                findings.append(Finding(
                    clause="7.21.4.1",
                    location=f"font {base} {tuple(key)}",
                    defect_description="Font program is not embedded",
                    proposed_repair=(
                        "Embed the font program (or a metrically compatible "
                        "substitute); not applied automatically — rendering-"
                        "altering and licensing-sensitive"
                    ),
                    repair_type=MODIFYING,
                    severity="blocks-compliance",
                    auto_safe=False,
                    apply=None,
                ))
    return findings


def detect_missing_space_refs(pdf: pikepdf.Pdf) -> list[Finding]:
    """Detect space refs to fonts whose program lacks the glyph (clause 7.21.4.1-2).

    Some source CFF subset fonts reference the space glyph (code 32) their embedded
    program does not contain; UA-1 7.21.4.1 requires every referenced glyph be
    present. Scoped strictly to glyph-deficient simple fonts (detected via fitz
    has_glyph, already a pipeline dependency) so fonts that legitimately contain the
    space are untouched; only run-trailing spaces are removed (rendering-neutral;
    /ActualText preserves the space for AT). One MODIFYING finding per affected page.
    """
    findings: list[Finding] = []
    cache: dict = {}
    for pidx, page in enumerate(pdf.pages):
        deficient, count = _page_deficient_space(page, cache)
        if count == 0:
            continue

        def _apply(page=page, deficient=deficient):
            _strip_space_page(pdf, page, deficient)

        findings.append(Finding(
            clause="7.21.4.1",
            location=f"page {pidx + 1}",
            defect_description=f"{count} space reference(s) to font(s) lacking the space glyph",
            proposed_repair="Remove run-trailing space codes for the glyph-deficient fonts",
            repair_type=MODIFYING,
            severity="blocks-compliance",
            auto_safe=True,
            apply=_apply,
        ))
    return findings
