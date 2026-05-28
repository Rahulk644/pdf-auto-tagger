import logging
import pikepdf

logger = logging.getLogger(__name__)

# Text-showing operators (the only ops that consume Stage-1 characters).
_TEXT_OPS = {"Tj", "TJ", "'", '"'}
# Operators that paint visible marks directly on the current content stream.
# (Do is resolved further: /Image marks here, /Form recurses separately.)
_PAINT_OPS = {"f", "F", "f*", "S", "s", "B", "B*", "b", "b*", "sh", "Do", "INLINE_IMAGE"}
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


def _text_len(instruction, op: str) -> int:
    """Number of glyphs drawn by a text-showing operator."""
    if op == "TJ":
        n = 0
        for item in instruction.operands[0]:
            if isinstance(item, pikepdf.String):
                n += len(str(item))
        return n
    if op == '"':
        # aw ac string " — the string is the last operand.
        return len(str(instruction.operands[-1]))
    # Tj and ' both take a single string operand.
    return len(str(instruction.operands[0]))


def _rewrite_stream(instructions, char_to_mcid, mcid_to_tag, do_subtype):
    """Run the marked-content state machine over a parsed content stream.

    Guarantees that every mark-producing operator ends up inside either a tagged
    BDC (text mapped to an MCID) or an /Artifact BMC sequence, with every
    sequence wholly contained within one text object and — for artifacts — one
    graphics-state level (proper nesting).

    char_to_mcid / mcid_to_tag describe page text tagging; pass empty dicts for
    form streams (everything becomes an artifact). ``do_subtype`` maps an XObject
    resource name to its /Subtype string so /Form invocations can be passed
    through (their inner marks are wrapped by the form pass) while /Image
    invocations are artifact-wrapped here.

    Returns (new_instructions, injected_mcids).
    """
    new_cs = []
    injected = set()
    current_char_idx = 0
    active = None  # None | ("tag", mcid) | ("art",)

    def close():
        nonlocal active
        if active is not None:
            new_cs.append(_emc())
            active = None

    def open_tag(mcid):
        nonlocal active
        injected.add(mcid)
        new_cs.append(_tag_bdc(mcid, mcid_to_tag[mcid]))
        active = ("tag", mcid)

    def open_art():
        nonlocal active
        new_cs.append(_artifact_bmc())
        active = ("art",)

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
            new_cs.append(ins)
            continue

        if op in _TEXT_OPS:
            op_len = _text_len(ins, op)
            target = None
            for i in range(current_char_idx, current_char_idx + op_len):
                if i in char_to_mcid:
                    target = char_to_mcid[i]
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
            if active != ("art",):
                close()
                open_art()
            new_cs.append(ins)
            continue

        # Path construction, clipping, colour, positioning, state — no marks.
        new_cs.append(ins)

    close()
    return new_cs, injected


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
    element_mcid_map: dict[str, int],
) -> set[int]:
    """
    Inject BDC/EMC around tagged text and /Artifact BMC/EMC around every other
    mark-producing operator in the page content stream, so that no content item
    is left outside marked content (PDF/UA-1 clause 7.1-3).

    Returns the set of MCIDs that were successfully injected.
    """
    # 1. Map absolute character indices to MCIDs based on TaggedElement merged_from
    char_to_mcid = {}
    mcid_to_tag = {}
    for el in tagged_elements:
        if el.page_num != page_num:
            continue

        if el.pdf_tag.value == "Table" and el.specialist_data.get("cells"):
            for cell in el.specialist_data["cells"]:
                if not cell.get("merged_from"):
                    continue
                cell_id = f"{el.element_id}_cell_{cell['row_idx']}_{cell['col_idx']}"
                if cell_id not in element_mcid_map:
                    continue

                mcid = element_mcid_map[cell_id]
                td_tag = "TH" if cell.get("is_header") or cell.get("is_row_header") else "TD"
                mcid_to_tag[mcid] = td_tag

                for source_id in cell["merged_from"]:
                    if source_id.startswith(f"p{page_num}_c"):
                        try:
                            char_idx = int(source_id.split("_c")[1])
                            char_to_mcid[char_idx] = mcid
                        except ValueError:
                            pass
        else:
            if el.element_id not in element_mcid_map:
                continue

            mcid = element_mcid_map[el.element_id]
            mcid_to_tag[mcid] = el.pdf_tag.value

            for source_id in el.merged_from:
                # Source IDs are like "p1_c5"
                if source_id.startswith(f"p{page_num}_c"):
                    try:
                        char_idx = int(source_id.split("_c")[1])
                        char_to_mcid[char_idx] = mcid
                    except ValueError:
                        pass

    logger.debug(
        "Page %d: char_to_mcid size=%d, unique MCIDs=%s",
        page_num, len(char_to_mcid), set(char_to_mcid.values()),
    )

    # 2. Parse and rewrite the content stream. Even with no tagged text we still
    #    run the state machine so bare graphics get artifact-wrapped.
    cs = pikepdf.parse_content_stream(page)
    new_cs, injected_mcids = _rewrite_stream(
        cs, char_to_mcid, mcid_to_tag, _xobject_subtype_resolver(page.obj)
    )

    # 3. Write back the new content stream
    page.Contents = pdf.make_stream(pikepdf.unparse_content_stream(new_cs))
    logger.debug(f"Page {page_num}: Injected BDC markers for {len(injected_mcids)} MCIDs.")
    return injected_mcids


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
        new_cs, _ = _rewrite_stream(cs, {}, {}, resolve)
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
