"""Tests for Stage 10 clause 7.1-3: artifact wrapping + page-local MCID
allocation + MCID-indexed ParentTree resolution."""

import pikepdf

from tagger.models.data_types import PDFTag, TaggedElement
from tagger.stage10_writeback.content_stream_writer import (
    _rewrite_stream,
    artifact_wrap_forms,
    inject_bdc_markers,
)
from tagger.stage10_writeback.struct_tree_writer import tag_untagged_pdf

_MC_OPEN = {"BDC", "BMC"}
_PAINT = {"f", "F", "f*", "S", "s", "B", "B*", "b", "b*", "sh", "Tj", "TJ", "'", '"'}


def _parse(data: bytes):
    pdf = pikepdf.new()
    stream = pdf.make_stream(data)
    return list(pikepdf.parse_content_stream(stream))


def _walk(instructions, do_subtype=lambda n: None):
    depth = 0
    art_stack = []
    mark_depths = []
    mcid_opens = []
    barrier_artifact_open = []
    art_open = lambda: any(art_stack)

    for ins in instructions:
        op = str(ins.operator)
        if op in _MC_OPEN:
            is_art = op == "BMC" and str(ins.operands[0]) == "/Artifact"
            if op == "BDC" and len(ins.operands) >= 2:
                try:
                    if "/MCID" in ins.operands[1]:
                        mcid_opens.append(int(ins.operands[1]["/MCID"]))
                except Exception:
                    pass
            art_stack.append(is_art)
            depth += 1
            continue
        if op == "EMC":
            depth -= 1
            if art_stack:
                art_stack.pop()
            assert depth >= 0, "EMC without matching BMC/BDC"
            continue
        if op in ("q", "Q", "BT", "ET"):
            barrier_artifact_open.append(art_open())
        if op == "Do":
            sub = do_subtype(str(ins.operands[0]) if ins.operands else "")
            if sub == "/Form":
                continue
            mark_depths.append(depth)
            continue
        if op in _PAINT or op == "INLINE_IMAGE":
            mark_depths.append(depth)

    return {
        "final_depth": depth,
        "mark_depths": mark_depths,
        "mcid_opens": mcid_opens,
        "barrier_artifact_open": barrier_artifact_open,
    }


def test_every_mark_wrapped_and_nested():
    data = (
        b"BT /F1 12 Tf 100 700 Td (Hello) Tj ( ) Tj (World) Tj ET\n"
        b"q 1 0 0 1 0 0 cm 10 10 100 100 re f Q\n"
        b"0 0 1 RG 5 5 m 50 50 l S\n"
        b"/Im0 Do\n"
        b"q /Fo0 Do Q\n"
        b"BT 200 200 Td (foot) Tj ET\n"
    )
    instrs = _parse(data)
    # "Hello"=idx0..4, " "=idx5 (unmapped), "World"=idx6..10 -> element "p".
    char_to_key = {i: "p" for i in list(range(0, 5)) + list(range(6, 11))}
    key_to_tag = {"p": "P"}
    subs = {"/Im0": "/Image", "/Fo0": "/Form"}
    do_subtype = lambda n: subs.get(n)

    new_cs, element_to_mcids, mcid_to_element, mcid_to_tag = _rewrite_stream(
        instrs, char_to_key, key_to_tag, do_subtype
    )
    facts = _walk(new_cs, do_subtype)

    assert element_to_mcids == {"p": [0]}
    assert mcid_to_element == {0: "p"}
    assert facts["final_depth"] == 0
    assert all(d >= 1 for d in facts["mark_depths"]), "a mark sits outside marked content"
    assert facts["mcid_opens"] == [0], f"unexpected MCID opens: {facts['mcid_opens']}"
    assert not any(facts["barrier_artifact_open"]), "artifact MC straddles a barrier"


def test_split_element_gets_distinct_mcids():
    # Element A is interrupted by element B, then resumes -> two runs, two MCIDs.
    data = b"BT (AAA) Tj (BBB) Tj (AAA) Tj ET\n"
    instrs = _parse(data)
    char_to_key = {0: "a", 1: "a", 2: "a", 3: "b", 4: "b", 5: "b", 6: "a", 7: "a", 8: "a"}
    key_to_tag = {"a": "P", "b": "H1"}
    new_cs, e2m, m2e, m2t = _rewrite_stream(instrs, char_to_key, key_to_tag, lambda n: None)

    assert e2m == {"a": [0, 2], "b": [1]}, e2m
    assert m2e == {0: "a", 1: "b", 2: "a"}
    # Page-local, contiguous from 0.
    assert sorted(m2e) == list(range(len(m2e)))
    assert m2t == {0: "P", 1: "H1", 2: "P"}


def test_no_content_operators_dropped_or_reordered():
    data = (
        b"BT /F1 12 Tf 100 700 Td (Hello) Tj ( ) Tj (World) Tj ET\n"
        b"q 10 10 100 100 re f Q\n"
        b"5 5 m 50 50 l S\n"
    )
    instrs = _parse(data)
    char_to_key = {i: "p" for i in list(range(0, 5)) + list(range(6, 11))}
    new_cs, *_ = _rewrite_stream(instrs, char_to_key, {"p": "P"}, lambda n: None)

    def content_ops(seq):
        return [str(i.operator) for i in seq if str(i.operator) not in {"BDC", "BMC", "EMC"}]

    assert content_ops(new_cs) == content_ops(instrs)


def test_empty_mapping_artifacts_everything():
    data = b"10 10 100 100 re f\nBT (x) Tj ET\n"
    instrs = _parse(data)
    new_cs, e2m, m2e, m2t = _rewrite_stream(instrs, {}, {}, lambda n: None)
    facts = _walk(new_cs)
    assert e2m == {} and m2e == {}
    assert facts["final_depth"] == 0
    assert all(d >= 1 for d in facts["mark_depths"])
    assert facts["mcid_opens"] == []


def test_form_xobject_marks_wrapped():
    pdf = pikepdf.new()
    form = pdf.make_stream(b"10 10 50 50 re f\n")
    form["/Type"] = pikepdf.Name("/XObject")
    form["/Subtype"] = pikepdf.Name("/Form")
    form["/BBox"] = pikepdf.Array([0, 0, 100, 100])
    form_ref = pdf.make_indirect(form)
    page = pdf.add_blank_page(page_size=(200, 200))
    page.obj["/Resources"] = pikepdf.Dictionary(
        {"/XObject": pikepdf.Dictionary({"/Fo0": form_ref})}
    )
    page.obj["/Contents"] = pdf.make_stream(b"q /Fo0 Do Q\n")

    assert artifact_wrap_forms(pdf) == 1
    facts = _walk(list(pikepdf.parse_content_stream(form_ref)))
    assert facts["final_depth"] == 0
    assert all(d >= 1 for d in facts["mark_depths"])


def _resolve_failures(pdf_path):
    """For every MCID opened in each page stream, walk
    page->/StructParents->ParentTree->array[mcid]->struct element. Returns the
    list of (page_idx, mcid) that fail to resolve to a struct element with /S."""
    fails = []
    with pikepdf.open(str(pdf_path)) as pdf:
        st = pdf.Root.get("/StructTreeRoot")
        assert st is not None, "no StructTreeRoot"
        nums = st.ParentTree.Nums
        key_to_arr = {int(nums[i]): nums[i + 1] for i in range(0, len(nums), 2)}
        for pi, page in enumerate(pdf.pages):
            sp = page.obj.get("/StructParents")
            arr = key_to_arr.get(int(sp)) if sp is not None else None
            for ins in pikepdf.parse_content_stream(page):
                if str(ins.operator) == "BDC" and len(ins.operands) >= 2 and "/MCID" in ins.operands[1]:
                    m = int(ins.operands[1]["/MCID"])
                    ok = arr is not None and m < len(arr)
                    if ok:
                        try:
                            ok = "/S" in arr[m]
                        except Exception:
                            ok = False
                    if not ok:
                        fails.append((pi, m))
    return fails


def test_end_to_end_every_mcid_resolves(tmp_path):
    """Synthetic Stage-10 run: build a 2-page PDF + tagged elements, tag it, and
    assert every content-stream MCID resolves through the ParentTree to a struct
    element (the deterministic proxy for veraPDF 7.1-3 = 0)."""
    src = pdf = pikepdf.new()
    # Page 1: two paragraphs. Page 2: one heading split across two text objects
    # of the same element (forces multi-MCID on a later page).
    p1 = pdf.add_blank_page(page_size=(612, 792))
    p1.obj["/Contents"] = pdf.make_stream(
        b"BT /F1 12 Tf 72 700 Td (Hello) Tj ET\nBT 72 680 Td (World) Tj ET\n"
        b"10 10 50 50 re f\n"  # bare graphic -> artifact
    )
    p2 = pdf.add_blank_page(page_size=(612, 792))
    p2.obj["/Contents"] = pdf.make_stream(
        b"BT /F1 12 Tf 72 700 Td (Title) Tj ET\nBT 72 680 Td (Title) Tj ET\n"
    )
    input_path = tmp_path / "in.pdf"
    pdf.save(str(input_path))
    pdf.close()

    els = [
        TaggedElement(element_id="e0", page_num=1, pdf_tag=PDFTag.H1, text="Hello",
                      bbox=(72, 92, 140, 104), merged_from=[f"p1_c{i}" for i in range(0, 5)]),
        TaggedElement(element_id="e1", page_num=1, pdf_tag=PDFTag.P, text="World",
                      bbox=(72, 112, 140, 124), merged_from=[f"p1_c{i}" for i in range(5, 10)]),
        # Page 2: one element whose glyphs are drawn in two text objects -> split.
        TaggedElement(element_id="e2", page_num=2, pdf_tag=PDFTag.H2, text="TitleTitle",
                      bbox=(72, 92, 140, 104), merged_from=[f"p2_c{i}" for i in range(0, 10)]),
    ]
    out_path = tmp_path / "out.pdf"
    tag_untagged_pdf(str(input_path), str(out_path), els, total_pages=2)

    fails = _resolve_failures(out_path)
    assert fails == [], f"unresolved content MCIDs: {fails}"

    # The split element on page 2 must have produced two distinct MCIDs.
    with pikepdf.open(str(out_path)) as o:
        cs = pikepdf.parse_content_stream(o.pages[1])
        mcids = [int(i.operands[1]["/MCID"]) for i in cs
                 if str(i.operator) == "BDC" and len(i.operands) >= 2 and "/MCID" in i.operands[1]]
    assert len(mcids) == 2 and len(set(mcids)) == 2, mcids
    assert sorted(mcids) == [0, 1], "page-2 MCIDs must be page-local (0,1)"


def test_end_to_end_table_and_list_resolve(tmp_path):
    """Exercise the table-cell and list (LBody) struct paths end to end and
    assert every content MCID resolves — these are the newest /K constructions."""
    pdf = pikepdf.new()
    # Page 1: a 1-column, 2-row table.
    t = pdf.add_blank_page(page_size=(612, 792))
    t.obj["/Contents"] = pdf.make_stream(
        b"BT /F1 12 Tf 72 700 Td (Hdr) Tj ET\nBT 72 680 Td (Val) Tj ET\n"
    )
    # Page 2: two list items.
    l = pdf.add_blank_page(page_size=(612, 792))
    l.obj["/Contents"] = pdf.make_stream(
        b"BT /F1 12 Tf 72 700 Td (one) Tj ET\nBT 72 680 Td (two) Tj ET\n"
    )
    input_path = tmp_path / "in2.pdf"
    pdf.save(str(input_path))
    pdf.close()

    table = TaggedElement(
        element_id="t0", page_num=1, pdf_tag=PDFTag.TABLE, text="Hdr Val",
        bbox=(72, 92, 200, 124),
        specialist_data={"cells": [
            {"row_idx": 0, "col_idx": 0, "text": "Hdr", "is_header": True,
             "merged_from": [f"p1_c{i}" for i in range(0, 3)]},
            {"row_idx": 1, "col_idx": 0, "text": "Val", "is_header": False,
             "merged_from": [f"p1_c{i}" for i in range(3, 6)]},
        ]},
    )
    li1 = TaggedElement(element_id="li1", page_num=2, pdf_tag=PDFTag.LI, text="one",
                        bbox=(72, 92, 140, 104), merged_from=[f"p2_c{i}" for i in range(0, 3)],
                        specialist_data={"list_label": "1.", "list_body": "one"})
    li2 = TaggedElement(element_id="li2", page_num=2, pdf_tag=PDFTag.LI, text="two",
                        bbox=(72, 112, 140, 124), merged_from=[f"p2_c{i}" for i in range(3, 6)],
                        specialist_data={"list_label": "2.", "list_body": "two"})

    out_path = tmp_path / "out2.pdf"
    tag_untagged_pdf(str(input_path), str(out_path), [table, li1, li2], total_pages=2)

    assert _resolve_failures(out_path) == [], "table/list content MCIDs must resolve"

    # LI must be parented (via /P) by the L container, not Document (clause 7.2-17).
    with pikepdf.open(str(out_path)) as o:
        def find_lis(node, out):
            if not isinstance(node, pikepdf.Dictionary):
                return
            if str(node.get("/S")) == "/LI":
                out.append(node)
            k = node.get("/K")
            if isinstance(k, pikepdf.Array):
                for c in k:
                    find_lis(c, out)
            elif isinstance(k, pikepdf.Dictionary):
                find_lis(k, out)
        lis = []
        find_lis(o.Root.StructTreeRoot.K, lis)
        assert lis, "expected LI elements"
        for li in lis:
            assert str(li.P.get("/S")) == "/L", "LI parent (/P) must be L"


def test_end_to_end_toci_wrapped_in_toc(tmp_path):
    """Every TOCI must be a child of a TOC (PDF/UA 7.2-26), never directly under
    Document. A consecutive TOCI run groups into one TOC."""
    pdf = pikepdf.new()
    p = pdf.add_blank_page(page_size=(612, 792))
    p.obj["/Contents"] = pdf.make_stream(
        b"BT /F1 12 Tf 72 700 Td (One) Tj ET\nBT 72 680 Td (Two) Tj ET\n"
    )
    input_path = tmp_path / "toc.pdf"
    pdf.save(str(input_path))
    pdf.close()

    els = [
        TaggedElement(element_id="c0", page_num=1, pdf_tag=PDFTag.TOCI, text="One",
                      bbox=(72, 92, 140, 104), merged_from=[f"p1_c{i}" for i in range(0, 3)]),
        TaggedElement(element_id="c1", page_num=1, pdf_tag=PDFTag.TOCI, text="Two",
                      bbox=(72, 112, 140, 124), merged_from=[f"p1_c{i}" for i in range(3, 6)]),
    ]
    out_path = tmp_path / "toc_out.pdf"
    tag_untagged_pdf(str(input_path), str(out_path), els, total_pages=1)

    assert _resolve_failures(out_path) == [], "TOCI content MCIDs must resolve"

    with pikepdf.open(str(out_path)) as o:
        doc = o.Root.StructTreeRoot.K
        doc_kids = list(doc.K)
        tocs = [k for k in doc_kids if str(k.get("/S")) == "/TOC"]
        assert len(tocs) == 1, f"expected one TOC container, got {len(tocs)}"
        tocis = list(tocs[0].K)
        assert len(tocis) == 2, "TOC should hold both TOCI entries"
        for t in tocis:
            assert str(t.get("/S")) == "/TOCI"
            assert str(t.P.get("/S")) == "/TOC", "TOCI parent must be TOC"
        assert not [k for k in doc_kids if str(k.get("/S")) == "/TOCI"], \
            "no TOCI may sit directly under Document"


def test_artifact_element_never_gets_mcid(tmp_path):
    """An ARTIFACT element's glyphs must become a plain /Artifact BMC, never a
    `/Artifact <</MCID>>` BDC (which is artifact-tagged-as-real-content and fails
    veraPDF 7.1-1/7.1-2)."""
    pdf = pikepdf.new()
    page = pdf.add_blank_page(page_size=(612, 792))
    page.obj["/Contents"] = pdf.make_stream(
        b"BT /F1 12 Tf 72 700 Td (Real) Tj ET\nBT 72 40 Td (Foot) Tj ET\n"
    )
    e0 = TaggedElement(element_id="e0", page_num=1, pdf_tag=PDFTag.P, text="Real",
                       bbox=(72, 92, 140, 104), merged_from=[f"p1_c{i}" for i in range(0, 4)])
    eA = TaggedElement(element_id="eA", page_num=1, pdf_tag=PDFTag.ARTIFACT, text="Foot",
                       bbox=(72, 752, 140, 764), merged_from=[f"p1_c{i}" for i in range(4, 8)])

    element_to_mcids, _m2e, mcid_to_tag = inject_bdc_markers(pdf, page, 1, [e0, eA])

    assert "eA" not in element_to_mcids, "artifact element must not receive an MCID"
    assert "e0" in element_to_mcids
    assert "Artifact" not in mcid_to_tag.values(), "no MCID may carry the /Artifact tag"

    cs = list(pikepdf.parse_content_stream(page))
    artifact_bdc = [i for i in cs if str(i.operator) == "BDC" and str(i.operands[0]) == "/Artifact"]
    plain_artifact = [i for i in cs if str(i.operator) == "BMC" and str(i.operands[0]) == "/Artifact"]
    assert not artifact_bdc, "artifact must not be emitted as a BDC with MCID"
    assert plain_artifact, "artifact glyphs should be wrapped in a plain /Artifact BMC"
