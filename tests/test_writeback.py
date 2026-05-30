"""Tests for Stage 10 — Struct tree writeback."""

import json
import pytest
from pathlib import Path

import pikepdf

from tagger.pipeline import AutoTaggerPipeline
from tagger.stage10_writeback.struct_tree_writer import tag_untagged_pdf
from tagger.models.data_types import PDFTag, TaggedElement
from tagger.config import LAYOUT

# Some integration assertions encode MinerU's specific layout output (e.g. exact
# heading levels). The CPU backend (TAGGER_LAYOUT_BACKEND=cpu — used to run the suite
# locally without spawning MinerU) produces a valid but different layout, so gate them.
mineru_only = pytest.mark.skipif(
    LAYOUT.backend != "mineru",
    reason="asserts MinerU-specific layout output; run with the mineru backend",
)


class TestStructTreeWriter:
    """Tests for struct tree creation."""

    def test_creates_struct_tree(self, tmp_path):
        """Writeback should create a valid struct tree."""
        output_pdf = tmp_path / "tagged.pdf"

        pipeline = AutoTaggerPipeline()
        pipeline.run(
            input_pdf="tests/fixtures/sample.pdf",
            output_pdf=str(output_pdf),
        )

        assert output_pdf.exists()

        # Verify struct tree
        pdf = pikepdf.open(str(output_pdf))
        root = pdf.Root

        # Check MarkInfo
        mark_info = root.get("/MarkInfo")
        assert mark_info is not None

        # Check StructTreeRoot
        str_root = root.get("/StructTreeRoot")
        assert str_root is not None
        assert str(str_root.get("/Type")) == "/StructTreeRoot"

        # Check Document element
        doc_elem = str_root.get("/K")
        assert doc_elem is not None
        assert str(doc_elem.get("/S")) == "/Document"

        # Check struct elements
        k_array = doc_elem.get("/K")
        assert len(k_array) > 0

        # Check Lang
        assert root.get("/Lang") is not None

        pdf.close()

    @mineru_only
    def test_heading_tags_in_struct_tree(self, tmp_path):
        """H1 and H2 should appear in struct tree."""
        output_pdf = tmp_path / "tagged.pdf"

        pipeline = AutoTaggerPipeline()
        pipeline.run(
            input_pdf="tests/fixtures/sample.pdf",
            output_pdf=str(output_pdf),
        )

        pdf = pikepdf.open(str(output_pdf))
        str_root = pdf.Root.get("/StructTreeRoot")
        doc_elem = str_root.get("/K")
        k_array = doc_elem.get("/K")

        tags = [str(elem.get("/S")) for elem in k_array]

        assert "/H1" in tags, f"Expected /H1 in {tags}"
        assert "/H2" in tags, f"Expected /H2 in {tags}"
        assert "/P" in tags, f"Expected /P in {tags}"

        pdf.close()

    def test_artifacts_excluded(self, tmp_path):
        """Artifacts should NOT appear in struct tree."""
        output_pdf = tmp_path / "tagged.pdf"

        pipeline = AutoTaggerPipeline()
        pipeline.run(
            input_pdf="tests/fixtures/sample.pdf",
            output_pdf=str(output_pdf),
        )

        pdf = pikepdf.open(str(output_pdf))
        str_root = pdf.Root.get("/StructTreeRoot")
        doc_elem = str_root.get("/K")
        k_array = doc_elem.get("/K")

        tags = [str(elem.get("/S")) for elem in k_array]
        assert "/Artifact" not in tags

        pdf.close()

    def test_actual_text_preserved(self, tmp_path):
        """ActualText should be set on struct elements."""
        output_pdf = tmp_path / "tagged.pdf"

        pipeline = AutoTaggerPipeline()
        pipeline.run(
            input_pdf="tests/fixtures/sample.pdf",
            output_pdf=str(output_pdf),
        )

        pdf = pikepdf.open(str(output_pdf))
        str_root = pdf.Root.get("/StructTreeRoot")
        doc_elem = str_root.get("/K")
        k_array = doc_elem.get("/K")

        # H1 element should have ActualText
        h1 = [e for e in k_array if str(e.get("/S")) == "/H1"][0]
        actual_text = str(h1.get("/ActualText"))
        assert "Auto-Tagger" in actual_text

        pdf.close()

    def test_page_tabs_set(self, tmp_path):
        """All pages should have /Tabs /S for structure reading order."""
        output_pdf = tmp_path / "tagged.pdf"

        pipeline = AutoTaggerPipeline()
        pipeline.run(
            input_pdf="tests/fixtures/sample.pdf",
            output_pdf=str(output_pdf),
        )

        pdf = pikepdf.open(str(output_pdf))
        for page in pdf.pages:
            tabs = page.obj.get("/Tabs")
            assert tabs is not None
            assert str(tabs) == "/S"

        pdf.close()


def test_struct_tree_preserves_reading_order_not_geometry(tmp_path):
    """Stage 10 must build the struct tree in the pipeline's reading order, NOT a
    geometric (top, left) re-sort. On a 2-column page a (top, left) sort interleaves
    the columns (a right-column line at top=90 lands before a left-column line at
    top=100). The incoming TaggedElement list is already column-aware reading order;
    the struct /K order must match it. Regression guard for the multi-column
    reading-order remediation fix (struct_tree_writer line 543)."""
    pdf = pikepdf.new()
    page = pdf.add_blank_page(page_size=(612, 792))
    # Physical content-stream order = the canonical reading order (one Tj run each).
    page.obj["/Contents"] = pdf.make_stream(
        b"BT /F1 12 Tf 72 740 Td (Title) Tj ET\n"      # chars 0-4
        b"BT 72 690 Td (LeftOne) Tj ET\n"               # chars 5-11
        b"BT 72 640 Td (LeftTwo) Tj ET\n"               # chars 12-18
        b"BT 320 700 Td (RightOne) Tj ET\n"             # chars 19-26
        b"BT 320 650 Td (RightTwo) Tj ET\n"             # chars 27-34
    )
    input_path = tmp_path / "twocol.pdf"
    pdf.save(str(input_path))
    pdf.close()

    def _mf(a, b):
        return [f"p1_c{i}" for i in range(a, b)]

    # bbox = (x0, y0, x1, y1) in standard top-left coords; the sort key was (y0, x0).
    # Incoming order is column-aware: title, full LEFT column, then full RIGHT column.
    # A (top, left) sort would instead yield Title, RightOne(70), LeftOne(100),
    # RightTwo(140), LeftTwo(150) — interleaving the columns.
    els = [
        TaggedElement(element_id="title", page_num=1, pdf_tag=PDFTag.H1, text="Title",
                      bbox=(72, 40, 300, 60), merged_from=_mf(0, 5)),
        TaggedElement(element_id="L1", page_num=1, pdf_tag=PDFTag.P, text="LeftOne",
                      bbox=(72, 100, 300, 120), merged_from=_mf(5, 12)),
        TaggedElement(element_id="L2", page_num=1, pdf_tag=PDFTag.P, text="LeftTwo",
                      bbox=(72, 150, 300, 170), merged_from=_mf(12, 19)),
        TaggedElement(element_id="R1", page_num=1, pdf_tag=PDFTag.P, text="RightOne",
                      bbox=(320, 70, 550, 90), merged_from=_mf(19, 27)),
        TaggedElement(element_id="R2", page_num=1, pdf_tag=PDFTag.P, text="RightTwo",
                      bbox=(320, 140, 550, 160), merged_from=_mf(27, 35)),
    ]
    out_path = tmp_path / "twocol_out.pdf"
    tag_untagged_pdf(str(input_path), str(out_path), els, total_pages=1)

    with pikepdf.open(str(out_path)) as o:
        kids = list(o.Root.StructTreeRoot.K.K)
        order = [str(k.get("/ActualText")) for k in kids]

    assert order == ["Title", "LeftOne", "LeftTwo", "RightOne", "RightTwo"], (
        f"struct /K not in reading order (got {order}); a geometric re-sort would "
        f"interleave columns as Title, RightOne, LeftOne, RightTwo, LeftTwo")


def test_widget_annotations_tagged_as_form(tmp_path):
    """An untagged form field (Widget annotation) must be wrapped in a /Form struct
    element with an OBJR back to the widget, get a fresh /StructParent, and — when it
    has no /TU — an accessible name backfilled from its field name (/T). Regression
    guard for the form-tagging gap (incumbent-baseline analysis: 16% of source PDFs
    carry widgets; PDF/UA 7.18.1 / Matterhorn requires each be tagged)."""
    pdf = pikepdf.new()
    page = pdf.add_blank_page(page_size=(612, 792))
    page.obj["/Contents"] = pdf.make_stream(b"BT /F1 12 Tf 72 740 Td (Form) Tj ET\n")
    widget = pdf.make_indirect(pikepdf.Dictionary({
        "/Type": pikepdf.Name.Annot,
        "/Subtype": pikepdf.Name.Widget,
        "/FT": pikepdf.Name.Tx,
        "/T": pikepdf.String("Full Name"),
        "/Rect": pikepdf.Array([72, 600, 300, 620]),
    }))
    page.obj["/Annots"] = pikepdf.Array([widget])
    input_path = tmp_path / "form.pdf"
    pdf.save(str(input_path))
    pdf.close()

    els = [TaggedElement(element_id="p1_c0", page_num=1, pdf_tag=PDFTag.P, text="Form",
                         bbox=(72, 40, 300, 60), merged_from=["p1_c0", "p1_c1", "p1_c2", "p1_c3"])]
    out_path = tmp_path / "form_out.pdf"
    tag_untagged_pdf(str(input_path), str(out_path), els, total_pages=1)

    with pikepdf.open(str(out_path)) as o:
        # collect every /S in the struct tree
        s_tags = []
        objr_targets = []
        def walk(n):
            if isinstance(n, pikepdf.Dictionary):
                if n.get("/S") is not None:
                    s_tags.append(str(n.get("/S")))
                if str(n.get("/Type", "")) == "/OBJR" and n.get("/Obj") is not None:
                    objr_targets.append(n["/Obj"])
                if n.get("/K") is not None:
                    walk(n.get("/K"))
            elif isinstance(n, pikepdf.Array):
                for x in n:
                    walk(x)
        walk(o.Root.StructTreeRoot.K)
        assert "/Form" in s_tags, f"expected a /Form struct element, got {s_tags}"
        # the widget annotation now carries a StructParent and an accessible name
        w = o.pages[0].obj["/Annots"][0]
        assert w.get("/StructParent") is not None, "widget missing /StructParent"
        assert str(w.get("/TU")) == "Full Name", f"expected /TU backfilled from /T, got {w.get('/TU')}"
        # the /Form's OBJR points at the widget
        assert any(t.objgen == w.objgen for t in objr_targets), "/Form OBJR does not target the widget"


def test_bare_url_text_autolinked(tmp_path):
    """Bare URL / email TEXT with no existing link annotation must be auto-detected
    and turned into a functional /Link annotation (/A /URI) which then gets a /Link
    struct element. Regression guard for the link auto-detection gap (incumbent-
    baseline coverage: mean per-doc link recall was 0.06 — we only tagged pre-existing
    annotations). Whole-token match must NOT link ordinary prose words."""
    pdf = pikepdf.new()
    page = pdf.add_blank_page(page_size=(612, 792))
    # standard Helvetica so pdfplumber can extract words from the content stream
    font = pdf.make_indirect(pikepdf.Dictionary({
        "/Type": pikepdf.Name.Font, "/Subtype": pikepdf.Name.Type1,
        "/BaseFont": pikepdf.Name.Helvetica}))
    page.obj["/Resources"] = pikepdf.Dictionary({"/Font": pikepdf.Dictionary({"/F1": font})})
    page.obj["/Contents"] = pdf.make_stream(
        b"BT /F1 12 Tf 72 700 Td (Email us at info@example.org or visit https://example.com today) Tj ET\n")
    input_path = tmp_path / "urls.pdf"
    pdf.save(str(input_path))
    pdf.close()

    els = [TaggedElement(element_id="p1_c0", page_num=1, pdf_tag=PDFTag.P,
                         text="Email us at info@example.org or visit https://example.com today",
                         bbox=(72, 80, 540, 100), merged_from=[f"p1_c{i}" for i in range(60)])]
    out_path = tmp_path / "urls_out.pdf"
    tag_untagged_pdf(str(input_path), str(out_path), els, total_pages=1)

    with pikepdf.open(str(out_path)) as o:
        link_uris = []
        for a in (o.pages[0].obj.get("/Annots", []) or []):
            if str(a.get("/Subtype")) == "/Link" and a.get("/A") is not None:
                u = a["/A"].get("/URI")
                if u is not None:
                    link_uris.append(str(u))
        s_tags = []
        def walk(n):
            if isinstance(n, pikepdf.Dictionary):
                if n.get("/S") is not None:
                    s_tags.append(str(n.get("/S")))
                if n.get("/K") is not None:
                    walk(n.get("/K"))
            elif isinstance(n, pikepdf.Array):
                for x in n:
                    walk(x)
        walk(o.Root.StructTreeRoot.K)

    assert "https://example.com" in link_uris, f"URL not auto-linked: {link_uris}"
    assert "mailto:info@example.org" in link_uris, f"email not auto-linked: {link_uris}"
    assert s_tags.count("/Link") >= 2, f"expected >=2 /Link struct elems, got {s_tags}"
    # ordinary words ("Email", "visit", "today") must NOT become links
    assert not any(u in ("Email", "visit", "today") for u in link_uris)


def test_signed_pdf_detection_and_skip(tmp_path):
    """Digitally signed PDFs must be detected and left UNMODIFIED — rewriting the byte
    stream invalidates the signature (legal/contractual integrity). Do-no-harm guard;
    49/774 docs in the baseline corpus carry a signature."""
    from tagger.stage10_writeback.struct_tree_writer import pdf_is_signed

    # unsigned plain PDF -> not detected
    plain = pikepdf.new(); plain.add_blank_page(page_size=(612, 792))
    p_plain = tmp_path / "plain.pdf"; plain.save(str(p_plain)); plain.close()
    assert pdf_is_signed(str(p_plain)) is False

    # signed PDF: AcroForm /SigFlags bit-1 + a /Sig field carrying a /V value
    signed = pikepdf.new(); signed.add_blank_page(page_size=(612, 792))
    sigfield = signed.make_indirect(pikepdf.Dictionary({
        "/FT": pikepdf.Name("/Sig"), "/T": pikepdf.String("Signature1"),
        "/V": signed.make_indirect(pikepdf.Dictionary({"/Type": pikepdf.Name("/Sig")}))}))
    signed.Root["/AcroForm"] = signed.make_indirect(pikepdf.Dictionary({
        "/SigFlags": 3, "/Fields": pikepdf.Array([sigfield])}))
    p_signed = tmp_path / "signed.pdf"; signed.save(str(p_signed)); signed.close()
    assert pdf_is_signed(str(p_signed)) is True

    # end-to-end: the pipeline must NOT add a struct tree to the signed doc
    out = tmp_path / "signed_out.pdf"
    AutoTaggerPipeline().run(input_pdf=str(p_signed), output_pdf=str(out))
    with pikepdf.open(str(out)) as o:
        assert "/StructTreeRoot" not in o.Root, "signed PDF was modified (struct tree added)"
