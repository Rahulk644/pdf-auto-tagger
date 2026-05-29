"""Tests for Stage 10 — Struct tree writeback."""

import json
import pytest
from pathlib import Path

import pikepdf

from tagger.pipeline import AutoTaggerPipeline
from tagger.stage10_writeback.struct_tree_writer import tag_untagged_pdf
from tagger.models.data_types import PDFTag, TaggedElement


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
