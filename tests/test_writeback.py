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
