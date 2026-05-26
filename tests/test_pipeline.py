"""Tests for the full pipeline integration."""

import json
import pytest
from pathlib import Path

from tagger.pipeline import AutoTaggerPipeline
from tagger.models.data_types import PDFTag


class TestPipelineIntegration:
    """Full pipeline integration tests."""

    def test_full_pipeline_runs(self, tmp_path):
        """Pipeline should complete without errors on sample PDF."""
        report_path = tmp_path / "report.json"

        pipeline = AutoTaggerPipeline()
        report = pipeline.run(
            input_pdf="tests/fixtures/sample.pdf",
            report_path=str(report_path),
        )

        assert report["total_pages"] == 3
        assert report["summary"]["total_elements"] > 0
        assert report["summary"]["total_time_seconds"] > 0

        # Report file should be written
        assert report_path.exists()

        # Confidence report should also exist
        conf_path = report_path.with_suffix(".confidence.json")
        assert conf_path.exists()

    def test_tag_distribution_makes_sense(self):
        """Tags should include headings, paragraphs, and artifacts."""
        pipeline = AutoTaggerPipeline()
        report = pipeline.run(input_pdf="tests/fixtures/sample.pdf")

        tags = report["tag_distribution"]
        assert "H1" in tags, "Should detect at least one title (H1)"
        assert "H2" in tags, "Should detect section headings (H2)"
        assert "P" in tags, "Should detect paragraphs"
        assert "Artifact" in tags, "Should detect page numbers as artifacts"

    def test_heading_hierarchy(self):
        """H1 should have larger font than H2."""
        pipeline = AutoTaggerPipeline()
        report = pipeline.run(input_pdf="tests/fixtures/sample.pdf")

        h1_elements = [el for el in report["elements"] if el["pdf_tag"] == "H1"]
        h2_elements = [el for el in report["elements"] if el["pdf_tag"] == "H2"]

        assert len(h1_elements) > 0
        assert len(h2_elements) > 0

        # H1 font size should be larger than H2
        h1_size = h1_elements[0]["font_size"]
        h2_size = h2_elements[0]["font_size"]
        assert h1_size > h2_size

    def test_page_numbers_are_artifacts(self):
        """Page numbers (1, 2, 3) should be tagged as Artifact."""
        pipeline = AutoTaggerPipeline()
        report = pipeline.run(input_pdf="tests/fixtures/sample.pdf")

        artifacts = [el for el in report["elements"] if el["pdf_tag"] == "Artifact"]
        artifact_texts = {el["text"].strip() for el in artifacts}

        # Our sample has page numbers 1, 2, 3
        assert "1" in artifact_texts
        assert "2" in artifact_texts
        assert "3" in artifact_texts

    def test_pipeline_timing(self):
        """All stages should have timing data."""
        pipeline = AutoTaggerPipeline()
        report = pipeline.run(input_pdf="tests/fixtures/sample.pdf")

        timings = report["stage_timings"]
        assert "stage0" in timings
        assert "stage1" in timings
        assert "stage2" in timings

        # Total time should be sum of stages (approximately)
        assert report["summary"]["total_time_seconds"] > 0
