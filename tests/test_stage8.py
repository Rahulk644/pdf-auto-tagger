"""Tests for Stage 8 — Semantic refinement modules."""

import pytest
from tagger.models.data_types import PDFTag, TaggedElement
from tagger.stage8_semantic.heading_ranker import assign_heading_levels
from tagger.stage8_semantic.toc_detector import detect_toc_entries
from tagger.stage8_semantic.artifact_detector import detect_artifacts, _extract_page_number
from tagger.stage8_semantic.caption_detector import detect_captions


def _make_tagged(text: str, tag: PDFTag, page: int = 1,
                 font_size: float = 11.0, y: float = 0,
                 eid: str | None = None) -> TaggedElement:
    """Helper to create a TaggedElement."""
    return TaggedElement(
        element_id=eid or f"p{page}_e0",
        page_num=page,
        pdf_tag=tag,
        text=text,
        bbox=(0, y, 100, y + 15),
        font_size=font_size,
        confidence=0.8,
    )


class TestHeadingRanker:
    """Tests for heading level assignment."""

    def test_two_sizes_map_to_h1_h2(self):
        """Largest font → H1, second → H2."""
        elements = [
            _make_tagged("Title", PDFTag.H1, font_size=24.0, eid="e1"),
            _make_tagged("Section", PDFTag.H2, font_size=18.0, eid="e2"),
            _make_tagged("Body text", PDFTag.P, font_size=11.0, eid="e3"),
        ]
        result = assign_heading_levels(elements)
        headings = [el for el in result if el.pdf_tag in (PDFTag.H1, PDFTag.H2)]
        assert headings[0].pdf_tag == PDFTag.H1  # 24pt
        assert headings[1].pdf_tag == PDFTag.H2  # 18pt

    def test_three_sizes(self):
        """Three distinct sizes → H1, H2, H3."""
        elements = [
            _make_tagged("Title", PDFTag.H1, font_size=28.0, eid="e1"),
            _make_tagged("Chapter", PDFTag.H2, font_size=20.0, eid="e2"),
            _make_tagged("Section", PDFTag.H3, font_size=16.0, eid="e3"),
        ]
        result = assign_heading_levels(elements)
        tags = [el.pdf_tag for el in result]
        assert PDFTag.H1 in tags
        assert PDFTag.H2 in tags
        assert PDFTag.H3 in tags

    def test_no_headings_is_noop(self):
        """If no headings exist, elements are unchanged."""
        elements = [
            _make_tagged("Body", PDFTag.P, font_size=11.0, eid="e1"),
        ]
        result = assign_heading_levels(elements)
        assert len(result) == 1
        assert result[0].pdf_tag == PDFTag.P


class TestTocDetector:
    """Tests for TOC entry detection."""

    def test_dot_leader_pattern(self):
        """Text with dot leaders + page number → TOCI."""
        elements = [
            _make_tagged("Introduction .............. 5", PDFTag.P, page=1, eid="e1"),
            _make_tagged("Methods .................. 12", PDFTag.P, page=1, eid="e2"),
            _make_tagged("Results .................. 25", PDFTag.P, page=1, eid="e3"),
        ]
        result = detect_toc_entries(elements, total_pages=10)
        toc_count = sum(1 for el in result if el.pdf_tag == PDFTag.TOCI)
        assert toc_count == 3

    def test_no_toc_in_middle(self):
        """TOC-like patterns on page 5 of 10 → not detected."""
        elements = [
            _make_tagged("Something .............. 5", PDFTag.P, page=5, eid="e1"),
        ]
        result = detect_toc_entries(elements, total_pages=10)
        toc_count = sum(1 for el in result if el.pdf_tag == PDFTag.TOCI)
        assert toc_count == 0


class TestArtifactDetector:
    """Tests for artifact (header/footer/page number) detection."""

    def test_extract_page_number_pure_digit(self):
        assert _extract_page_number("5") == 5

    def test_extract_page_number_dashes(self):
        assert _extract_page_number("- 5 -") == 5

    def test_extract_page_number_page_prefix(self):
        assert _extract_page_number("Page 42") == 42

    def test_extract_page_number_not_a_number(self):
        assert _extract_page_number("Hello World") is None

    def test_repeated_text_across_pages(self):
        """Same text at same Y on 3+ pages → Artifact."""
        elements = [
            _make_tagged("Company Name", PDFTag.P, page=1, y=10, eid="e1"),
            _make_tagged("Company Name", PDFTag.P, page=2, y=10, eid="e2"),
            _make_tagged("Company Name", PDFTag.P, page=3, y=10, eid="e3"),
            _make_tagged("Regular content", PDFTag.P, page=1, y=200, eid="e4"),
        ]
        result = detect_artifacts(elements)
        artifacts = [el for el in result if el.pdf_tag == PDFTag.ARTIFACT]
        assert len(artifacts) == 3

    def test_sequential_page_numbers(self):
        """Sequential numbers at consistent Y → Artifact."""
        elements = [
            _make_tagged("1", PDFTag.P, page=1, y=750, eid="e1"),
            _make_tagged("2", PDFTag.P, page=2, y=750, eid="e2"),
            _make_tagged("3", PDFTag.P, page=3, y=750, eid="e3"),
            _make_tagged("4", PDFTag.P, page=4, y=750, eid="e4"),
        ]
        result = detect_artifacts(elements)
        artifacts = [el for el in result if el.pdf_tag == PDFTag.ARTIFACT]
        assert len(artifacts) >= 3


class TestMarginFurniture:
    """Single-page margin-band furniture detection (no cross-page repetition)."""

    PH = {1: 1000.0, 2: 1000.0}  # standard-DPI page heights

    def test_page_number_single_page(self):
        """A page number in the top margin is artifacted on its own page."""
        els = [_make_tagged("16", PDFTag.P, page=2, y=40, eid="e1")]  # frac ~0.05
        detect_artifacts(els, page_heights=self.PH)
        assert els[0].pdf_tag == PDFTag.ARTIFACT

    def test_roman_numeral_page_number(self):
        """Roman-numeral page numbers ('xi') the digit passes miss are caught."""
        els = [_make_tagged("xi", PDFTag.P, page=1, y=40, eid="e1")]
        detect_artifacts(els, page_heights=self.PH)
        assert els[0].pdf_tag == PDFTag.ARTIFACT

    def test_running_header_in_band(self):
        """A short running header high in the page becomes an Artifact."""
        els = [_make_tagged("Table of Contents", PDFTag.P, page=1, y=40, eid="e1")]
        detect_artifacts(els, page_heights=self.PH)
        assert els[0].pdf_tag == PDFTag.ARTIFACT

    def test_footer_in_bottom_band(self):
        """Furniture in the bottom margin band is caught too."""
        els = [_make_tagged("62", PDFTag.P, page=1, y=960, eid="e1")]  # frac ~0.97
        detect_artifacts(els, page_heights=self.PH)
        assert els[0].pdf_tag == PDFTag.ARTIFACT

    def test_real_heading_below_band_spared(self):
        """A real heading just below the top margin (frac ~0.11) is NOT touched."""
        els = [_make_tagged("Introduction", PDFTag.H1, page=1, y=105, eid="e1")]
        detect_artifacts(els, page_heights=self.PH)
        assert els[0].pdf_tag == PDFTag.H1

    def test_long_line_in_band_spared(self):
        """A long line that begins inside the band is real content, not furniture."""
        text = "this is a long opening body line that happens to start very high"
        els = [_make_tagged(text, PDFTag.P, page=1, y=40, eid="e1")]
        detect_artifacts(els, page_heights=self.PH)
        assert els[0].pdf_tag == PDFTag.P

    def test_figure_in_band_spared(self):
        """Non-text structural content in the band is never reclassified."""
        els = [_make_tagged("", PDFTag.FIGURE, page=1, y=40, eid="e1")]
        detect_artifacts(els, page_heights=self.PH)
        assert els[0].pdf_tag == PDFTag.FIGURE

    def test_skipped_without_page_heights(self):
        """Margin pass is inert when page heights are unavailable (no regression)."""
        els = [_make_tagged("16", PDFTag.P, page=2, y=40, eid="e1")]
        detect_artifacts(els)  # no page_heights
        assert els[0].pdf_tag == PDFTag.P


class TestCaptionDetector:
    """Tests for caption detection."""

    def test_figure_caption_after(self):
        """Text starting with 'Figure N' right after a Figure → Caption."""
        elements = [
            _make_tagged("", PDFTag.FIGURE, page=1, y=100, eid="e1"),
            _make_tagged("Figure 1: Comparison chart", PDFTag.P, page=1, y=120, eid="e2"),
        ]
        result = detect_captions(elements)
        assert result[1].pdf_tag == PDFTag.CAPTION

    def test_table_caption_before(self):
        """Text starting with 'Table N' right before a Table → Caption."""
        elements = [
            _make_tagged("Table 3: Summary of results", PDFTag.P, page=1, y=100, eid="e1"),
            _make_tagged("", PDFTag.TABLE, page=1, y=120, eid="e2"),
        ]
        result = detect_captions(elements)
        assert result[0].pdf_tag == PDFTag.CAPTION

    def test_non_caption_not_tagged(self):
        """Regular paragraph near a figure should NOT become caption."""
        elements = [
            _make_tagged("", PDFTag.FIGURE, page=1, y=100, eid="e1"),
            _make_tagged("This is just a regular paragraph of text.", PDFTag.P, page=1, y=120, eid="e2"),
        ]
        result = detect_captions(elements)
        assert result[1].pdf_tag == PDFTag.P  # Unchanged
