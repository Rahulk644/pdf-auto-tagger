"""Tests for Stage 7 — Cross-page merger."""

import pytest
from tagger.models.data_types import PDFTag, TaggedElement
from tagger.stage7_cross_page.cross_page_merger import (
    merge_cross_page,
    _is_split_paragraph,
    _is_list_continuation,
)


def _make_el(text: str, tag: PDFTag, page: int, y: float = 0,
             eid: str | None = None, font_size: float = 11.0) -> TaggedElement:
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


class TestSplitParagraph:
    """Tests for split paragraph detection."""

    def test_continuation_detected(self):
        """Paragraph ending without punctuation + next starts lowercase → split."""
        last = _make_el("The quick brown fox jumped over the", PDFTag.P, page=1)
        first = _make_el("lazy dog and went to sleep.", PDFTag.P, page=2)
        assert _is_split_paragraph(last, first) is True

    def test_normal_break_not_detected(self):
        """Paragraph ending with period → not a split."""
        last = _make_el("End of paragraph.", PDFTag.P, page=1)
        first = _make_el("Start of new paragraph.", PDFTag.P, page=2)
        assert _is_split_paragraph(last, first) is False

    def test_different_tags_not_detected(self):
        """H2 → P is not a split paragraph."""
        last = _make_el("Heading", PDFTag.H2, page=1)
        first = _make_el("paragraph text", PDFTag.P, page=2)
        assert _is_split_paragraph(last, first) is False

    def test_different_font_size_not_detected(self):
        """Different font sizes → not a split."""
        last = _make_el("some text without ending", PDFTag.P, page=1, font_size=11)
        first = _make_el("different size text", PDFTag.P, page=2, font_size=18)
        assert _is_split_paragraph(last, first) is False


class TestListContinuation:
    """Tests for list continuation detection."""

    def test_matching_list_items(self):
        """Same X position, both LI → continuation."""
        last = _make_el("• Item one", PDFTag.LI, page=1)
        first = _make_el("• Item two", PDFTag.LI, page=2)
        assert _is_list_continuation(last, first) is True

    def test_non_list_not_detected(self):
        """Non-LI tags → not a list continuation."""
        last = _make_el("Just text", PDFTag.P, page=1)
        first = _make_el("More text", PDFTag.P, page=2)
        assert _is_list_continuation(last, first) is False


class TestMergeCrossPage:
    """Integration tests for cross-page merging."""

    def test_no_continuations_in_simple_doc(self):
        """Simple doc with clean breaks → no continuations."""
        elements = [
            _make_el("Chapter 1: Introduction", PDFTag.H2, page=1, y=100, eid="e1"),
            _make_el("This is paragraph one.", PDFTag.P, page=1, y=200, eid="e2"),
            _make_el("1", PDFTag.ARTIFACT, page=1, y=750, eid="e3"),
            _make_el("Chapter 2: Methods", PDFTag.H2, page=2, y=100, eid="e4"),
            _make_el("This is paragraph two.", PDFTag.P, page=2, y=200, eid="e5"),
            _make_el("2", PDFTag.ARTIFACT, page=2, y=750, eid="e6"),
        ]
        result = merge_cross_page(elements, total_pages=2)
        cross_page_count = sum(1 for el in result if el.cross_page)
        assert cross_page_count == 0

    def test_split_paragraph_detected(self):
        """Paragraph spanning pages → cross_page flag set."""
        elements = [
            _make_el("The process involves multiple", PDFTag.P, page=1, y=700, eid="e1"),
            _make_el("steps that must be followed carefully.", PDFTag.P, page=2, y=50, eid="e2"),
        ]
        result = merge_cross_page(elements, total_pages=2)
        # First element of next page should be flagged
        next_page_el = [el for el in result if el.page_num == 2][0]
        assert next_page_el.cross_page is True
