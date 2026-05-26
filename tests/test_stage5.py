"""Tests for Stage 5 — Text handler."""

import pytest
from tagger.models.data_types import (
    LayoutCategory,
    LayoutRegion,
    PageElement,
    PDFTag,
)
from tagger.stage5_specialists.text_handler import handle_text_regions


def _make_region(category: LayoutCategory, idx: int = 0,
                 matched: list[str] | None = None) -> LayoutRegion:
    return LayoutRegion(
        region_id=f"r1_{idx}",
        page_num=1,
        bbox=(0, idx * 20, 100, (idx + 1) * 20),
        category=category,
        reading_order=idx,
        confidence=0.8,
        matched_elements=matched or [],
    )


def _make_element(text: str, idx: int = 0) -> PageElement:
    return PageElement(
        element_id=f"p1_e{idx}",
        page_num=1,
        text=text,
        bbox=(0, idx * 20, 100, (idx + 1) * 20),
        font_size=11.0,
    )


class TestTextHandler:
    """Tests for text region handling."""

    def test_title_becomes_h1(self):
        """TITLE layout → H1 tag."""
        el = _make_element("Document Title", idx=0)
        region = _make_region(LayoutCategory.TITLE, idx=0, matched=["p1_e0"])
        result = handle_text_regions([region], [el], page_num=1)
        assert len(result) == 1
        assert result[0].pdf_tag == PDFTag.H1

    def test_section_header_becomes_h2(self):
        """SECTION_HEADER layout → H2 tag."""
        el = _make_element("Introduction", idx=0)
        region = _make_region(LayoutCategory.SECTION_HEADER, idx=0, matched=["p1_e0"])
        result = handle_text_regions([region], [el], page_num=1)
        assert result[0].pdf_tag == PDFTag.H2

    def test_bullet_list_detection(self):
        """Paragraph starting with bullet → LI."""
        el = _make_element("• First item in list", idx=0)
        region = _make_region(LayoutCategory.TEXT, idx=0, matched=["p1_e0"])
        result = handle_text_regions([region], [el], page_num=1)
        assert result[0].pdf_tag == PDFTag.LI

    def test_numbered_list_detection(self):
        """Paragraph starting with number + period → LI."""
        el = _make_element("1. First numbered item", idx=0)
        region = _make_region(LayoutCategory.TEXT, idx=0, matched=["p1_e0"])
        result = handle_text_regions([region], [el], page_num=1)
        assert result[0].pdf_tag == PDFTag.LI

    def test_footnote_becomes_note(self):
        """FOOTNOTE layout → Note tag."""
        el = _make_element("This is a footnote reference.", idx=0)
        region = _make_region(LayoutCategory.FOOTNOTE, idx=0, matched=["p1_e0"])
        result = handle_text_regions([region], [el], page_num=1)
        assert result[0].pdf_tag == PDFTag.NOTE

    def test_regular_text_stays_p(self):
        """Regular TEXT layout → P tag."""
        el = _make_element("This is regular paragraph text.", idx=0)
        region = _make_region(LayoutCategory.TEXT, idx=0, matched=["p1_e0"])
        result = handle_text_regions([region], [el], page_num=1)
        assert result[0].pdf_tag == PDFTag.P
