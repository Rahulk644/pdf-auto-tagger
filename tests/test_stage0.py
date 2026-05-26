"""Tests for Stage 0 — Page classifier."""

import pytest
from tagger.models.data_types import PageType
from tagger.stage0_classifier.page_classifier import (
    _compute_unicode_validity,
    _compute_image_coverage,
    _decide_page_type,
    classify_pages,
)


class TestDecidePageType:
    """Tests for the classification decision tree."""

    def test_pure_native(self):
        """Many chars, no images → NATIVE."""
        pt, conf = _decide_page_type(
            char_count=500,
            unicode_validity=1.0,
            image_coverage=0.0,
            char_density=5.0,
        )
        assert pt == PageType.NATIVE
        assert conf >= 0.90

    def test_pure_scanned(self):
        """Zero chars, full image → SCANNED."""
        pt, conf = _decide_page_type(
            char_count=0,
            unicode_validity=1.0,
            image_coverage=0.95,
            char_density=0.0,
        )
        assert pt == PageType.SCANNED
        assert conf >= 0.90

    def test_mixed_page(self):
        """Many chars + large image → MIXED or NATIVE."""
        pt, conf = _decide_page_type(
            char_count=200,
            unicode_validity=1.0,
            image_coverage=0.75,
            char_density=2.0,
        )
        assert pt in (PageType.MIXED, PageType.NATIVE)

    def test_corrupt_bad_unicode(self):
        """Chars present but Unicode validity very low → CORRUPT."""
        pt, conf = _decide_page_type(
            char_count=100,
            unicode_validity=0.50,
            image_coverage=0.0,
            char_density=1.0,
        )
        assert pt == PageType.CORRUPT

    def test_blank_page(self):
        """No chars, no images → CORRUPT (blank)."""
        pt, conf = _decide_page_type(
            char_count=0,
            unicode_validity=1.0,
            image_coverage=0.0,
            char_density=0.0,
        )
        assert pt == PageType.CORRUPT

    def test_few_chars_high_image(self):
        """Few chars + high image → MIXED."""
        pt, conf = _decide_page_type(
            char_count=10,
            unicode_validity=1.0,
            image_coverage=0.85,
            char_density=0.1,
        )
        assert pt == PageType.MIXED


class TestUnicodeValidity:
    """Tests for Unicode validity checking."""

    def test_all_valid(self):
        """Normal ASCII chars → 1.0."""
        chars = [{"text": c} for c in "Hello World"]
        assert _compute_unicode_validity(chars) == 1.0

    def test_empty_chars(self):
        """No chars → 1.0 (vacuously valid)."""
        assert _compute_unicode_validity([]) == 1.0

    def test_empty_text(self):
        """Chars with empty text → 1.0."""
        assert _compute_unicode_validity([{"text": ""}]) == 1.0


class TestClassifyPages:
    """Integration tests using a real PDF."""

    def test_sample_pdf_all_native(self):
        """Our test PDF should classify all pages as native."""
        results = classify_pages("tests/fixtures/sample.pdf")
        assert len(results) == 3
        for r in results:
            assert r.page_type == PageType.NATIVE
            assert r.confidence >= 0.90
            assert r.char_count > 0
            assert r.image_coverage == 0.0
            assert r.unicode_validity == 1.0
