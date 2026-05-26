"""Tests for coordinate transformer."""

import pytest
from tagger.stage1_extraction.coord_transformer import (
    pdf_to_standard,
    standard_to_pdf,
    pdfplumber_to_standard,
    compute_iou,
    bbox_contains,
)
from tagger.config import STANDARD_DPI, PDF_NATIVE_DPI


class TestPdfToStandard:
    """Tests for PDF-native → standard coordinate transform."""

    def test_origin_transform(self):
        """Bottom-left origin should flip to top-left origin."""
        page_height = 792  # Letter size
        bbox = (0, 0, 72, 72)  # 1-inch box at bottom-left

        result = pdf_to_standard(bbox, page_height)

        # At 150 DPI, 72pt → 150px
        scale = STANDARD_DPI / PDF_NATIVE_DPI  # 150/72 ≈ 2.083

        # x0 should stay at 0
        assert result[0] == pytest.approx(0.0)
        # y0 should be near the bottom of the page in standard coords
        assert result[1] == pytest.approx((792 - 72) * scale)
        # x1 should be 72 * scale
        assert result[2] == pytest.approx(72 * scale)
        # y1 should be at the very bottom
        assert result[3] == pytest.approx(792 * scale)

    def test_roundtrip(self):
        """pdf_to_standard → standard_to_pdf should return original."""
        page_height = 792
        original = (100, 200, 300, 400)

        standard = pdf_to_standard(original, page_height)
        restored = standard_to_pdf(standard, page_height)

        assert restored[0] == pytest.approx(original[0], abs=0.1)
        assert restored[1] == pytest.approx(original[1], abs=0.1)
        assert restored[2] == pytest.approx(original[2], abs=0.1)
        assert restored[3] == pytest.approx(original[3], abs=0.1)


class TestPdfplumberToStandard:
    """Tests for pdfplumber → standard coordinate transform."""

    def test_scale_only(self):
        """pdfplumber coords should only be scaled, not Y-flipped."""
        bbox = (72, 72, 144, 144)  # 1-inch box
        page_height = 792

        result = pdfplumber_to_standard(bbox, page_height)
        scale = STANDARD_DPI / PDF_NATIVE_DPI

        assert result[0] == pytest.approx(72 * scale)
        assert result[1] == pytest.approx(72 * scale)
        assert result[2] == pytest.approx(144 * scale)
        assert result[3] == pytest.approx(144 * scale)


class TestComputeIoU:
    """Tests for IoU calculation."""

    def test_identical_boxes(self):
        """Same box → IoU = 1.0."""
        assert compute_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)

    def test_no_overlap(self):
        """Disjoint boxes → IoU = 0.0."""
        assert compute_iou((0, 0, 5, 5), (10, 10, 15, 15)) == 0.0

    def test_partial_overlap(self):
        """Overlapping boxes → 0 < IoU < 1."""
        iou = compute_iou((0, 0, 10, 10), (5, 5, 15, 15))
        assert 0 < iou < 1

    def test_contained(self):
        """One box inside another → IoU = inner_area / outer_area."""
        iou = compute_iou((0, 0, 20, 20), (5, 5, 15, 15))
        # Inner area = 100, outer area = 400, union = 400
        assert iou == pytest.approx(100 / 400)


class TestBboxContains:
    """Tests for containment check."""

    def test_fully_contained(self):
        assert bbox_contains((0, 0, 100, 100), (10, 10, 90, 90))

    def test_not_contained(self):
        assert not bbox_contains((0, 0, 100, 100), (50, 50, 150, 150))

    def test_tolerance(self):
        """Slightly outside but within tolerance → contained."""
        assert bbox_contains((10, 10, 90, 90), (9, 9, 91, 91), tolerance=2.0)
