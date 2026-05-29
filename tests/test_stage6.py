"""Tests for Stage 6 — Consistency validator."""

import pytest
from tagger.config import VALIDATOR
from tagger.models.confidence import ConfidenceTracker
from tagger.models.data_types import (
    PDFTag,
    TaggedElement,
    TableStructure,
    FormulaResult,
    FigureInfo,
)
from tagger.stage6_validator.consistency_validator import (
    validate_elements,
    ValidationContext,
    SingleCellTableRule,
    EmptyTableRule,
    InvalidLatexRule,
    ZeroCharElementRule,
    TinyFigureRule,
    OverlappingRegionRule,
)


def _make_el(
    tag: PDFTag,
    text: str = "Some text",
    bbox: tuple = (0, 0, 100, 50),
    eid: str = "e1",
    page: int = 1,
    confidence: float = 0.9,
) -> TaggedElement:
    """Helper to create a TaggedElement."""
    return TaggedElement(
        element_id=eid,
        page_num=page,
        pdf_tag=tag,
        text=text,
        bbox=bbox,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# SingleCellTableRule
# ---------------------------------------------------------------------------

class TestSingleCellTableRule:
    """Tests for the 1×1 table → paragraph reclassification rule."""

    def test_single_cell_reclassified(self):
        """A 1×1 table should be reclassified to P."""
        el = _make_el(PDFTag.TABLE, eid="t1")
        table = TableStructure(
            region_id="t1", html="<table><tr><td>one</td></tr></table>",
            num_rows=1, num_cols=1, has_header=False, confidence=0.5,
        )
        ctx = ValidationContext(
            all_elements=[el],
            table_structures={"t1": table},
        )

        rule = SingleCellTableRule()
        result = rule.check(el, ctx)

        assert result is not None
        assert result.action == "reclassify"
        assert result.new_tag == PDFTag.P

    def test_normal_table_passes(self):
        """A 3×4 table should not be flagged."""
        el = _make_el(PDFTag.TABLE, eid="t2")
        table = TableStructure(
            region_id="t2", html="<table>...</table>",
            num_rows=3, num_cols=4, has_header=True, confidence=0.9,
        )
        ctx = ValidationContext(
            all_elements=[el],
            table_structures={"t2": table},
        )

        rule = SingleCellTableRule()
        result = rule.check(el, ctx)
        assert result is None

    def test_non_table_skipped(self):
        """Non-table elements should be skipped."""
        el = _make_el(PDFTag.P, eid="p1")
        ctx = ValidationContext(all_elements=[el])

        rule = SingleCellTableRule()
        result = rule.check(el, ctx)
        assert result is None


# ---------------------------------------------------------------------------
# EmptyTableRule
# ---------------------------------------------------------------------------

class TestEmptyTableRule:
    """Tests for tables with 0 extracted cells."""

    def test_empty_table_flagged(self):
        """A 0×0 table should be flagged for review."""
        el = _make_el(PDFTag.TABLE, eid="t1")
        table = TableStructure(
            region_id="t1", html="<table></table>",
            num_rows=0, num_cols=0, has_header=False, confidence=0.3,
        )
        ctx = ValidationContext(
            all_elements=[el],
            table_structures={"t1": table},
        )

        rule = EmptyTableRule()
        result = rule.check(el, ctx)

        assert result is not None
        assert result.action == "flag"

    def test_table_with_cells_passes(self):
        """A table with cells should pass."""
        el = _make_el(PDFTag.TABLE, eid="t2")
        table = TableStructure(
            region_id="t2", html="<table><tr><td>a</td></tr></table>",
            num_rows=1, num_cols=1, has_header=False, confidence=0.8,
        )
        ctx = ValidationContext(
            all_elements=[el],
            table_structures={"t2": table},
        )

        rule = EmptyTableRule()
        result = rule.check(el, ctx)
        assert result is None


# ---------------------------------------------------------------------------
# InvalidLatexRule
# ---------------------------------------------------------------------------

class TestInvalidLatexRule:
    """Tests for broken formula LaTeX."""

    def test_empty_latex_flagged(self):
        """Empty LaTeX should be flagged."""
        el = _make_el(PDFTag.FORMULA, eid="f1")
        formula = FormulaResult(
            region_id="f1", latex="", is_inline=False, confidence=0.5,
        )
        ctx = ValidationContext(
            all_elements=[el],
            formula_results={"f1": formula},
        )

        rule = InvalidLatexRule()
        result = rule.check(el, ctx)

        assert result is not None
        assert result.action == "flag"

    def test_unmatched_braces_flagged(self):
        """LaTeX with unmatched braces should be flagged."""
        el = _make_el(PDFTag.FORMULA, eid="f2")
        formula = FormulaResult(
            region_id="f2", latex=r"\frac{a}{b", is_inline=False, confidence=0.7,
        )
        ctx = ValidationContext(
            all_elements=[el],
            formula_results={"f2": formula},
        )

        rule = InvalidLatexRule()
        result = rule.check(el, ctx)

        assert result is not None
        assert result.action == "flag"
        assert "unmatched braces" in result.reason.lower()

    def test_valid_latex_passes(self):
        """Valid LaTeX should pass."""
        el = _make_el(PDFTag.FORMULA, eid="f3")
        formula = FormulaResult(
            region_id="f3", latex=r"\frac{a}{b}", is_inline=False, confidence=0.9,
        )
        ctx = ValidationContext(
            all_elements=[el],
            formula_results={"f3": formula},
        )

        rule = InvalidLatexRule()
        result = rule.check(el, ctx)
        assert result is None

    def test_non_formula_skipped(self):
        """Non-formula elements should be skipped."""
        el = _make_el(PDFTag.P)
        ctx = ValidationContext(all_elements=[el])

        rule = InvalidLatexRule()
        result = rule.check(el, ctx)
        assert result is None


# ---------------------------------------------------------------------------
# ZeroCharElementRule
# ---------------------------------------------------------------------------

class TestZeroCharElementRule:
    """Tests for elements with no text content."""

    def test_empty_text_flagged(self):
        """Element with empty text should be reclassified to Artifact (the rule
        evolved from 'flag' to 'reclassify->Artifact' so the empty element no
        longer remains in the struct tree as a content-less /P — see Stage 6
        ZeroCharElementRule)."""
        el = _make_el(PDFTag.P, text="")
        ctx = ValidationContext(all_elements=[el])

        rule = ZeroCharElementRule()
        result = rule.check(el, ctx)

        assert result is not None
        assert result.action == "reclassify"
        assert result.new_tag == PDFTag.ARTIFACT

    def test_whitespace_only_flagged(self):
        """Element with only whitespace should be flagged."""
        el = _make_el(PDFTag.H1, text="   \n\t  ")
        ctx = ValidationContext(all_elements=[el])

        rule = ZeroCharElementRule()
        result = rule.check(el, ctx)

        assert result is not None

    def test_figure_with_no_text_passes(self):
        """Figures don't need text content."""
        el = _make_el(PDFTag.FIGURE, text="")
        ctx = ValidationContext(all_elements=[el])

        rule = ZeroCharElementRule()
        result = rule.check(el, ctx)
        assert result is None

    def test_artifact_with_no_text_passes(self):
        """Artifacts don't need text content."""
        el = _make_el(PDFTag.ARTIFACT, text="")
        ctx = ValidationContext(all_elements=[el])

        rule = ZeroCharElementRule()
        result = rule.check(el, ctx)
        assert result is None

    def test_normal_text_passes(self):
        """Element with text should pass."""
        el = _make_el(PDFTag.P, text="Hello world")
        ctx = ValidationContext(all_elements=[el])

        rule = ZeroCharElementRule()
        result = rule.check(el, ctx)
        assert result is None


# ---------------------------------------------------------------------------
# TinyFigureRule
# ---------------------------------------------------------------------------

class TestTinyFigureRule:
    """Tests for undersized figures → Artifact."""

    def test_tiny_figure_reclassified(self):
        """A 5×5 figure should become an Artifact."""
        el = _make_el(PDFTag.FIGURE, bbox=(0, 0, 5, 5))
        ctx = ValidationContext(all_elements=[el])

        rule = TinyFigureRule()
        result = rule.check(el, ctx)

        assert result is not None
        assert result.action == "reclassify"
        assert result.new_tag == PDFTag.ARTIFACT

    def test_normal_figure_passes(self):
        """A 200×300 figure should pass."""
        el = _make_el(PDFTag.FIGURE, bbox=(0, 0, 200, 300))
        ctx = ValidationContext(all_elements=[el])

        rule = TinyFigureRule()
        result = rule.check(el, ctx)
        assert result is None

    def test_non_figure_skipped(self):
        """Non-figure elements should be skipped."""
        el = _make_el(PDFTag.P, bbox=(0, 0, 5, 5))
        ctx = ValidationContext(all_elements=[el])

        rule = TinyFigureRule()
        result = rule.check(el, ctx)
        assert result is None


# ---------------------------------------------------------------------------
# OverlappingRegionRule
# ---------------------------------------------------------------------------

class TestOverlappingRegionRule:
    """Tests for duplicate/overlapping detections."""

    def test_identical_overlap_flagged(self):
        """Two elements at same position should be flagged."""
        el1 = _make_el(PDFTag.P, bbox=(10, 10, 100, 50), eid="e1")
        el2 = _make_el(PDFTag.H2, bbox=(10, 10, 100, 50), eid="e2")
        ctx = ValidationContext(all_elements=[el1, el2])

        rule = OverlappingRegionRule()
        result = rule.check(el1, ctx)

        assert result is not None
        assert result.action == "flag"
        assert "e2" in result.reason

    def test_no_overlap_passes(self):
        """Non-overlapping elements should pass."""
        el1 = _make_el(PDFTag.P, bbox=(0, 0, 50, 50), eid="e1")
        el2 = _make_el(PDFTag.P, bbox=(200, 200, 300, 300), eid="e2")
        ctx = ValidationContext(all_elements=[el1, el2])

        rule = OverlappingRegionRule()
        result = rule.check(el1, ctx)
        assert result is None

    def test_cross_page_overlap_ignored(self):
        """Overlap on different pages should be ignored."""
        el1 = _make_el(PDFTag.P, bbox=(10, 10, 100, 50), eid="e1", page=1)
        el2 = _make_el(PDFTag.P, bbox=(10, 10, 100, 50), eid="e2", page=2)
        ctx = ValidationContext(all_elements=[el1, el2])

        rule = OverlappingRegionRule()
        result = rule.check(el1, ctx)
        assert result is None


# ---------------------------------------------------------------------------
# Integration: validate_elements
# ---------------------------------------------------------------------------

class TestValidateElements:
    """Integration tests for the full validation pipeline."""

    def test_reclassify_updates_tag(self):
        """Validation should modify element tags in-place."""
        el = _make_el(PDFTag.FIGURE, bbox=(0, 0, 5, 5), eid="f1")
        tracker = ConfidenceTracker()
        ctx = ValidationContext(all_elements=[el])

        validate_elements([el], tracker, ctx)

        assert el.pdf_tag == PDFTag.ARTIFACT

    def test_flag_sets_review(self):
        """Flagged elements should have needs_review set."""
        el = _make_el(PDFTag.P, text="", eid="e1")
        tracker = ConfidenceTracker()
        ctx = ValidationContext(all_elements=[el])

        validate_elements([el], tracker, ctx)

        assert el.needs_review is True
        assert el.review_reason is not None

    def test_valid_elements_unchanged(self):
        """Valid elements should not be modified."""
        el = _make_el(PDFTag.P, text="This is fine.", eid="e1")
        tracker = ConfidenceTracker()
        ctx = ValidationContext(all_elements=[el])

        original_tag = el.pdf_tag
        original_conf = el.confidence

        validate_elements([el], tracker, ctx)

        assert el.pdf_tag == original_tag
        assert el.confidence == original_conf
        assert el.needs_review is False

    def test_first_matching_rule_only(self):
        """Only the first matching rule should be applied."""
        # A tiny figure (5×5) with no text — TinyFigureRule fires first
        # because ZeroCharElementRule skips Figures
        el = _make_el(PDFTag.FIGURE, text="", bbox=(0, 0, 5, 5), eid="f1")
        tracker = ConfidenceTracker()
        ctx = ValidationContext(all_elements=[el])

        validate_elements([el], tracker, ctx)

        # Should be reclassified to Artifact (TinyFigureRule)
        assert el.pdf_tag == PDFTag.ARTIFACT
