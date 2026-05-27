"""
Stage 6 — Consistency validator.

Pure Python rule engine that catches broken extractions before they
touch the PDF.  Each rule checks for a specific failure pattern and
either reclassifies the element or flags it for human review.

Elements failing validation get confidence < 0.6 → routed to
the human review queue, not written to the PDF.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod

from tagger.config import VALIDATOR
from tagger.models.confidence import ConfidenceTracker
from tagger.models.data_types import (
    LayoutCategory,
    LayoutRegion,
    TaggedElement,
    PDFTag,
    TableStructure,
    FormulaResult,
    FigureInfo,
)

logger = logging.getLogger(__name__)


class ValidationRule(ABC):
    """Base class for validation rules."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable rule name."""
        ...

    @abstractmethod
    def check(
        self,
        element: TaggedElement,
        context: ValidationContext,
    ) -> ValidationResult | None:
        """
        Check an element against this rule.

        Returns None if the rule doesn't apply, or a ValidationResult
        if a problem was found.
        """
        ...


class ValidationResult:
    """Result of a validation check."""

    def __init__(
        self,
        rule_name: str,
        action: str,
        reason: str,
        new_tag: PDFTag | None = None,
        new_confidence: float | None = None,
    ):
        self.rule_name = rule_name
        self.action = action  # "reclassify", "flag", "remove"
        self.reason = reason
        self.new_tag = new_tag
        self.new_confidence = new_confidence


class ValidationContext:
    """Context data available to validation rules."""

    def __init__(
        self,
        all_elements: list[TaggedElement],
        table_structures: dict[str, TableStructure] | None = None,
        formula_results: dict[str, FormulaResult] | None = None,
        figure_infos: dict[str, FigureInfo] | None = None,
    ):
        from collections import defaultdict
        self.all_elements = all_elements
        self.table_structures = table_structures or {}
        self.formula_results = formula_results or {}
        self.figure_infos = figure_infos or {}
        
        self.elements_by_page = defaultdict(list)
        for e in all_elements:
            self.elements_by_page[e.page_num].append(e)


# ---------------------------------------------------------------------------
# Concrete validation rules
# ---------------------------------------------------------------------------

class SingleCellTableRule(ValidationRule):
    """Table with only 1 cell → likely a misclassified paragraph."""

    @property
    def name(self) -> str:
        return "single_cell_table"

    def check(self, element: TaggedElement, context: ValidationContext) -> ValidationResult | None:
        if element.pdf_tag != PDFTag.TABLE:
            return None

        table = context.table_structures.get(element.element_id)
        if table is None:
            return None

        if table.num_rows * table.num_cols <= 1:
            return ValidationResult(
                rule_name=self.name,
                action="reclassify",
                reason=f"Table has only {table.num_rows}×{table.num_cols} cells — likely a paragraph",
                new_tag=PDFTag.P,
                new_confidence=VALIDATOR.failed_confidence_cap,
            )
        return None


class EmptyTableRule(ValidationRule):
    """Table with 0 extracted cells → extraction failed."""

    @property
    def name(self) -> str:
        return "empty_table"

    def check(self, element: TaggedElement, context: ValidationContext) -> ValidationResult | None:
        if element.pdf_tag != PDFTag.TABLE:
            return None

        table = context.table_structures.get(element.element_id)
        if table is not None and table.num_rows == 0 and table.num_cols == 0:
            return ValidationResult(
                rule_name=self.name,
                action="flag",
                reason="Table extraction returned 0 cells",
                new_confidence=VALIDATOR.failed_confidence_cap,
            )
        return None


class InvalidLatexRule(ValidationRule):
    """Empty or obviously broken LaTeX → formula extraction failed."""

    @property
    def name(self) -> str:
        return "invalid_latex"

    def check(self, element: TaggedElement, context: ValidationContext) -> ValidationResult | None:
        if element.pdf_tag != PDFTag.FORMULA:
            return None

        formula = context.formula_results.get(element.element_id)
        if formula is None:
            return None

        latex = formula.latex.strip()
        if not latex:
            return ValidationResult(
                rule_name=self.name,
                action="flag",
                reason="Formula extraction returned empty LaTeX",
                new_confidence=VALIDATOR.failed_confidence_cap,
            )

        # Check for obviously broken LaTeX (unmatched braces, etc.)
        if latex.count("{") != latex.count("}"):
            return ValidationResult(
                rule_name=self.name,
                action="flag",
                reason=f"LaTeX has unmatched braces: {latex[:50]}...",
                new_confidence=VALIDATOR.failed_confidence_cap,
            )

        return None


class ZeroCharElementRule(ValidationRule):
    """Element with 0 text characters → extraction failure."""

    @property
    def name(self) -> str:
        return "zero_char_element"

    def check(self, element: TaggedElement, context: ValidationContext) -> ValidationResult | None:
        # Figures don't need text
        if element.pdf_tag in (PDFTag.FIGURE, PDFTag.ARTIFACT):
            return None

        if not element.text or not element.text.strip():
            return ValidationResult(
                rule_name=self.name,
                action="reclassify",
                reason="Element has no text content — suppressing as artifact",
                new_tag=PDFTag.ARTIFACT,
                new_confidence=0.0,
            )
        return None


class TinyFigureRule(ValidationRule):
    """Figure smaller than 20×20px → likely decorative noise → Artifact."""

    @property
    def name(self) -> str:
        return "tiny_figure"

    def check(self, element: TaggedElement, context: ValidationContext) -> ValidationResult | None:
        if element.pdf_tag != PDFTag.FIGURE:
            return None

        width = element.bbox[2] - element.bbox[0]
        height = element.bbox[3] - element.bbox[1]

        if (
            width < VALIDATOR.min_figure_width_px
            or height < VALIDATOR.min_figure_height_px
        ):
            return ValidationResult(
                rule_name=self.name,
                action="reclassify",
                reason=f"Figure too small ({width:.0f}×{height:.0f}px) — likely decorative",
                new_tag=PDFTag.ARTIFACT,
                new_confidence=0.8,
            )
        return None


class OverlappingRegionRule(ValidationRule):
    """Two elements with >80% IoU → likely duplicate detection."""

    @property
    def name(self) -> str:
        return "overlapping_region"

    def check(self, element: TaggedElement, context: ValidationContext) -> ValidationResult | None:
        from tagger.stage1_extraction.coord_transformer import compute_iou

        for other in context.elements_by_page[element.page_num]:
            if other.element_id <= element.element_id:
                continue

            iou = compute_iou(element.bbox, other.bbox)
            if iou >= VALIDATOR.overlap_iou_threshold:
                if other.confidence < element.confidence:
                    continue  # Let the lower-confidence element get flagged when it's evaluated
                
                return ValidationResult(
                    rule_name=self.name,
                    action="flag",
                    reason=f"Overlaps with {other.element_id} (IoU={iou:.2f})",
                    new_confidence=VALIDATOR.failed_confidence_cap,
                )
        return None


class StandaloneCurrencyRule(ValidationRule):
    """Standalone currency symbols (like '$') should be suppressed as Artifacts."""

    @property
    def name(self) -> str:
        return "standalone_currency"

    def check(self, element: TaggedElement, context: ValidationContext) -> ValidationResult | None:
        if element.pdf_tag in (PDFTag.FIGURE, PDFTag.ARTIFACT, PDFTag.TABLE):
            return None

        if element.text and element.text.strip() == "$":
            return ValidationResult(
                rule_name=self.name,
                action="reclassify",
                reason="Standalone currency symbol — suppressing as artifact",
                new_tag=PDFTag.ARTIFACT,
                new_confidence=0.8,
            )
        return None


# ---------------------------------------------------------------------------
# Validator engine
# ---------------------------------------------------------------------------

# Default rule set
DEFAULT_RULES: list[ValidationRule] = [
    SingleCellTableRule(),
    EmptyTableRule(),
    InvalidLatexRule(),
    ZeroCharElementRule(),
    StandaloneCurrencyRule(),
    TinyFigureRule(),
    OverlappingRegionRule(),
]


def validate_elements(
    elements: list[TaggedElement],
    tracker: ConfidenceTracker,
    context: ValidationContext | None = None,
    rules: list[ValidationRule] | None = None,
) -> list[TaggedElement]:
    """
    Run all validation rules on all elements.

    Modifies elements in-place (tag, confidence, review flags)
    and logs adjustments to the ConfidenceTracker.

    Args:
        elements: Tagged elements to validate.
        tracker: Confidence tracker for logging.
        context: Additional context (table structures, formulas, etc.).
        rules: Override default rules if needed.

    Returns:
        The same list of elements (modified in-place).
    """
    if context is None:
        context = ValidationContext(all_elements=elements)
    else:
        context.all_elements = elements

    if rules is None:
        rules = DEFAULT_RULES

    flagged_count = 0
    reclassified_count = 0

    for element in elements:
        for rule in rules:
            result = rule.check(element, context)
            if result is None:
                continue

            if result.action == "reclassify" and result.new_tag is not None:
                logger.info(
                    "Rule '%s': reclassifying %s from %s → %s: %s",
                    result.rule_name, element.element_id,
                    element.pdf_tag.value, result.new_tag.value,
                    result.reason,
                )
                element.pdf_tag = result.new_tag
                reclassified_count += 1

            if result.new_confidence is not None:
                tracker.adjust_confidence(
                    element,
                    new_confidence=result.new_confidence,
                    reason=f"[{result.rule_name}] {result.reason}",
                    stage="stage6_validator",
                )
                flagged_count += 1

            if result.action == "flag":
                tracker.flag_for_review(
                    element,
                    reason=f"[{result.rule_name}] {result.reason}",
                    stage="stage6_validator",
                )

            # Only apply the first matching rule per element
            break

    logger.info(
        "Validation complete: %d reclassified, %d flagged for review out of %d elements",
        reclassified_count, flagged_count, len(elements),
    )
    return elements
