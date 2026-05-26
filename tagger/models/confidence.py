"""
Confidence scoring and human review queue management.

Every stage can adjust element confidence. Elements falling below the
review threshold are collected into a review queue for human inspection
or selective routing to the Gemma semantic validator.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

from tagger.config import PIPELINE
from tagger.models.data_types import TaggedElement


@dataclass
class ReviewItem:
    """An element that needs human review or Gemma validation."""

    element_id: str
    page_num: int
    current_tag: str
    text_snippet: str
    confidence: float
    reason: str
    stage: str
    """Which stage flagged this element (e.g., 'stage6_validator')."""


class ConfidenceTracker:
    """
    Accumulates confidence adjustments and manages the review queue.

    Usage:
        tracker = ConfidenceTracker()
        tracker.flag_for_review(element, reason="single-cell table", stage="stage6")
        ...
        report = tracker.generate_report()
    """

    def __init__(self, review_threshold: float | None = None):
        self.review_threshold = review_threshold or PIPELINE.review_threshold
        self._review_queue: list[ReviewItem] = []
        self._confidence_log: list[dict] = []

    def adjust_confidence(
        self,
        element: TaggedElement,
        new_confidence: float,
        reason: str,
        stage: str,
    ) -> None:
        """
        Update an element's confidence and log the change.

        If the new confidence falls below the review threshold,
        the element is automatically flagged for review.
        """
        old_confidence = element.confidence
        element.confidence = max(0.0, min(1.0, new_confidence))

        self._confidence_log.append({
            "element_id": element.element_id,
            "page_num": element.page_num,
            "old_confidence": round(old_confidence, 3),
            "new_confidence": round(element.confidence, 3),
            "reason": reason,
            "stage": stage,
        })

        if element.confidence < self.review_threshold:
            self.flag_for_review(element, reason=reason, stage=stage)

    def flag_for_review(
        self,
        element: TaggedElement,
        reason: str,
        stage: str,
    ) -> None:
        """Explicitly add an element to the human review queue."""
        element.needs_review = True
        element.review_reason = reason

        self._review_queue.append(ReviewItem(
            element_id=element.element_id,
            page_num=element.page_num,
            current_tag=element.pdf_tag.value if hasattr(element.pdf_tag, 'value') else str(element.pdf_tag),
            text_snippet=element.text[:100] if element.text else "",
            confidence=element.confidence,
            reason=reason,
            stage=stage,
        ))

    @property
    def review_queue(self) -> list[ReviewItem]:
        """All elements flagged for review, in insertion order."""
        return list(self._review_queue)

    @property
    def review_count(self) -> int:
        return len(self._review_queue)

    def generate_report(self) -> dict:
        """
        Generate a JSON-serializable confidence report.

        Includes:
        - Summary statistics
        - Full review queue
        - Confidence adjustment log
        """
        return {
            "summary": {
                "total_flagged_for_review": self.review_count,
                "review_threshold": self.review_threshold,
                "total_confidence_adjustments": len(self._confidence_log),
            },
            "review_queue": [
                {
                    "element_id": item.element_id,
                    "page_num": item.page_num,
                    "current_tag": item.current_tag,
                    "text_snippet": item.text_snippet,
                    "confidence": round(item.confidence, 3),
                    "reason": item.reason,
                    "stage": item.stage,
                }
                for item in self._review_queue
            ],
            "confidence_adjustments": self._confidence_log,
        }

    def save_report(self, output_path: str | Path) -> None:
        """Write the confidence report to a JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.generate_report(), f, indent=2, ensure_ascii=False)
