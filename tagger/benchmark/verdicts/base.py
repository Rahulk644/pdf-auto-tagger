"""Verdict contract: one `derive_verdict(pdf_path, criterion) -> Verdict`.

A Verdict is the deterministic (veraPDF + structural) reading of one accessibility
criterion on one PDF. Applied to our OUTPUT it feeds remediation-rate; applied to
the ORIGINAL it feeds checker-agreement (see harness). NOT a substitute for the
expert WCAG verdict — it derives from the expert DISCRIMINATOR per the calibrated
contracts (see project memory project-benchmark-pdfa-design).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pikepdf


class CannotDeriveReason(Enum):
    PipelineError = "pipeline_error"
    Scanned = "scanned"
    TrivialDoc = "trivial_doc"
    ProxyUnreliable = "proxy_unreliable"
    MissingFeature = "missing_feature"      # criterion not addressed by this pipeline
    NoElementsOfType = "no_elements_of_type"  # e.g. no figures/links -> not_present


@dataclass
class Verdict:
    status: str  # "passed" | "failed" | "cannot_derive"
    reason: CannotDeriveReason | None = None
    detail: dict = field(default_factory=dict)  # evidence for audit

    @classmethod
    def passed(cls, **detail) -> "Verdict":
        return cls("passed", None, detail)

    @classmethod
    def failed(cls, **detail) -> "Verdict":
        return cls("failed", None, detail)

    @classmethod
    def cannot(cls, reason: CannotDeriveReason, **detail) -> "Verdict":
        return cls("cannot_derive", reason, detail)

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "reason": self.reason.value if self.reason else None,
            "detail": self.detail,
        }


def derive_verdict(pdf_path: str, criterion: str) -> Verdict:
    """Dispatch to the per-criterion verdict function. Criteria this pipeline does
    not address (color_contrast; fonts_readability readability axis) -> cannot_derive
    (MissingFeature) so the scorecard reports them as "not addressed", never failed.
    """
    from tagger.benchmark.verdicts import (
        alt_text,
        font_embedding,
        functional_hyperlinks,
        logical_reading_order,
        semantic_tagging,
        table_structure,
        unicode_mapping,
    )

    dispatch = {
        "semantic_tagging": semantic_tagging.verdict,
        "table_structure": table_structure.verdict,
        "functional_hyperlinks": functional_hyperlinks.verdict,
        "logical_reading_order": logical_reading_order.verdict,
        # alt_text_quality PRIMARY = deterministic presence (5a); the Gemma quality
        # sub-axis (5b) is reported separately via gemma_quality, not here.
        "alt_text_quality": alt_text.verdict,
        # font_embedding + unicode_mapping = adjacent FILE-CONFORMANCE axes (NOT
        # benchmark-labeled criteria); surfaced as their own scorecard columns.
        "font_embedding": font_embedding.verdict,
        "unicode_mapping": unicode_mapping.verdict,
    }
    fn = dispatch.get(criterion)
    if fn is None:
        return Verdict.cannot(CannotDeriveReason.MissingFeature,
                              note=f"{criterion} not addressed by tagger")
    try:
        with pikepdf.open(pdf_path) as pdf:
            return fn(pdf, pdf_path)
    except Exception as e:  # never let one doc crash the harness
        return Verdict.cannot(CannotDeriveReason.PipelineError, error=str(e))
