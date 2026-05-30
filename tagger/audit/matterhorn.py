"""Matterhorn Protocol 1.1 labelling layer over the ACT/PDF-UA audit.

The Matterhorn Protocol (PDF Association) is the conformance-testing model for
PDF/UA-1: 31 checkpoints / 136 failure conditions, each citing an ISO 14289-1
clause. PDF/UA auditors (PAC, axesCheck) report in Matterhorn failure-condition
IDs. This module is a pure REPORTING layer — it does not add checks; it maps the
rules `tagger.audit.act_rules` already evaluates to their Matterhorn failure-
condition IDs so our output can be spoken in the same language as PAC.

Failure-condition IDs/descriptions below are taken from the published Matterhorn
1.1 condition list. Where a rule we check has no single clean Matterhorn cousin
(e.g. "empty heading" is a WCAG/ACT rule, not a distinct Matterhorn FC) the
`confidence` field says so rather than inventing an ID.
"""
from __future__ import annotations

from dataclasses import dataclass

from tagger.audit.act_rules import AuditReport, audit_pdf


@dataclass(frozen=True)
class MatterhornCondition:
    fc_id: str            # e.g. "13-004" ("" when no direct FC exists)
    description: str
    iso_clause: str       # ISO 14289-1 clause the FC cites
    confidence: str       # "verified" | "checkpoint" | "no-direct-fc"


# rule_id (from act_rules) -> the Matterhorn failure conditions it exercises.
# A rule can map to several FCs (e.g. the title rule covers both dc:title and
# DisplayDocTitle). IDs verified against the published Matterhorn 1.1 list.
RULE_TO_MATTERHORN: dict[str, list[MatterhornCondition]] = {
    "ACT-6cfa84": [
        MatterhornCondition("13-004", "Figure tag alternative or replacement text missing",
                            "7.3", "verified"),
    ],
    "ACT-36b590": [
        # Empty heading is a WCAG 1.3.1 / ACT requirement; Matterhorn has no
        # dedicated empty-heading FC — closest is checkpoint 14 (headings).
        MatterhornCondition("", "Heading element is empty (no direct Matterhorn FC; WCAG 1.3.1)",
                            "7.4", "no-direct-fc"),
    ],
    "ACT-b40fd1": [
        MatterhornCondition("11-001", "Natural language for text in page content cannot be determined",
                            "7.2", "verified"),
    ],
    "PDFUA-7.4.2": [
        MatterhornCondition("14-003", "Numbered heading levels in descending sequence are skipped",
                            "7.4.2", "verified"),
        MatterhornCondition("14-002", "Uses numbered headings but the first heading tag is not H1",
                            "7.4.2", "verified"),
    ],
    "PDFUA-7.1-10": [
        MatterhornCondition("06-003", "Metadata stream does not contain dc:title",
                            "7.1", "verified"),
        MatterhornCondition("07-001", "ViewerPreferences dictionary does not contain DisplayDocTitle key",
                            "7.1", "verified"),
    ],
    "PDFUA-7.5.2": [
        # Caption association lives under the table/figure structure checkpoints;
        # no single verified FC id, mapped at checkpoint level.
        MatterhornCondition("", "Caption not associated with its Figure/Table (checkpoint-level)",
                            "7.5", "checkpoint"),
    ],
    "PDFUA-7.5.3": [
        MatterhornCondition("16-001", "An Lbl or LBody is not a child of an LI; or an LI is not a child of L",
                            "7.6", "checkpoint"),
    ],
    "PDFUA-7.1-1": [
        MatterhornCondition("01-005", "Content is neither marked as Artifact nor tagged as real content",
                            "7.1", "verified"),
        MatterhornCondition("06-002", "Catalog metadata stream lacks the PDF/UA identifier",
                            "5", "verified"),
    ],
}


@dataclass
class MatterhornResult:
    fc_id: str
    description: str
    iso_clause: str
    status: str            # pass / fail / not_applicable (from the source rule)
    source_rule: str
    confidence: str
    notes: str = ""


def to_matterhorn(report: AuditReport) -> list[MatterhornResult]:
    """Re-express an AuditReport as Matterhorn failure-condition results. Each
    rule's pass/fail/N-A status is propagated to every FC it maps to."""
    out: list[MatterhornResult] = []
    for r in report.results:
        for cond in RULE_TO_MATTERHORN.get(r.rule_id, []):
            out.append(MatterhornResult(
                fc_id=cond.fc_id or "(no FC)",
                description=cond.description,
                iso_clause=cond.iso_clause,
                status=r.status,
                source_rule=r.rule_id,
                confidence=cond.confidence,
                notes=r.notes,
            ))
    return out


def main() -> None:
    import sys
    if len(sys.argv) < 2:
        print("usage: python -m tagger.audit.matterhorn <pdf> [pdf ...]")
        sys.exit(2)
    for path in sys.argv[1:]:
        rep = audit_pdf(path)
        rows = to_matterhorn(rep)
        print(f"\n== {path} == (Matterhorn 1.1 view)")
        for m in rows:
            mark = {"pass": "PASS", "fail": "FAIL"}.get(m.status, " NA ")
            conf = "" if m.confidence == "verified" else f"  [{m.confidence}]"
            print(f"  [{mark}] {m.fc_id:<8} {m.description}  (ISO {m.iso_clause}){conf}")


if __name__ == "__main__":
    main()
