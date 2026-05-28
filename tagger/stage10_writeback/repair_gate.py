"""Detect / classify / gate / repair for Stage-10 structural changes.

PREP policy (per Rajat) is that the tool never silently changes a client's
document. Our pipeline makes *additive* accessibility changes (struct tree,
marked content, artifact wrapping, link tagging) plus a few *modifying* repairs
that alter real source objects (font descriptors, content-stream show strings).
This module gates only the modifying repairs behind a user-controlled mode.

Boundary test (apply to any future repair): does the operation alter objects a
NON-accessibility tool sees — existing content, fonts, real objects? If yes it is
`MODIFYING` and gated; if it only adds or cleans up the accessibility layer it is
`ADDITIVE` and runs unconditionally.

Gated surface today = the 3 font repairs (CIDSet delete, .notdef strip, missing-
space strip). Everything else is additive and runs inline during tagging.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

ADDITIVE = "additive"
MODIFYING = "modifying"

# repair_mode values
AUTO = "auto"            # apply all modifying repairs (default; differentiator vs PREP)
CONFIRM = "confirm"      # apply only modifying repairs whose finding_id is approved
FLAG_ONLY = "flag-only"  # never apply modifying repairs; only report them
MODES = (AUTO, CONFIRM, FLAG_ONLY)


@dataclass
class Finding:
    """One detected structural repair. Modifying findings are gated by mode."""

    clause: str                       # veraPDF/UA clause it addresses
    location: str                     # page/object/element descriptor
    defect_description: str
    proposed_repair: str
    repair_type: str                  # ADDITIVE | MODIFYING
    severity: str = "blocks-compliance"   # | "quality"
    auto_safe: bool = True            # reversible / pixel-identical
    # The mutation to perform when applied; None for report-only findings.
    apply: Optional[Callable[[], None]] = field(default=None, repr=False, compare=False)
    status: str = "detected"          # detected|applied|pending|reported
    finding_id: str = field(default="", init=False)

    def __post_init__(self):
        if not self.finding_id:
            h = hashlib.sha1(f"{self.clause}|{self.location}".encode()).hexdigest()[:12]
            self.finding_id = f"{self.clause}-{h}"

    def to_dict(self) -> dict:
        return {
            "finding_id": self.finding_id,
            "clause": self.clause,
            "location": self.location,
            "defect_description": self.defect_description,
            "proposed_repair": self.proposed_repair,
            "repair_type": self.repair_type,
            "severity": self.severity,
            "auto_safe": self.auto_safe,
            "status": self.status,
        }


def gate_and_apply(
    findings: list[Finding],
    repair_mode: str = AUTO,
    approved_ids: Optional[set[str]] = None,
) -> list[Finding]:
    """Apply findings per mode, stamping each with a terminal status.

    Additive findings are always applied. Modifying findings: `auto` applies all,
    `confirm` applies only those whose finding_id is in `approved_ids`, `flag-only`
    applies none. Mutates each finding's `status` in place; returns the list.
    """
    if repair_mode not in MODES:
        raise ValueError(f"unknown repair_mode {repair_mode!r}; expected one of {MODES}")
    approved = set(approved_ids or ())

    for f in findings:
        if f.repair_type == ADDITIVE:
            _apply(f)
            continue
        # modifying
        if repair_mode == AUTO:
            _apply(f)
        elif repair_mode == CONFIRM:
            if f.finding_id in approved:
                _apply(f)
            else:
                f.status = "pending"
        else:  # FLAG_ONLY
            f.status = "reported"
    return findings


def _apply(f: Finding) -> None:
    if f.apply is not None:
        f.apply()
    f.status = "applied"


def build_report(findings: list[Finding], repair_mode: str) -> dict:
    """Structured findings report; focuses on the gated (modifying) repairs."""
    modifying = [f for f in findings if f.repair_type == MODIFYING]
    counts: dict[str, int] = {}
    for f in modifying:
        counts[f.status] = counts.get(f.status, 0) + 1
    return {
        "repair_mode": repair_mode,
        "summary": {
            "modifying_total": len(modifying),
            "applied": counts.get("applied", 0),
            "pending": counts.get("pending", 0),
            "reported": counts.get("reported", 0),
            "additive_total": sum(1 for f in findings if f.repair_type == ADDITIVE),
        },
        "findings": [f.to_dict() for f in modifying],
    }


def write_report(findings: list[Finding], path: str | Path, repair_mode: str) -> dict:
    report = build_report(findings, repair_mode)
    try:
        Path(path).write_text(json.dumps(report, indent=2))
    except Exception as e:  # report is non-fatal
        logger.warning("Could not write repair report to %s: %s", path, e)
    return report


def load_approved_ids(approval_file: str | Path | None) -> set[str]:
    """Read approved finding_ids (one per line, # comments allowed) for confirm mode."""
    if not approval_file:
        return set()
    p = Path(approval_file)
    if not p.exists():
        logger.warning("Approval file %s not found; treating as empty approval set.", p)
        return set()
    ids = set()
    for line in p.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            ids.add(line)
    return ids
