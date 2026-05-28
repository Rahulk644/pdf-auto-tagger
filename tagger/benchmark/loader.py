"""Unit 3 — benchmark loader.

Reads `data/dataset.json`, inventories /StructTreeRoot per doc, and yields one
DocTask per (doc x criterion) the doc has a label for. Pure read — no pipeline
invocation. The harness consumes this iterator.

Routing note: the loader records `is_tagged` (StructTreeRoot present). The
RATIFIED remediation policy is strip+V2-for-all (all benchmark docs are
pre-tagged, so V1 retag wouldn't meaningfully remediate) — see project memory
project-benchmark-impl-design. `is_tagged` is kept for reporting + the deferred
alternative routing, NOT as a regen-scope filter under the current policy.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import pikepdf

# The 7 benchmark criteria. The 4 fully-addressable have verdict fns; alt_text +
# font_embedding (Unit 4) are partial; color_contrast + fonts_readability are
# not addressed (reported transparently, never scored as failed).
ADDRESSED_CRITERIA = frozenset({
    "semantic_tagging", "table_structure", "functional_hyperlinks",
    "logical_reading_order", "alt_text_quality", "fonts_readability",
})


@dataclass
class DocTask:
    openalex_id: str
    criterion: str
    expert_label: str           # passed | failed | not_present | cannot_tell
    pdf_path: str               # absolute, resolved against benchmark_root
    is_tagged: bool             # /StructTreeRoot present in the original
    normalized_compliance: float | None = None
    adobe6_compliance: bool | None = None
    input_pdfs: list[str] = field(default_factory=list)
    load_error: str | None = None  # set if the PDF can't be opened/inventoried

    @property
    def route(self) -> str:
        """Original-state routing (informational). Remediation policy = strip+V2."""
        return "V1" if self.is_tagged else "V2"


def _inventory(pdf_path: Path) -> tuple[bool, str | None]:
    """(is_tagged, load_error) — opens the PDF to check /StructTreeRoot."""
    if not pdf_path.exists():
        return False, f"missing: {pdf_path}"
    try:
        with pikepdf.open(str(pdf_path)) as pdf:
            return pdf.Root.get("/StructTreeRoot") is not None, None
    except Exception as e:
        return False, str(e)


def load_benchmark(
    benchmark_root: str | Path,
    dataset_json: str | Path | None = None,
) -> Iterator[DocTask]:
    """Yield a DocTask per (doc x criterion) with a label, routing recorded.

    NP/CT labels are NOT dropped here — they are modeled in the report.
    """
    root = Path(benchmark_root)
    ds = Path(dataset_json) if dataset_json else root / "data" / "dataset.json"
    data = json.loads(Path(ds).read_text())
    for criterion, labels in data.get("tasks", {}).items():
        for label, entries in labels.items():
            for e in entries:
                pdf_path = (root / e["pdf_path"]) if e.get("pdf_path") else root
                is_tagged, err = _inventory(pdf_path)
                yield DocTask(
                    openalex_id=e.get("openalex_id", "?"),
                    criterion=criterion,
                    expert_label=label,
                    pdf_path=str(pdf_path),
                    is_tagged=is_tagged,
                    normalized_compliance=e.get("normalized_compliance"),
                    adobe6_compliance=e.get("adobe6_compliance"),
                    input_pdfs=[str(root / p) for p in e.get("input_pdfs", [])],
                    load_error=err,
                )
