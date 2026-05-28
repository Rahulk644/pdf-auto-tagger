"""Contract 4 — logical_reading_order (WCAG 1.3.2).

veraPDF only covers the MECHANISM (/Tabs /S); it cannot judge order CORRECTNESS.
Structural predicate = geometric monotonicity of struct-tree (assistive reading)
order vs page geometry. THRESHOLD = 0.85, calibrated on reading_order controlled
pairs (passed 0.867-0.905, failed 0.714-0.835; clean gap). HONESTY CAVEAT:
geometric monotonicity is a PROXY — solid for single-column, weak for
multi-column (separation held with multi-column aggregated, so the per-page
cannot_derive carve-out is a refinement, not required for v1).
"""
from tagger.benchmark.struct_utils import reading_monotonicity
from tagger.benchmark.verdicts.base import CannotDeriveReason, Verdict

MONOTONICITY_THRESHOLD = 0.85


def _tabs_ok(pdf) -> bool:
    return all(str(pg.obj.get("/Tabs")) == "/S" for pg in pdf.pages)


def verdict(pdf, pdf_path) -> Verdict:
    if pdf.Root.get("/StructTreeRoot") is None:
        return Verdict.failed(note="no struct tree / untagged")

    mono = reading_monotonicity(pdf, pdf_path)
    if mono is None:
        return Verdict.cannot(CannotDeriveReason.TrivialDoc, note="no content pairs")

    tabs = _tabs_ok(pdf)
    detail = dict(tabs_s=tabs, monotonicity=round(mono, 3),
                  threshold=MONOTONICITY_THRESHOLD)
    if tabs and mono >= MONOTONICITY_THRESHOLD:
        return Verdict.passed(**detail)
    note = "missing /Tabs /S" if not tabs else "reading order diverges from geometry"
    return Verdict.failed(**detail, note=note)
