"""Contract 5a — alt_text_presence (WCAG 1.1.1). The deterministic PRIMARY axis.

Every /Figure struct elem must carry a non-empty /Alt (or /ActualText). HANDOFF:
this evaluates ONLY existing /Figure elems — a real image artifacted instead of
tagged Figure is a `semantic_tagging` figure-coverage failure, NOT an
alt_text_presence failure (exactly one criterion claims it). Quality (5b, whether
the alt actually describes the image) is the SEPARATE Gemma sub-axis in
gemma_quality.judge_alt_quality, reported apart.
"""
from tagger.benchmark.struct_utils import iter_struct_elems
from tagger.benchmark.verdicts.base import CannotDeriveReason, Verdict


def verdict(pdf, pdf_path) -> Verdict:
    figs = [node for tag, node in iter_struct_elems(pdf) if tag == "/Figure"]
    if not figs:
        return Verdict.cannot(CannotDeriveReason.NoElementsOfType, figures=0)
    missing = sum(1 for f in figs
                  if not (str(f.get("/Alt") or "").strip()
                          or str(f.get("/ActualText") or "").strip()))
    detail = dict(figures=len(figs), missing_alt=missing)
    return Verdict.passed(**detail) if missing == 0 else Verdict.failed(**detail)
