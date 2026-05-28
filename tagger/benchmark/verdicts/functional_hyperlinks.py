"""Contract 3 — functional_hyperlinks (WCAG 2.4.4 / 2.4.9).

RECALIBRATED (hand-validation F3): experts passed a doc whose links are
functional via an /A action (51/51) but have NO /Contents (veraPDF flagged it).
So "functional" = links resolve to a valid action/destination, NOT the PDF/UA
/Contents key (/Contents is the fix WE add in remediation, not the expert bar).
Descriptiveness (2.4.9, "click here" vs meaningful) is a SEPARATE Gemma sub-axis.
"""
from tagger.benchmark.verdicts.base import CannotDeriveReason, Verdict


def _link_annots(pdf):
    out = []
    for page in pdf.pages:
        for a in (page.obj.get("/Annots") or []):
            if str(a.get("/Subtype")) == "/Link":
                out.append(a)
    return out


def verdict(pdf, pdf_path) -> Verdict:
    links = _link_annots(pdf)
    if not links:
        return Verdict.cannot(CannotDeriveReason.NoElementsOfType, links=0)

    functional = sum(1 for a in links
                     if a.get("/A") is not None or a.get("/Dest") is not None)
    detail = dict(links=len(links), functional=functional)
    if functional == len(links):
        return Verdict.passed(**detail)
    return Verdict.failed(**detail, note="some links lack an /A action or /Dest")
