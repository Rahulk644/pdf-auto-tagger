"""Contract 3 — functional_hyperlinks (WCAG 2.4.4 / 2.4.9).

RECALIBRATED twice:
  - F3 (hand-validation): "functional" = link resolves to a valid action/dest,
    NOT the PDF/UA /Contents key (experts passed /Contents-less docs).
  - At-scale diagnosis (125-doc checker): the /A-only predicate scored 50% — 3/5
    expert-failed docs are links with /A but NOT tagged (no Link struct elem via
    OBJR), a real 7.18.5-1 structural defect the pipeline repairs (e0ec9af) and
    the predicate must measure. So: passed = every link has /A (or /Dest) AND is
    tagged via OBJR. Lifts agreement 50%->80%.
    RESIDUAL (irreducible, documented): W2991007371 + W3018272089 — passed/failed
    renditions IDENTICAL on every deterministic signal (tagging, targets, dest
    resolution, text); our predicate is slightly stricter than the lenient expert
    on these 2 untagged-but-passed docs. Descriptiveness (2.4.9) is the SEPARATE
    Gemma sub-axis and does NOT explain these (text is identical citation markers).
"""
from tagger.benchmark.struct_utils import objr_referenced_objgens
from tagger.benchmark.verdicts.base import CannotDeriveReason, Verdict


def _link_annots(pdf):
    out = []
    for page in pdf.pages:
        for a in (page.obj.get("/Annots") or []):
            if str(a.get("/Subtype")) == "/Link":
                out.append(a)
    return out


def _objgen(a):
    try:
        return a.objgen
    except Exception:
        return None


def verdict(pdf, pdf_path) -> Verdict:
    links = _link_annots(pdf)
    if not links:
        return Verdict.cannot(CannotDeriveReason.NoElementsOfType, links=0)

    tagged_set = objr_referenced_objgens(pdf, "/Link")
    functional = sum(1 for a in links
                     if a.get("/A") is not None or a.get("/Dest") is not None)
    tagged = sum(1 for a in links if _objgen(a) in tagged_set)
    detail = dict(links=len(links), functional=functional, tagged=tagged)

    if functional == len(links) and tagged == len(links):
        return Verdict.passed(**detail)
    notes = []
    if functional < len(links):
        notes.append("some links lack /A or /Dest")
    if tagged < len(links):
        notes.append("some links not tagged (no Link struct elem via OBJR)")
    return Verdict.failed(**detail, note="; ".join(notes))
