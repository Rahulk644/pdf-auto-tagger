"""Contract 1 — semantic_tagging (WCAG 1.3.1).

RECALIBRATED (hand-validation F3): veraPDF conformance anti-correlated with the
expert verdict (passed had MORE veraPDF fails). The real discriminator is
STRUCTURE: an expert-passed doc has a heading hierarchy and/or list structure; an
expert-failed doc is FLAT (all paragraphs/spans, no headings, no lists). Tags are
RoleMap-resolved. Short legitimately-simple docs (< N text blocks) pass on
presence + non-degeneracy alone (no heading/list requirement).
"""
from tagger.benchmark.struct_utils import tag_counts_open
from tagger.benchmark.verdicts.base import CannotDeriveReason, Verdict

N_BLOCKS = 5  # below this many text blocks, don't require headings/lists
_HEADINGS = ("/H1", "/H2", "/H3", "/H4", "/H5", "/H6")
_TEXT = _HEADINGS + ("/P", "/LI")


def verdict(pdf, pdf_path) -> Verdict:
    counts = tag_counts_open(pdf)
    if not counts:
        return Verdict.failed(note="no struct tree / untagged")

    text_blocks = sum(counts.get(t, 0) for t in _TEXT)
    has_heading = any(counts.get(h, 0) for h in _HEADINGS)
    has_list = counts.get("/L", 0) > 0 and counts.get("/LI", 0) > 0
    # non-degenerate = there is real structure beyond bare Span/Document wrappers
    meaningful = text_blocks + counts.get("/Table", 0) + counts.get("/Figure", 0)
    non_degenerate = meaningful > 0

    detail = dict(text_blocks=text_blocks, has_heading=has_heading,
                  has_list=has_list, non_degenerate=non_degenerate)

    if not non_degenerate:
        return Verdict.failed(**detail, note="degenerate: no meaningful structure")
    if text_blocks < N_BLOCKS:
        # short, legitimately-simple doc: presence + non-degeneracy suffice
        return Verdict.passed(**detail, note="short doc; presence suffices")
    if has_heading or has_list:
        return Verdict.passed(**detail)
    return Verdict.failed(**detail, note="flat: no heading hierarchy or list structure")
