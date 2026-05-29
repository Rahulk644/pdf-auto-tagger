"""Stage 8d — PDF/UA-1 + WCAG / W3C-ACT structural enforcer.

Companion to heading_hierarchy_enforcer.py. Runs on the flat TaggedElement
list after every other Stage-8 pass (heading_ranker, toc_detector,
artifact_detector, caption_detector, list_builder) and applies the
deterministic structural rules the upstream passes don't enforce on their
own. Modifies elements in place.

Rules (all derive from PDF/UA-1 / WCAG 1.3.1 / W3C ACT):

S1. Empty body element -> /Artifact.
    /P, /Caption, /Note, /BlockQuote with empty or whitespace-only text are
    reclassified. (Empty headings are already handled in R3 of the heading
    enforcer.) Empty blocks in the struct tree are an ACT failure for
    "Text element has accessible name" / WCAG 1.3.1.

S2. Punctuation-only body element -> /Artifact.
    Same as the heading R4 but for body /P (e.g. a stray "* * *" that the
    figure-caption pass kept around).

S3. Figure missing /Alt -> set placeholder.
    PDF/UA-1 clause 7.18.4 requires every /Figure to have an alt text or
    /ActualText. We already inject placeholders in Stage 9, but a /Figure
    inserted later (e.g. by a specialist) might slip through; this is the
    belt-and-braces guarantee.

S4. Caption with no adjacent Figure/Table -> demote to /P.
    Per PDF/UA-1 7.5.2, /Caption must structurally accompany a Figure or
    Table. A floating /Caption is an ACT failure.

This is deterministic, doc-agnostic, and font-free — it just walks the
element list and applies the rules.
"""
from __future__ import annotations

import logging
import re

from tagger.models.data_types import PDFTag, TaggedElement

logger = logging.getLogger(__name__)

_BODY_TAGS = {PDFTag.P, PDFTag.CAPTION, PDFTag.BLOCKQUOTE, PDFTag.NOTE}
_ALPHANUM_RE = re.compile(r"[A-Za-z0-9]")
_PLACEHOLDER_ALT = "Figure (description needed)."
_CAPTION_NEIGHBOUR_PX = 80  # distance allowed between a Caption and Figure/Table
_FIGURE_OR_TABLE = {PDFTag.FIGURE, PDFTag.TABLE}


def enforce_pdfua_structural(elements: list[TaggedElement]) -> dict:
    """Apply S1-S4 to `elements` in place. Returns a stats dict."""
    stats = {
        "empty_body_artifacted": 0,
        "punct_body_artifacted": 0,
        "figure_alt_filled": 0,
        "caption_demoted_to_p": 0,
    }

    # S1 + S2 — body-element cleanup
    for el in elements:
        if el.pdf_tag not in _BODY_TAGS:
            continue
        txt = (el.text or "").strip()
        if not txt:
            el.pdf_tag = PDFTag.ARTIFACT
            stats["empty_body_artifacted"] += 1
        elif not _ALPHANUM_RE.search(txt):
            el.pdf_tag = PDFTag.ARTIFACT
            stats["punct_body_artifacted"] += 1

    # S3 — every surviving Figure must have alt_text
    for el in elements:
        if el.pdf_tag == PDFTag.FIGURE and not (el.alt_text or "").strip():
            el.alt_text = _PLACEHOLDER_ALT
            el.needs_review = True
            stats["figure_alt_filled"] += 1

    # S4 — Caption must structurally accompany a Figure or Table.
    # Heuristic: a Caption whose bbox is within _CAPTION_NEIGHBOUR_PX of a
    # Figure or Table on the same page (above OR below) is fine; any other
    # /Caption is demoted to /P (still readable, just not falsely associated).
    by_page: dict = {}
    for el in elements:
        if el.pdf_tag in _FIGURE_OR_TABLE:
            by_page.setdefault(el.page_num, []).append(el)
    for el in elements:
        if el.pdf_tag != PDFTag.CAPTION:
            continue
        neighbours = by_page.get(el.page_num, [])
        if not _has_neighbour(el, neighbours):
            el.pdf_tag = PDFTag.P
            stats["caption_demoted_to_p"] += 1

    if any(v for v in stats.values()):
        logger.info(
            "PDF/UA structural enforcer: empty-body->artifact=%d, "
            "punct-body->artifact=%d, figure-alt-filled=%d, "
            "caption->P=%d",
            stats["empty_body_artifacted"], stats["punct_body_artifacted"],
            stats["figure_alt_filled"], stats["caption_demoted_to_p"],
        )
    return stats


def _has_neighbour(cap: TaggedElement, others: list[TaggedElement]) -> bool:
    cx0, cy0, cx1, cy1 = cap.bbox
    for o in others:
        ox0, oy0, ox1, oy1 = o.bbox
        # vertical gap above OR below within threshold AND any x-overlap
        x_overlap = min(cx1, ox1) - max(cx0, ox0)
        if x_overlap <= 0:
            continue
        gap_below = cy0 - oy1  # caption is below the figure
        gap_above = oy0 - cy1  # caption is above the figure
        if (0 <= gap_below <= _CAPTION_NEIGHBOUR_PX
                or 0 <= gap_above <= _CAPTION_NEIGHBOUR_PX):
            return True
    return False
