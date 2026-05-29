"""Stage 8c — PDF/UA-1 heading-hierarchy enforcer.

Runs AFTER `heading_ranker.assign_heading_levels` has chosen H1-H6 from font
tiers. This module re-walks the heading sequence in reading order and applies
the deterministic PDF/UA-1 / WCAG 1.3.1 rules the ranker can't see on its own:

R1. No skip (PDF/UA-1 clause 7.4.2):
    "Heading levels shall not be skipped in successive heading levels."
    Example: H1 -> H3 becomes H1 -> H2. The shift is applied cumulatively
    forward so the rest of the document's relative levels are preserved.

R2. First-heading-is-H1:
    A document whose first heading is H2/H3/... gets every heading promoted
    by the offset. Together with R1 this guarantees H1 anchors the doc.

R3. No empty heading (WCAG 1.3.1 / G130):
    A heading whose text is empty or whitespace-only is reclassified to
    /Artifact -- screen readers must skip it.

R4. Heading text not just punctuation (WCAG 2.4.6):
    A heading whose text is only non-alphanumeric symbols
    (e.g. "* * *", "---") is reclassified to /Artifact too.

The enforcer is structural / deterministic -- no font, no model. It runs after
the heading_ranker so it operates on the rank already assigned and can rely on
elements appearing in reading order in the list it receives.
"""
from __future__ import annotations

import logging
import re

from tagger.models.data_types import PDFTag, TaggedElement

logger = logging.getLogger(__name__)

_HEADING_TAGS = (PDFTag.H1, PDFTag.H2, PDFTag.H3, PDFTag.H4, PDFTag.H5, PDFTag.H6)
_TAG_TO_LEVEL = {t: i + 1 for i, t in enumerate(_HEADING_TAGS)}
_LEVEL_TO_TAG = {i + 1: t for i, t in enumerate(_HEADING_TAGS)}
_ALPHANUM_RE = re.compile(r"[A-Za-z0-9]")


def enforce_heading_hierarchy(elements: list[TaggedElement]) -> dict:
    """Apply R1-R4 above to `elements` in place. Returns a stats dict so the
    pipeline can log what was changed (and tests can assert on it)."""
    stats = {
        "no_skip_demotions": 0,
        "first_h1_promotions": 0,
        "empty_artifacted": 0,
        "punct_only_artifacted": 0,
        "headings_seen": 0,
    }

    # R3 + R4 first — drop empty / punctuation-only headings BEFORE we look at
    # the sequence, otherwise an empty H2 between H1 and H3 would falsely
    # look like a skip.
    for el in elements:
        if el.pdf_tag not in _HEADING_TAGS:
            continue
        txt = (el.text or "").strip()
        if not txt:
            el.pdf_tag = PDFTag.ARTIFACT
            stats["empty_artifacted"] += 1
        elif not _ALPHANUM_RE.search(txt):
            el.pdf_tag = PDFTag.ARTIFACT
            stats["punct_only_artifacted"] += 1

    # R2 — first heading should be H1. Compute the offset and shift everyone
    # by the same amount (so H2 -> H1 implies the rest of the doc loses one
    # level too). This preserves relative depth.
    headings = [el for el in elements if el.pdf_tag in _HEADING_TAGS]
    stats["headings_seen"] = len(headings)
    if not headings:
        return stats

    first_level = _TAG_TO_LEVEL[headings[0].pdf_tag]
    if first_level > 1:
        shift = first_level - 1
        for el in headings:
            lvl = _TAG_TO_LEVEL[el.pdf_tag] - shift
            lvl = max(1, lvl)  # clamp
            el.pdf_tag = _LEVEL_TO_TAG[lvl]
            stats["first_h1_promotions"] += 1

    # R1 — re-walk and collapse forward skips. If H{n} is followed by H{n+k}
    # with k>1, demote the second heading to H{n+1} and remember the same
    # shift for the rest of the document (so a nested H3 elsewhere stays
    # consistent relative to its parent).
    last_level = 0
    for el in headings:
        lvl = _TAG_TO_LEVEL[el.pdf_tag]
        # A heading can stay the same, go down by any amount, but can only go
        # UP by +1 from the previous heading. If it jumps further (last=1,
        # this=3), demote to last+1 (=2).
        if last_level == 0 or lvl <= last_level + 1:
            last_level = lvl
            continue
        new_lvl = last_level + 1
        stats["no_skip_demotions"] += 1
        el.pdf_tag = _LEVEL_TO_TAG[new_lvl]
        last_level = new_lvl

    if (stats["no_skip_demotions"] or stats["first_h1_promotions"]
            or stats["empty_artifacted"] or stats["punct_only_artifacted"]):
        logger.info(
            "Heading hierarchy: no-skip=%d, first-H1 shifts=%d, "
            "empty->artifact=%d, punct->artifact=%d (over %d headings)",
            stats["no_skip_demotions"], stats["first_h1_promotions"],
            stats["empty_artifacted"], stats["punct_only_artifacted"],
            stats["headings_seen"],
        )
    return stats
