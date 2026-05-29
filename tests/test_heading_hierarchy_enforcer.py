"""Tests for the Stage-8 PDF/UA-1 heading-hierarchy enforcer."""
from dataclasses import dataclass

from tagger.models.data_types import PDFTag
from tagger.stage8_semantic.heading_hierarchy_enforcer import enforce_heading_hierarchy


@dataclass
class FakeEl:
    pdf_tag: PDFTag
    text: str = "x"


def _seq(*pairs):
    return [FakeEl(tag, txt) for tag, txt in pairs]


def test_r1_no_skip_demotes_h1_to_h3():
    els = _seq((PDFTag.H1, "A"), (PDFTag.H3, "B"))
    stats = enforce_heading_hierarchy(els)
    assert els[0].pdf_tag == PDFTag.H1
    assert els[1].pdf_tag == PDFTag.H2  # was H3, demoted to H2
    assert stats["no_skip_demotions"] == 1


def test_r1_allows_going_back_down_any_amount():
    # H1 -> H2 -> H3 -> H1 is legal (going DOWN to a previous level is fine)
    els = _seq(
        (PDFTag.H1, "A"), (PDFTag.H2, "B"), (PDFTag.H3, "C"), (PDFTag.H1, "D"),
    )
    stats = enforce_heading_hierarchy(els)
    assert [e.pdf_tag for e in els] == [PDFTag.H1, PDFTag.H2, PDFTag.H3, PDFTag.H1]
    assert stats["no_skip_demotions"] == 0


def test_r2_first_heading_h2_gets_promoted_with_following():
    # Doc starts with H2 -> shift everyone up by 1
    els = _seq((PDFTag.H2, "A"), (PDFTag.H3, "B"))
    enforce_heading_hierarchy(els)
    assert els[0].pdf_tag == PDFTag.H1
    assert els[1].pdf_tag == PDFTag.H2


def test_r3_empty_heading_becomes_artifact():
    els = _seq((PDFTag.H1, "A"), (PDFTag.H2, "   "), (PDFTag.H2, "B"))
    stats = enforce_heading_hierarchy(els)
    assert els[1].pdf_tag == PDFTag.ARTIFACT
    assert stats["empty_artifacted"] == 1


def test_r4_punctuation_only_heading_becomes_artifact():
    els = _seq((PDFTag.H1, "Title"), (PDFTag.H2, "* * *"))
    stats = enforce_heading_hierarchy(els)
    assert els[1].pdf_tag == PDFTag.ARTIFACT
    assert stats["punct_only_artifacted"] == 1


def test_no_headings_is_a_noop():
    from tagger.models.data_types import PDFTag as T
    els = [FakeEl(T.P, "body")]
    stats = enforce_heading_hierarchy(els)
    assert els[0].pdf_tag == T.P
    assert stats["headings_seen"] == 0


def test_skip_after_artifact_still_collapsed():
    # H1, (empty H2 -> artifact), H4 should still demote H4 to H2 (the
    # empty H2 was dropped first so the sequence is now H1 -> H4)
    els = _seq(
        (PDFTag.H1, "Top"), (PDFTag.H2, "  "), (PDFTag.H4, "Deep"),
    )
    stats = enforce_heading_hierarchy(els)
    assert els[0].pdf_tag == PDFTag.H1
    assert els[1].pdf_tag == PDFTag.ARTIFACT
    assert els[2].pdf_tag == PDFTag.H2
    assert stats["no_skip_demotions"] == 1
    assert stats["empty_artifacted"] == 1
