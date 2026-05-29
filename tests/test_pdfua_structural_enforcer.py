"""Tests for the Stage-8 PDF/UA-1 structural enforcer (S1-S4)."""
from dataclasses import dataclass, field

from tagger.models.data_types import PDFTag
from tagger.stage8_semantic.pdfua_structural_enforcer import enforce_pdfua_structural


@dataclass
class FakeEl:
    pdf_tag: PDFTag
    text: str = "x"
    bbox: tuple = (0.0, 0.0, 100.0, 20.0)
    page_num: int = 1
    alt_text: str = ""
    needs_review: bool = False


def test_s1_empty_p_becomes_artifact():
    els = [FakeEl(PDFTag.P, ""), FakeEl(PDFTag.P, "real body")]
    stats = enforce_pdfua_structural(els)
    assert els[0].pdf_tag == PDFTag.ARTIFACT
    assert els[1].pdf_tag == PDFTag.P
    assert stats["empty_body_artifacted"] == 1


def test_s2_punctuation_only_p_becomes_artifact():
    els = [FakeEl(PDFTag.P, "* * *"), FakeEl(PDFTag.P, "Hello.")]
    stats = enforce_pdfua_structural(els)
    assert els[0].pdf_tag == PDFTag.ARTIFACT
    assert els[1].pdf_tag == PDFTag.P
    assert stats["punct_body_artifacted"] == 1


def test_s3_figure_without_alt_gets_placeholder():
    fig = FakeEl(PDFTag.FIGURE, "")  # figure with no alt_text
    stats = enforce_pdfua_structural([fig])
    assert fig.alt_text  # populated
    assert fig.needs_review is True
    assert stats["figure_alt_filled"] == 1


def test_s3_figure_with_alt_is_unchanged():
    fig = FakeEl(PDFTag.FIGURE, "")
    fig.alt_text = "Photograph."
    enforce_pdfua_structural([fig])
    assert fig.alt_text == "Photograph."
    assert fig.needs_review is False


def test_s4_caption_adjacent_to_figure_kept():
    fig = FakeEl(PDFTag.FIGURE, "", bbox=(10.0, 10.0, 100.0, 100.0), page_num=1)
    cap = FakeEl(PDFTag.CAPTION, "Figure 1: ...",
                 bbox=(10.0, 110.0, 100.0, 130.0), page_num=1)  # just below figure
    stats = enforce_pdfua_structural([fig, cap])
    assert cap.pdf_tag == PDFTag.CAPTION  # kept
    assert stats["caption_demoted_to_p"] == 0


def test_s4_orphan_caption_demoted_to_p():
    cap = FakeEl(PDFTag.CAPTION, "A free-floating caption.",
                 bbox=(10.0, 110.0, 100.0, 130.0), page_num=1)
    stats = enforce_pdfua_structural([cap])
    assert cap.pdf_tag == PDFTag.P  # demoted
    assert stats["caption_demoted_to_p"] == 1


def test_s4_caption_on_wrong_page_demoted():
    fig = FakeEl(PDFTag.FIGURE, "", bbox=(10.0, 10.0, 100.0, 100.0), page_num=1)
    cap = FakeEl(PDFTag.CAPTION, "Caption", bbox=(10.0, 110.0, 100.0, 130.0),
                 page_num=2)  # different page
    enforce_pdfua_structural([fig, cap])
    assert cap.pdf_tag == PDFTag.P
