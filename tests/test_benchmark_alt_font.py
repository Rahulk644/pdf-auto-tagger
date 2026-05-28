"""Unit 4 — alt_text_presence (5a), font_embedding (adjacent), Gemma stubs (5b),
and dispatch handling of not-addressed criteria."""
import pikepdf
from pikepdf import Array, Dictionary, Name

from tagger.benchmark import gemma_quality
from tagger.benchmark.verdicts import alt_text, font_embedding
from tagger.benchmark.verdicts.base import CannotDeriveReason, derive_verdict


def _pdf_with_figures(tmp_path, name, alts):
    """1-page PDF with one /Figure struct elem per entry in `alts` (None = no /Alt)."""
    pdf = pikepdf.new()
    pdf.add_blank_page(page_size=(200, 200))
    figs = []
    for a in alts:
        d = {"/S": Name("/Figure")}
        if a is not None:
            d["/Alt"] = pikepdf.String(a)
        figs.append(pdf.make_indirect(Dictionary(d)))
    doc = pdf.make_indirect(Dictionary({"/S": Name("/Document"), "/K": Array(figs)}))
    pdf.Root["/StructTreeRoot"] = pdf.make_indirect(
        Dictionary({"/Type": Name.StructTreeRoot, "/K": doc}))
    p = tmp_path / name
    pdf.save(str(p)); pdf.close()
    return str(p)


def test_alt_presence_passed(tmp_path):
    p = _pdf_with_figures(tmp_path, "a.pdf", ["a desc", "b desc"])
    with pikepdf.open(p) as d:
        assert alt_text.verdict(d, p).status == "passed"


def test_alt_presence_failed_missing(tmp_path):
    p = _pdf_with_figures(tmp_path, "b.pdf", ["a desc", None])
    with pikepdf.open(p) as d:
        v = alt_text.verdict(d, p)
    assert v.status == "failed" and v.detail["missing_alt"] == 1


def test_alt_presence_no_figures_cannot_derive(tmp_path):
    pdf = pikepdf.new(); pdf.add_blank_page(page_size=(200, 200))
    doc = pdf.make_indirect(Dictionary({"/S": Name("/Document"), "/K": Array([])}))
    pdf.Root["/StructTreeRoot"] = pdf.make_indirect(
        Dictionary({"/Type": Name.StructTreeRoot, "/K": doc}))
    p = tmp_path / "c.pdf"; pdf.save(str(p)); pdf.close()
    with pikepdf.open(str(p)) as d:
        v = alt_text.verdict(d, str(p))
    assert v.status == "cannot_derive" and v.reason == CannotDeriveReason.NoElementsOfType


def _font_pdf(tmp_path, name, embedded):
    pdf = pikepdf.new(); pdf.add_blank_page(page_size=(200, 200))
    page = pdf.pages[0]
    fd = {"/Type": Name.FontDescriptor, "/FontName": Name("/Arial"), "/Flags": 32}
    if embedded:
        fd["/FontFile2"] = pdf.make_stream(b"\x00")
    fdo = pdf.make_indirect(Dictionary(fd))
    font = pdf.make_indirect(Dictionary({
        "/Type": Name.Font, "/Subtype": Name.TrueType,
        "/BaseFont": Name("/Arial"), "/FontDescriptor": fdo}))
    page.obj["/Resources"] = Dictionary({"/Font": Dictionary({"/F1": font})})
    p = tmp_path / name; pdf.save(str(p)); pdf.close()
    return str(p)


def test_font_embedding_passed(tmp_path):
    p = _font_pdf(tmp_path, "f1.pdf", embedded=True)
    with pikepdf.open(p) as d:
        assert font_embedding.verdict(d, p).status == "passed"


def test_font_embedding_failed(tmp_path):
    p = _font_pdf(tmp_path, "f2.pdf", embedded=False)
    with pikepdf.open(p) as d:
        v = font_embedding.verdict(d, p)
    assert v.status == "failed" and v.detail["unembedded_fonts"] == 1


def test_gemma_stubs_cannot_derive():
    assert gemma_quality.judge_alt_quality("x.jpg", "alt").reason == CannotDeriveReason.MissingFeature
    assert gemma_quality.judge_link_descriptiveness("click here").reason == CannotDeriveReason.MissingFeature


def test_dispatch_not_addressed_criteria(tmp_path):
    # color_contrast + fonts_readability are not addressed -> MissingFeature, not failed
    p = _font_pdf(tmp_path, "d.pdf", embedded=True)
    for crit in ("color_contrast", "fonts_readability"):
        v = derive_verdict(p, crit)
        assert v.status == "cannot_derive" and v.reason == CannotDeriveReason.MissingFeature


def test_dispatch_routes_alt_and_font(tmp_path):
    p = _pdf_with_figures(tmp_path, "e.pdf", ["desc"])
    assert derive_verdict(p, "alt_text_quality").status == "passed"
    p2 = _font_pdf(tmp_path, "g.pdf", embedded=False)
    assert derive_verdict(p2, "font_embedding").status == "failed"
