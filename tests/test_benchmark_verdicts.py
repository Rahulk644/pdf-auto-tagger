"""Verdict-derivation functions (the 4 fully-addressable criteria).

Synthetic mini-PDFs exercise each locked/recalibrated contract deterministically;
a skipif'd integration test confirms the verdicts match the hand-derivation on the
downloaded benchmark sample.
"""
from pathlib import Path

import pikepdf
import pytest
from pikepdf import Array, Dictionary, Name

from tagger.benchmark.verdicts.base import CannotDeriveReason, derive_verdict
from tagger.benchmark.verdicts import (
    functional_hyperlinks,
    logical_reading_order,
    semantic_tagging,
    table_structure,
)


def _se(pdf, s, **extra):
    return pdf.make_indirect(Dictionary({"/S": Name(s), **extra}))


def _struct_pdf(tmp_path, name, make_kids):
    """1-page PDF whose Document /K is built by make_kids(pdf) (same Pdf, no cross-refs)."""
    pdf = pikepdf.new()
    pdf.add_blank_page(page_size=(612, 792))
    doc = pdf.make_indirect(Dictionary({"/S": Name("/Document"), "/K": Array(make_kids(pdf))}))
    sr = pdf.make_indirect(Dictionary({"/Type": Name.StructTreeRoot, "/K": doc}))
    pdf.Root["/StructTreeRoot"] = sr
    pdf.Root["/MarkInfo"] = Dictionary({"/Marked": True})
    p = tmp_path / name
    pdf.save(str(p))
    pdf.close()
    return str(p)


def _open(path):
    return pikepdf.open(path)


# ----------------------------------------------------------- semantic_tagging ---

def test_semantic_passed_with_heading(tmp_path):
    p = _struct_pdf(tmp_path, "a.pdf",
                    lambda pdf: [_se(pdf, "/H1")] + [_se(pdf, "/P") for _ in range(5)])
    with _open(p) as d:
        assert semantic_tagging.verdict(d, p).status == "passed"


def test_semantic_failed_flat(tmp_path):
    p = _struct_pdf(tmp_path, "b.pdf", lambda pdf: [_se(pdf, "/P") for _ in range(6)])
    with _open(p) as d:
        v = semantic_tagging.verdict(d, p)
    assert v.status == "failed" and "flat" in v.detail.get("note", "")


def test_semantic_short_doc_passes_on_presence(tmp_path):
    p = _struct_pdf(tmp_path, "c.pdf", lambda pdf: [_se(pdf, "/P"), _se(pdf, "/P")])
    with _open(p) as d:
        assert semantic_tagging.verdict(d, p).status == "passed"


def test_semantic_untagged_failed(tmp_path):
    pdf = pikepdf.new()
    pdf.add_blank_page(page_size=(612, 792))
    p = tmp_path / "u.pdf"
    pdf.save(str(p))
    pdf.close()
    with _open(str(p)) as d:
        assert semantic_tagging.verdict(d, str(p)).status == "failed"


# ------------------------------------------------------------ table_structure ---

def _table(pdf, with_th):
    cells = [_se(pdf, "/TH" if with_th else "/TD"), _se(pdf, "/TD")]
    tr = pdf.make_indirect(Dictionary({"/S": Name("/TR"), "/K": Array(cells)}))
    return pdf.make_indirect(Dictionary({"/S": Name("/Table"), "/K": Array([tr])}))


def test_table_passed_with_th(tmp_path):
    p = _struct_pdf(tmp_path, "t1.pdf", lambda pdf: [_table(pdf, True)])
    with _open(p) as d:
        assert table_structure.verdict(d, p).status == "passed"


def test_table_failed_no_th(tmp_path):
    p = _struct_pdf(tmp_path, "t2.pdf", lambda pdf: [_table(pdf, False)])
    with _open(p) as d:
        v = table_structure.verdict(d, p)
    assert v.status == "failed" and "TH" in v.detail.get("note", "")


def test_table_failed_no_table(tmp_path):
    p = _struct_pdf(tmp_path, "t3.pdf", lambda pdf: [_se(pdf, "/P")])
    with _open(p) as d:
        assert table_structure.verdict(d, p).status == "failed"


# ------------------------------------------------------- functional_hyperlinks ---

def _link_pdf(tmp_path, name, with_action):
    pdf = pikepdf.new()
    pdf.add_blank_page(page_size=(612, 792))
    page = pdf.pages[0]
    a = Dictionary({"/Subtype": Name.Link, "/Rect": Array([0, 0, 10, 10])})
    if with_action:
        a["/A"] = Dictionary({"/S": Name("/URI"), "/URI": pikepdf.String("https://x")})
    page.obj["/Annots"] = Array([pdf.make_indirect(a)])
    p = tmp_path / name
    pdf.save(str(p)); pdf.close()
    return str(p)


def test_links_passed_with_action(tmp_path):
    p = _link_pdf(tmp_path, "l1.pdf", True)
    with _open(p) as d:
        assert functional_hyperlinks.verdict(d, p).status == "passed"


def test_links_failed_without_action(tmp_path):
    p = _link_pdf(tmp_path, "l2.pdf", False)
    with _open(p) as d:
        assert functional_hyperlinks.verdict(d, p).status == "failed"


def test_links_none_cannot_derive(tmp_path):
    pdf = pikepdf.new(); pdf.add_blank_page(page_size=(612, 792))
    p = tmp_path / "l3.pdf"; pdf.save(str(p)); pdf.close()
    with _open(str(p)) as d:
        v = functional_hyperlinks.verdict(d, str(p))
    assert v.status == "cannot_derive" and v.reason == CannotDeriveReason.NoElementsOfType


# ------------------------------------------------------ logical_reading_order ---

def _reading_pdf(tmp_path, name, order, tabs=True):
    """3 text runs at descending y (MCID 0 top..2 bottom); struct /K in `order`."""
    pdf = pikepdf.new()
    pdf.add_blank_page(page_size=(612, 792))
    page = pdf.pages[0]
    page.obj["/Resources"] = Dictionary(
        {"/Font": Dictionary({"/F1": pdf.make_indirect(Dictionary(
            {"/Type": Name.Font, "/Subtype": Name.Type1, "/BaseFont": Name("/Helvetica")}))})})
    cs = (b"BT /F1 12 Tf 1 0 0 1 50 700 Tm /P <</MCID 0>> BDC (Alpha) Tj EMC "
          b"1 0 0 1 50 500 Tm /P <</MCID 1>> BDC (Beta) Tj EMC "
          b"1 0 0 1 50 300 Tm /P <</MCID 2>> BDC (Gamma) Tj EMC ET")
    page.obj["/Contents"] = pdf.make_stream(cs)
    if tabs:
        page.obj["/Tabs"] = Name("/S")
    elems = {i: pdf.make_indirect(Dictionary(
        {"/S": Name("/P"), "/Pg": page.obj, "/K": i})) for i in range(3)}
    doc = pdf.make_indirect(Dictionary(
        {"/S": Name("/Document"), "/K": Array([elems[i] for i in order])}))
    sr = pdf.make_indirect(Dictionary({"/Type": Name.StructTreeRoot, "/K": doc}))
    pdf.Root["/StructTreeRoot"] = sr
    pdf.Root["/MarkInfo"] = Dictionary({"/Marked": True})
    p = tmp_path / name
    pdf.save(str(p)); pdf.close()
    return str(p)


def test_reading_order_passed_monotonic(tmp_path):
    p = _reading_pdf(tmp_path, "r1.pdf", order=[0, 1, 2])
    with _open(p) as d:
        v = logical_reading_order.verdict(d, p)
    assert v.status == "passed" and v.detail["monotonicity"] == 1.0


def test_reading_order_failed_scrambled(tmp_path):
    p = _reading_pdf(tmp_path, "r2.pdf", order=[2, 1, 0])
    with _open(p) as d:
        v = logical_reading_order.verdict(d, p)
    assert v.status == "failed"


def test_reading_order_failed_no_tabs(tmp_path):
    p = _reading_pdf(tmp_path, "r3.pdf", order=[0, 1, 2], tabs=False)
    with _open(p) as d:
        assert logical_reading_order.verdict(d, p).status == "failed"


# ---------------------------------------------- integration on benchmark sample ---

_SAMPLE = Path("/tmp/pdfa_sample")
_skip = pytest.mark.skipif(not _SAMPLE.exists(), reason="benchmark sample not present")


@_skip
@pytest.mark.parametrize("fn,crit,expect", [
    ("sem_passed.pdf", "semantic_tagging", "passed"),
    ("sem_failed.pdf", "semantic_tagging", "failed"),
    ("table_passed.pdf", "table_structure", "passed"),
    ("links_passed.pdf", "functional_hyperlinks", "passed"),
    ("ro_W04_passed.pdf", "logical_reading_order", "passed"),
    ("ro_W04_failed.pdf", "logical_reading_order", "failed"),
    ("ro_W38_passed.pdf", "logical_reading_order", "passed"),
    ("ro_W38_failed.pdf", "logical_reading_order", "failed"),
])
def test_sample_matches_hand_derivation(fn, crit, expect):
    assert derive_verdict(str(_SAMPLE / fn), crit).status == expect
