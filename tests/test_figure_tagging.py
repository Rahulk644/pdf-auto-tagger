"""Stage 10 figure tagging: image paint-ops inside a Figure region become /Figure.

Image-only Figure regions carry no text → no MCID → were dropped at writeback and
their images fell through to /Artifact (no alt text). These tests exercise the real
tag_untagged_pdf end-to-end on a synthetic PDF (no MinerU needed): an image inside
a Figure element's bbox must become a /Figure struct elem with /Alt and /Figure
marked content; an image outside any figure must stay an Artifact.
"""
import pikepdf
import pytest

from tagger.models.data_types import PDFTag, TaggedElement
from tagger.stage10_writeback.content_stream_writer import (
    _affine_mul,
    _image_rect_std,
)
from tagger.stage10_writeback.struct_tree_writer import tag_untagged_pdf


def _make_pdf_with_image(path, cm):
    """1-page 200x200 PDF that paints a 2x2 image via `q <cm> cm /Im0 Do Q`."""
    pdf = pikepdf.new()
    pdf.add_blank_page(page_size=(200, 200))
    page = pdf.pages[0]
    img = pdf.make_stream(b"\xff\x00\x00\x00\xff\x00\x00\x00\xff\xff\xff\x00")
    img["/Type"] = pikepdf.Name.XObject
    img["/Subtype"] = pikepdf.Name.Image
    img["/Width"] = 2
    img["/Height"] = 2
    img["/ColorSpace"] = pikepdf.Name.DeviceRGB
    img["/BitsPerComponent"] = 8
    page.obj["/Resources"] = pikepdf.Dictionary(
        {"/XObject": pikepdf.Dictionary({"/Im0": img})}
    )
    cmstr = " ".join(str(v) for v in cm)
    page.obj["/Contents"] = pdf.make_stream(f"q {cmstr} cm /Im0 Do Q".encode())
    pdf.save(str(path))
    pdf.close()


def _figure(bbox, eid="p1_fig0"):
    return TaggedElement(
        element_id=eid, page_num=1, pdf_tag=PDFTag.FIGURE, text="",
        bbox=bbox, alt_text="A test figure.", merged_from=[],
    )


def _struct_tags(pdf):
    found = []

    def walk(n):
        if isinstance(n, pikepdf.Dictionary):
            if n.get("/S") is not None:
                found.append(n)
            k = n.get("/K")
            for c in (k if isinstance(k, pikepdf.Array) else [k] if k is not None else []):
                walk(c)

    walk(pdf.Root.StructTreeRoot.get("/K"))
    return found


def _bdc_tag_before_do(pdf):
    """The marked-content tag wrapping the (first) image Do, or None if bare."""
    cs = list(pikepdf.parse_content_stream(pdf.pages[0]))
    do_idx = next(i for i, ins in enumerate(cs) if str(ins.operator) == "Do")
    for j in range(do_idx - 1, -1, -1):
        op = str(cs[j].operator)
        if op in ("BDC", "BMC"):
            return str(cs[j].operands[0])
        if op == "EMC":
            return None  # image sits after a closed sequence → bare
    return None


# ----------------------------------------------------------------- unit math ---

def test_affine_mul_identity():
    I = (1, 0, 0, 1, 0, 0)
    m = (100, 0, 0, 100, 50, 50)
    assert _affine_mul(m, I) == m
    assert _affine_mul(I, m) == m


def test_image_rect_std_maps_unit_square():
    # cm = scale 100 + translate (50,50); page 200pt tall. Image covers PDF
    # (50,50)-(150,150) -> standard 150-DPI top-left via the 150/72 scale.
    rect = _image_rect_std((100, 0, 0, 100, 50, 50), 200.0)
    s = 150.0 / 72.0
    assert rect[0] == pytest.approx(50 * s)
    assert rect[2] == pytest.approx(150 * s)
    # y flips: top = (200 - 150)*s, bottom = (200 - 50)*s
    assert rect[1] == pytest.approx((200 - 150) * s)
    assert rect[3] == pytest.approx((200 - 50) * s)


# --------------------------------------------------------------- end-to-end ---

def test_image_in_figure_becomes_figure(tmp_path):
    src, out = tmp_path / "in.pdf", tmp_path / "out.pdf"
    _make_pdf_with_image(src, cm=(100, 0, 0, 100, 50, 50))  # std rect ~ (104,104,312,312)
    tag_untagged_pdf(str(src), str(out), [_figure((100, 100, 320, 320))], total_pages=1)

    pdf = pikepdf.open(str(out))
    figs = [n for n in _struct_tags(pdf) if str(n.get("/S")) == "/Figure"]
    assert len(figs) == 1, "expected one /Figure struct element"
    assert figs[0].get("/Alt") is not None, "Figure must carry /Alt"
    assert _bdc_tag_before_do(pdf) == "/Figure", "image must be /Figure marked content"
    pdf.close()


def test_image_outside_figure_stays_artifact(tmp_path):
    src, out = tmp_path / "in.pdf", tmp_path / "out.pdf"
    _make_pdf_with_image(src, cm=(20, 0, 0, 20, 5, 5))  # tiny image, far from figure
    # Figure region elsewhere on the page; the image does not fall inside it.
    tag_untagged_pdf(str(src), str(out), [_figure((300, 300, 400, 400))], total_pages=1)

    pdf = pikepdf.open(str(out))
    figs = [n for n in _struct_tags(pdf) if str(n.get("/S")) == "/Figure"]
    assert figs == [], "no image inside the figure region → no /Figure struct elem"
    assert _bdc_tag_before_do(pdf) == "/Artifact", "unmatched image stays Artifact"
    pdf.close()
