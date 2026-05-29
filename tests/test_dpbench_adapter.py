"""dp-bench adapter: tagged-PDF struct tree -> GT-convention markdown.

Builds a synthetic tagged PDF via the real Stage-10 writer (tag_untagged_pdf) so
the adapter is exercised against genuine pipeline output (struct tree + /ActualText),
then asserts the markdown conventions: heading -> '# ', list item -> '- ', table ->
'<table>', paragraph -> plain text, in reading order; figures/artifacts skipped.
"""
import pikepdf

from tagger.models.data_types import PDFTag, TaggedElement
from tagger.stage10_writeback.struct_tree_writer import tag_untagged_pdf
from tagger.benchmark.dpbench.adapter import pdf_to_markdown


def _mf(a, b):
    return [f"p1_c{i}" for i in range(a, b)]


def test_adapter_emits_gt_conventions(tmp_path):
    pdf = pikepdf.new()
    page = pdf.add_blank_page(page_size=(612, 792))
    page.obj["/Contents"] = pdf.make_stream(
        b"BT /F1 12 Tf 72 740 Td (Title) Tj ET\n"        # 0-4  heading
        b"BT 72 700 Td (Body text here) Tj ET\n"          # 5-18 paragraph
        b"BT 72 660 Td (one) Tj ET\n"                     # 19-21 list item 1
        b"BT 72 640 Td (two) Tj ET\n"                     # 22-24 list item 2
        b"BT 72 600 Td (Hdr) Tj ET\n"                     # 25-27 table header cell
        b"BT 72 580 Td (Val) Tj ET\n"                     # 28-30 table data cell
    )
    in_path = tmp_path / "in.pdf"
    pdf.save(str(in_path)); pdf.close()

    els = [
        TaggedElement(element_id="h", page_num=1, pdf_tag=PDFTag.H1, text="Title",
                      bbox=(72, 40, 300, 60), merged_from=_mf(0, 5)),
        TaggedElement(element_id="p", page_num=1, pdf_tag=PDFTag.P, text="Body text here",
                      bbox=(72, 80, 300, 100), merged_from=_mf(5, 19)),
        TaggedElement(element_id="li1", page_num=1, pdf_tag=PDFTag.LI, text="one",
                      bbox=(72, 120, 300, 140), merged_from=_mf(19, 22),
                      specialist_data={"list_label": "1.", "list_body": "one"}),
        TaggedElement(element_id="li2", page_num=1, pdf_tag=PDFTag.LI, text="two",
                      bbox=(72, 150, 300, 170), merged_from=_mf(22, 25),
                      specialist_data={"list_label": "2.", "list_body": "two"}),
        TaggedElement(element_id="t", page_num=1, pdf_tag=PDFTag.TABLE, text="Hdr Val",
                      bbox=(72, 190, 300, 240),
                      specialist_data={"cells": [
                          {"row_idx": 0, "col_idx": 0, "text": "Hdr", "is_header": True,
                           "merged_from": _mf(25, 28)},
                          {"row_idx": 1, "col_idx": 0, "text": "Val", "is_header": False,
                           "merged_from": _mf(28, 31)},
                      ]}),
    ]
    out_path = tmp_path / "out.pdf"
    tag_untagged_pdf(str(in_path), str(out_path), els, total_pages=1)

    md = pdf_to_markdown(str(out_path))
    lines = [ln for ln in md.split("\n\n") if ln.strip()]

    assert "# Title" in md
    assert "Body text here" in md
    assert "- one" in md and "- two" in md
    # table present with both cell texts in a 2-row grid (th-vs-td is pipeline's
    # call and TEDS-neutral, so don't assert the cell tag)
    assert "<table>" in md and md.count("<tr>") == 2
    assert ">Hdr<" in md and ">Val<" in md
    # reading order preserved: heading before body before list before table
    assert md.index("# Title") < md.index("Body text here") < md.index("- one") < md.index("<table>")


def test_adapter_untagged_returns_empty(tmp_path):
    pdf = pikepdf.new()
    pdf.add_blank_page(page_size=(612, 792))
    p = tmp_path / "u.pdf"
    pdf.save(str(p)); pdf.close()
    assert pdf_to_markdown(str(p)) == ""
