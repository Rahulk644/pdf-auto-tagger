"""dp-bench harness + report (end-to-end on synthetic tagged PDFs, CPU-only)."""
import pikepdf
from pytest import approx

from tagger.models.data_types import PDFTag, TaggedElement
from tagger.stage10_writeback.struct_tree_writer import tag_untagged_pdf
from tagger.benchmark.dpbench.harness import run_dpbench
from tagger.benchmark.dpbench.report import format_scorecard, load_published, PUBLISHED


def _make_pred_pdf(path, tmp_path):
    pdf = pikepdf.new()
    page = pdf.add_blank_page(page_size=(612, 792))
    page.obj["/Contents"] = pdf.make_stream(
        b"BT /F1 12 Tf 72 740 Td (Title) Tj ET\nBT 72 700 Td (Body text here) Tj ET\n"
    )
    src = tmp_path / "src.pdf"
    pdf.save(str(src)); pdf.close()
    els = [
        TaggedElement(element_id="h", page_num=1, pdf_tag=PDFTag.H1, text="Title",
                      bbox=(72, 40, 300, 60), merged_from=[f"p1_c{i}" for i in range(0, 5)]),
        TaggedElement(element_id="p", page_num=1, pdf_tag=PDFTag.P, text="Body text here",
                      bbox=(72, 80, 300, 100), merged_from=[f"p1_c{i}" for i in range(5, 19)]),
    ]
    tag_untagged_pdf(str(src), str(path), els, total_pages=1)


def test_harness_scores_matching_doc_and_missing_pred(tmp_path):
    gt_dir = tmp_path / "gt"; gt_dir.mkdir()
    pred_dir = tmp_path / "pred"; pred_dir.mkdir()

    # doc A: GT matches what the adapter emits -> near-perfect
    (gt_dir / "A.md").write_text("# Title\n\nBody text here", encoding="utf-8")
    _make_pred_pdf(pred_dir / "A.pdf", tmp_path)

    # doc B: GT exists, no prediction PDF -> missing, scored against empty
    (gt_dir / "B.md").write_text("# Heading\n\nsome other content", encoding="utf-8")

    res = run_dpbench(gt_dir, pred_dir)
    by_id = {d.document_id: d for d in res.documents}

    assert by_id["A"].nid == approx(1.0)
    assert by_id["A"].mhs == approx(1.0)
    assert by_id["A"].teds is None                  # no table in GT
    assert by_id["B"].prediction_available is False
    assert by_id["B"].mhs == approx(0.0)            # GT has heading, empty pred has none
    assert res.aggregate["document_count"] == 2
    assert res.aggregate["missing_predictions"] == 1


def test_harness_doc_id_filter(tmp_path):
    gt_dir = tmp_path / "gt"; gt_dir.mkdir()
    (gt_dir / "A.md").write_text("# T\n\nx", encoding="utf-8")
    (gt_dir / "B.md").write_text("# T\n\ny", encoding="utf-8")
    res = run_dpbench(gt_dir, tmp_path / "pred", doc_ids=["A"])
    assert [d.document_id for d in res.documents] == ["A"]


def test_scorecard_includes_our_row_and_references():
    agg = {"score": {"overall_mean": 0.80, "nid_mean": 0.91, "nid_s_mean": 0.92,
                     "teds_mean": 0.45, "teds_s_mean": 0.50, "mhs_mean": 0.74,
                     "mhs_s_mean": 0.80},
           "document_count": 7, "teds_count": 4, "mhs_count": 6, "missing_predictions": 0}
    card = format_scorecard(agg)
    assert "auto-tagger (V2)" in card
    assert "opendataloader-hybrid" in card and "opendataloader (Fast)" in card
    assert "0.450" in card                          # our teds rendered
    assert "n=7 docs" in card


def test_load_published_falls_back_without_repo(tmp_path):
    pub = load_published(tmp_path)                   # no prediction/ dir
    assert pub == PUBLISHED
