"""Benchmark loader: dataset.json -> DocTask iterator with /StructTreeRoot routing."""
import json

import pikepdf
from pikepdf import Array, Dictionary, Name

from tagger.benchmark.loader import DocTask, load_benchmark


def _tagged_pdf(path):
    pdf = pikepdf.new()
    pdf.add_blank_page(page_size=(200, 200))
    doc = pdf.make_indirect(Dictionary({"/S": Name("/Document"), "/K": Array([])}))
    pdf.Root["/StructTreeRoot"] = pdf.make_indirect(
        Dictionary({"/Type": Name.StructTreeRoot, "/K": doc}))
    pdf.save(str(path))
    pdf.close()


def _untagged_pdf(path):
    pdf = pikepdf.new()
    pdf.add_blank_page(page_size=(200, 200))
    pdf.save(str(path))
    pdf.close()


def _benchmark_root(tmp_path):
    """Minimal benchmark tree: one tagged (passed) + one untagged (failed) doc."""
    (tmp_path / "data" / "processed").mkdir(parents=True)
    _tagged_pdf(tmp_path / "data" / "processed" / "A.pdf")
    _untagged_pdf(tmp_path / "data" / "processed" / "B.pdf")
    dataset = {
        "name": "mini", "version": "1.0",
        "tasks": {
            "semantic_tagging": {
                "passed": [{"openalex_id": "A", "pdf_path": "data/processed/A.pdf",
                            "normalized_compliance": 0.5, "adobe6_compliance": True}],
                "failed": [{"openalex_id": "B", "pdf_path": "data/processed/B.pdf",
                            "normalized_compliance": 0.2, "adobe6_compliance": False}],
            },
            "table_structure": {
                "not_present": [{"openalex_id": "A", "pdf_path": "data/processed/A.pdf"}],
                "cannot_tell": [{"openalex_id": "B", "pdf_path": "data/processed/B.pdf"}],
            },
        },
    }
    (tmp_path / "data" / "dataset.json").write_text(json.dumps(dataset))
    return tmp_path


def test_loader_yields_all_doc_criterion_pairs(tmp_path):
    tasks = list(load_benchmark(_benchmark_root(tmp_path)))
    assert len(tasks) == 4  # 2 criteria x 2 labels
    keys = {(t.openalex_id, t.criterion, t.expert_label) for t in tasks}
    assert ("A", "semantic_tagging", "passed") in keys
    assert ("B", "semantic_tagging", "failed") in keys


def test_loader_routing_matches_struct_tree(tmp_path):
    tasks = {t.openalex_id: t for t in load_benchmark(_benchmark_root(tmp_path))
             if t.criterion == "semantic_tagging"}
    assert tasks["A"].is_tagged is True and tasks["A"].route == "V1"
    assert tasks["B"].is_tagged is False and tasks["B"].route == "V2"


def test_loader_keeps_np_ct_labels(tmp_path):
    labels = {(t.criterion, t.expert_label) for t in load_benchmark(_benchmark_root(tmp_path))}
    assert ("table_structure", "not_present") in labels
    assert ("table_structure", "cannot_tell") in labels


def test_loader_carries_reference_signals(tmp_path):
    a = next(t for t in load_benchmark(_benchmark_root(tmp_path))
             if t.openalex_id == "A" and t.criterion == "semantic_tagging")
    assert a.normalized_compliance == 0.5 and a.adobe6_compliance is True


def test_loader_records_missing_pdf_as_load_error(tmp_path):
    (tmp_path / "data").mkdir(parents=True)
    ds = {"tasks": {"semantic_tagging": {
        "passed": [{"openalex_id": "Z", "pdf_path": "data/processed/missing.pdf"}]}}}
    (tmp_path / "data" / "dataset.json").write_text(json.dumps(ds))
    t = next(iter(load_benchmark(tmp_path)))
    assert t.load_error is not None and t.is_tagged is False
