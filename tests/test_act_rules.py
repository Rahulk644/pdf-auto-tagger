"""Smoke tests for the ACT-rules auditor on real outputs we already produce."""
import os
import pytest
from tagger.audit.act_rules import audit_pdf

# Use the synthetic-scanned doc we ship as a fixture; it goes through the
# pipeline end-to-end so the audit has all the catalog and struct elements
# to inspect.
FIXTURE = "tests/fixtures/mixed_native_scanned.pdf"


def _produce_tagged(tmp_path):
    """Run the pipeline on the mixed-pages fixture (already used by
    test_mixed_pages); audit the tagged output."""
    from tagger.config import LAYOUT
    if LAYOUT.backend != "cpu":
        pytest.skip("requires TAGGER_LAYOUT_BACKEND=cpu (no MinerU locally)")
    if not os.path.exists(FIXTURE):
        pytest.skip("fixture missing")
    from tagger.pipeline import AutoTaggerPipeline
    out = tmp_path / "tagged.pdf"
    AutoTaggerPipeline().run(
        input_pdf=FIXTURE, output_pdf=str(out), report_path=str(out.with_suffix(".json")))
    return str(out)


def test_auditor_returns_a_report_with_all_implemented_rules(tmp_path):
    rep = audit_pdf(_produce_tagged(tmp_path))
    rule_ids = {r.rule_id for r in rep.results}
    # All eight rules in act_rules.py should be present
    assert {"ACT-6cfa84", "ACT-36b590", "ACT-b40fd1",
            "PDFUA-7.4.2", "PDFUA-7.1-10", "PDFUA-7.5.2",
            "PDFUA-7.5.3", "PDFUA-7.1-1"} <= rule_ids


def test_auditor_zero_failures_on_our_own_output(tmp_path):
    """Audit our pipeline's output against itself — our Stage-8 enforcers
    should guarantee zero ACT-rule failures."""
    rep = audit_pdf(_produce_tagged(tmp_path))
    fails = [r for r in rep.results if r.status == "fail"]
    assert not fails, f"unexpected ACT failures on our own output: {[(r.rule_id, r.notes) for r in fails]}"


def test_auditor_handles_missing_file_gracefully(tmp_path):
    rep = audit_pdf(str(tmp_path / "does_not_exist.pdf"))
    assert any(r.status == "fail" and r.rule_id == "io" for r in rep.results)


# --- PDFUA-7.4.2 heading-hierarchy rules (validated against veraPDF-corpus 7.4.x) ---
# Synthesize a minimal tagged struct tree with given heading tags so we can exercise
# the failure modes the pipeline itself never produces (it emits valid hierarchies).
def _audit_headings(tmp_path, tags):
    import pikepdf
    from pikepdf import Dictionary, Name, Array, String
    pdf = pikepdf.new()
    pdf.add_blank_page()
    kids = [pdf.make_indirect(Dictionary(
        Type=Name.StructElem, S=Name("/" + t), ActualText=String("x"))) for t in tags]
    doc_el = pdf.make_indirect(Dictionary(Type=Name.StructElem, S=Name("/Document"),
                                          K=Array(kids)))
    pdf.Root.StructTreeRoot = pdf.make_indirect(
        Dictionary(Type=Name.StructTreeRoot, K=doc_el))
    pdf.Root.MarkInfo = Dictionary(Marked=True)
    p = tmp_path / "h.pdf"
    pdf.save(str(p))
    rep = audit_pdf(str(p))
    return next(r.status for r in rep.results if r.rule_id == "PDFUA-7.4.2")


def test_heading_valid_hierarchy_passes(tmp_path):
    assert _audit_headings(tmp_path, ["H1", "H2", "H3"]) == "pass"


def test_heading_level_skip_fails(tmp_path):
    assert _audit_headings(tmp_path, ["H1", "H2", "H4"]) == "fail"


def test_heading_first_not_h1_fails(tmp_path):
    assert _audit_headings(tmp_path, ["H2", "H3", "H4"]) == "fail"


def test_heading_mixed_numbered_unnumbered_fails(tmp_path):
    assert _audit_headings(tmp_path, ["H1", "H"]) == "fail"
