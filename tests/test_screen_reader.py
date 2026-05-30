"""Deterministic screen-reader linearizer (cross-platform AT simulation)."""
import os

import pikepdf
import pytest

from tagger.audit.screen_reader import linearize, smell_test

FIXTURE = "tests/fixtures/conformance/native_with_formulas.pdf"


def _tagged(tmp_path):
    from tagger.config import LAYOUT
    if LAYOUT.backend not in ("cpu", "picodet"):
        pytest.skip("requires a CPU layout backend")
    if not os.path.exists(FIXTURE):
        pytest.skip("fixture missing")
    from tagger.pipeline import AutoTaggerPipeline
    out = tmp_path / "t.pdf"
    AutoTaggerPipeline().run(input_pdf=FIXTURE, output_pdf=str(out),
                             report_path=str(out.with_suffix(".json")))
    return str(out)


def test_tagged_output_produces_clean_transcript(tmp_path):
    t = linearize(_tagged(tmp_path))
    assert t.announcements, "a tagged doc should yield announcements"
    # Artifacts must never be announced — a reader skips page furniture.
    assert all(a.role != "artifact" for a in t.announcements)
    # Our own output should hit no screen-reader issues.
    assert not t.issues, f"unexpected issues: {[(a.role, a.issue) for a in t.issues]}"


def test_untagged_pdf_is_flagged(tmp_path):
    """A document with no StructTreeRoot gives a screen-reader user nothing —
    the linearizer must say so rather than returning an empty-but-clean result."""
    p = tmp_path / "blank.pdf"
    pdf = pikepdf.new()
    pdf.add_blank_page(page_size=(612, 792))
    pdf.save(str(p))
    pdf.close()
    issues = smell_test(str(p))
    assert any("StructTreeRoot" in a.issue for a in issues)


def test_formula_announced_as_formula(tmp_path):
    t = linearize(_tagged(tmp_path))
    # native_with_formulas.pdf has display formulas → at least one formula
    # announcement, each carrying a text equivalent (no "no text equivalent" issue).
    formulas = [a for a in t.announcements if a.role == "formula"]
    assert formulas, "expected formula announcements"
    assert all(not a.issue for a in formulas)
