"""Tests for the repair-gating system (detect / classify / gate / repair).

Two layers:
  1. Pure gate control-flow on synthetic Findings (no PDF, always runs).
  2. Detector + apply coherence on incumbent PDFs as real defect fixtures (skipped if
     the fixtures are absent). the incumbent's tagged outputs still carry the inherited
     font defects, so they make ideal MinerU-free fixtures.
"""

from pathlib import Path

import pikepdf
import pytest

from tagger.stage10_writeback.content_stream_writer import (
    detect_cidsets,
    detect_missing_space_refs,
    detect_notdef_refs,
    detect_unembedded_fonts,
)
from tagger.stage10_writeback.repair_gate import (
    ADDITIVE,
    AUTO,
    CONFIRM,
    FLAG_ONLY,
    MODIFYING,
    Finding,
    build_report,
    gate_and_apply,
)


def _modifying(applied, clause, location):
    return Finding(
        clause=clause,
        location=location,
        defect_description="d",
        proposed_repair="r",
        repair_type=MODIFYING,
        apply=lambda: applied.append(location),
    )


def _additive(applied):
    return Finding(
        clause="additive",
        location="struct-tree",
        defect_description="d",
        proposed_repair="r",
        repair_type=ADDITIVE,
        apply=lambda: applied.append("ADD"),
    )


def _report_only(clause, location):
    """A modifying finding with no automatic apply (e.g. unembedded font)."""
    return Finding(
        clause=clause, location=location, defect_description="d",
        proposed_repair="r", repair_type=MODIFYING, auto_safe=False, apply=None,
    )


def _nonsafe(applied, clause, location):
    """A modifying finding that alters rendering (auto_safe=False) but is applicable."""
    return Finding(
        clause=clause, location=location, defect_description="d",
        proposed_repair="r", repair_type=MODIFYING, auto_safe=False,
        apply=lambda: applied.append(location),
    )


# ---------------------------------------------------------------- pure gate ---

def test_auto_applies_all_modifying_and_additive():
    applied = []
    fs = [_modifying(applied, "7.21.4.2", "fontA"),
          _modifying(applied, "7.21.8", "p1"),
          _additive(applied)]
    gate_and_apply(fs, AUTO)
    assert set(applied) == {"fontA", "p1", "ADD"}
    assert all(f.status == "applied" for f in fs)


def test_flag_only_applies_no_modifying_but_additive_runs():
    applied = []
    m = _modifying(applied, "7.21.8", "p1")
    a = _additive(applied)
    gate_and_apply([m, a], FLAG_ONLY)
    assert applied == ["ADD"]            # additive ran, modifying did not
    assert m.status == "reported"
    assert a.status == "applied"


def test_confirm_empty_approval_applies_no_modifying():
    applied = []
    m = _modifying(applied, "7.21.8", "p1")
    a = _additive(applied)
    gate_and_apply([m, a], CONFIRM, approved_ids=set())
    assert applied == ["ADD"]            # additive always; modifying pending
    assert m.status == "pending"


def test_confirm_partial_approval_applies_exact_subset():
    applied = []
    m1 = _modifying(applied, "7.21.4.2", "fontA")
    m2 = _modifying(applied, "7.21.8", "p1")
    gate_and_apply([m1, m2], CONFIRM, approved_ids={m1.finding_id})
    assert applied == ["fontA"]
    assert m1.status == "applied"
    assert m2.status == "pending"


def test_report_lists_modifying_with_status():
    applied = []
    m = _modifying(applied, "7.21.8", "p1")
    gate_and_apply([m, _additive(applied)], FLAG_ONLY)
    rep = build_report([m], FLAG_ONLY)
    assert rep["repair_mode"] == "flag-only"
    assert rep["summary"]["modifying_total"] == 1
    assert rep["summary"]["reported"] == 1
    assert rep["findings"][0]["clause"] == "7.21.8"
    assert rep["findings"][0]["status"] == "reported"


def test_finding_id_stable_and_addressable():
    a = Finding("7.21.8", "page 1", "d", "r", MODIFYING)
    b = Finding("7.21.8", "page 1", "d", "r", MODIFYING)
    assert a.finding_id == b.finding_id            # stable hash of clause+location
    c = Finding("7.21.8", "page 2", "d", "r", MODIFYING)
    assert c.finding_id != a.finding_id


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        gate_and_apply([], "bogus")


def test_report_only_modifying_never_applied_even_in_auto():
    """A report-only finding (no apply) is surfaced, never silently applied."""
    applied = []
    r = _report_only("7.21.4.1", "font Arial")
    gate_and_apply([r, _additive(applied)], AUTO)
    assert r.status == "reported"
    assert applied == ["ADD"]  # only the additive ran


def test_non_auto_safe_modifying_pending_in_auto():
    """auto holds a rendering-altering repair pending; it never auto-applies."""
    applied = []
    m = _nonsafe(applied, "7.21.4.1", "font Arial")
    gate_and_apply([m], AUTO)
    assert m.status == "pending"
    assert applied == []


def test_non_auto_safe_modifying_applies_when_confirmed():
    """Explicit approval in confirm mode overrides auto_safe and applies it."""
    applied = []
    m = _nonsafe(applied, "7.21.4.1", "font Arial")
    gate_and_apply([m], CONFIRM, approved_ids={m.finding_id})
    assert m.status == "applied"
    assert applied == ["font Arial"]


def test_detect_unembedded_fonts_synthetic():
    """A font whose descriptor lacks any FontFile is a report-only modifying find."""
    pdf = pikepdf.new()
    pdf.add_blank_page(page_size=(200, 200))
    page = pdf.pages[0]
    fd = pdf.make_indirect(pikepdf.Dictionary({
        "/Type": pikepdf.Name.FontDescriptor, "/FontName": pikepdf.Name("/Arial"),
    }))
    bad = pdf.make_indirect(pikepdf.Dictionary({
        "/Type": pikepdf.Name.Font, "/Subtype": pikepdf.Name.TrueType,
        "/BaseFont": pikepdf.Name("/Arial"), "/FontDescriptor": fd,
    }))
    fd2 = pdf.make_indirect(pikepdf.Dictionary({
        "/Type": pikepdf.Name.FontDescriptor, "/FontName": pikepdf.Name("/Emb"),
        "/FontFile2": pdf.make_stream(b"\x00"),
    }))
    good = pdf.make_indirect(pikepdf.Dictionary({
        "/Type": pikepdf.Name.Font, "/Subtype": pikepdf.Name.TrueType,
        "/BaseFont": pikepdf.Name("/Emb"), "/FontDescriptor": fd2,
    }))
    page.obj["/Resources"] = pikepdf.Dictionary({
        "/Font": pikepdf.Dictionary({"/F1": bad, "/F2": good})
    })

    fs = detect_unembedded_fonts(pdf)
    assert len(fs) == 1
    f = fs[0]
    assert f.clause == "7.21.4.1" and f.repair_type == MODIFYING
    assert f.auto_safe is False and f.apply is None
    # routed through the gate, it is surfaced but not applied
    gate_and_apply(fs, AUTO)
    assert f.status == "reported"


# --- detectors on real font-defect fixtures (in-repo, neutral-named, CI-runnable) ---

_FX = Path(__file__).parent / "fixtures" / "font_defects"
_OSTEO = _FX / "notdef_space.pdf"      # has .notdef + missing-space refs
_MISSOURI = _FX / "cidset.pdf"          # has incomplete CIDSets
_skip = pytest.mark.skipif(not _FX.exists(), reason="font-defect fixtures not present")


@_skip
def test_detect_cidsets_finds_defects():
    with pikepdf.open(str(_MISSOURI)) as pdf:
        fs = detect_cidsets(pdf)
    assert fs, "expected CIDSet findings on the cidset fixture"
    assert all(f.clause == "7.21.4.2" and f.repair_type == MODIFYING for f in fs)


@_skip
def test_detect_then_auto_apply_is_idempotent():
    """auto applies what detect finds; re-detect then finds nothing (coherence)."""
    with pikepdf.open(str(_OSTEO)) as pdf:
        nd = detect_notdef_refs(pdf)
        sp = detect_missing_space_refs(pdf)
        assert nd, "expected .notdef findings on the incumbent Osteo"
        assert sp, "expected missing-space findings on the incumbent Osteo"
        gate_and_apply(nd + sp, AUTO)
        assert detect_notdef_refs(pdf) == []
        assert detect_missing_space_refs(pdf) == []


@_skip
def test_flag_only_leaves_defects_and_report_predicts_them():
    """flag-only applies nothing; defects remain; report names every one."""
    with pikepdf.open(str(_OSTEO)) as pdf:
        fs = detect_notdef_refs(pdf) + detect_missing_space_refs(pdf)
        clauses_found = {f.clause for f in fs}
        gate_and_apply(fs, FLAG_ONLY)
        assert all(f.status == "reported" for f in fs)
        # nothing was applied -> defects still detectable
        assert detect_notdef_refs(pdf)
        assert detect_missing_space_refs(pdf)
        rep = build_report(fs, FLAG_ONLY)
        assert {x["clause"] for x in rep["findings"]} == clauses_found
        assert {"7.21.8", "7.21.4.1"} <= clauses_found  # matches veraPDF baseline


# ----------------------------------------- end-to-end gate via tag_untagged_pdf ---

import json

from pikepdf import Dictionary, Name

from tagger.stage10_writeback.struct_tree_writer import tag_untagged_pdf


def _pdf_with_unembedded_font(path):
    """1-page PDF whose only font (Arial) has a descriptor but no FontFile."""
    pdf = pikepdf.new()
    pdf.add_blank_page(page_size=(200, 200))
    page = pdf.pages[0]
    fd = pdf.make_indirect(Dictionary({
        "/Type": Name.FontDescriptor, "/FontName": Name("/Arial"), "/Flags": 32,
    }))
    font = pdf.make_indirect(Dictionary({
        "/Type": Name.Font, "/Subtype": Name.TrueType,
        "/BaseFont": Name("/Arial"), "/FontDescriptor": fd,
    }))
    page.obj["/Resources"] = Dictionary({"/Font": Dictionary({"/F1": font})})
    page.obj["/Contents"] = pdf.make_stream(b"")
    pdf.save(str(path))
    pdf.close()


def _font_embedded(path):
    with pikepdf.open(str(path)) as pdf:
        for page in pdf.pages:
            fonts = (page.obj.get("/Resources") or {}).get("/Font") or {}
            for _n, f in fonts.items():
                fd = f.get("/FontDescriptor")
                if fd is not None and any(
                    fd.get(k) is not None for k in ("/FontFile", "/FontFile2", "/FontFile3")
                ):
                    return True
    return False


@pytest.mark.parametrize("mode", [AUTO, CONFIRM, FLAG_ONLY])
def test_unembedded_font_reported_never_embedded_e2e(tmp_path, mode):
    """The font-embedding repair (added after the gate existed) flows through the
    real Stage-10 gate as report-only: surfaced in every mode, never embedded."""
    src, out = tmp_path / "in.pdf", tmp_path / "out.pdf"
    _pdf_with_unembedded_font(src)
    tag_untagged_pdf(str(src), str(out), [], total_pages=1,
                     repair_mode=mode, approved_ids=set())

    report = json.loads((out.with_suffix(".repairs.json")).read_text())
    fonts = [f for f in report["findings"] if f["clause"] == "7.21.4.1"]
    assert len(fonts) == 1, "unembedded-font finding must appear in the report"
    assert fonts[0]["status"] == "reported", f"must be report-only in {mode}"
    assert fonts[0]["repair_type"] == MODIFYING and fonts[0]["auto_safe"] is False
    # The gate must NEVER silently embed a font, in any mode.
    assert not _font_embedded(out), f"font must stay unembedded in {mode}"


def test_report_mode_recorded_in_repairs_json(tmp_path):
    """The repairs.json records the mode it ran under (audit trail)."""
    src, out = tmp_path / "in.pdf", tmp_path / "out.pdf"
    _pdf_with_unembedded_font(src)
    tag_untagged_pdf(str(src), str(out), [], total_pages=1, repair_mode=FLAG_ONLY)
    report = json.loads((out.with_suffix(".repairs.json")).read_text())
    assert report["repair_mode"] == FLAG_ONLY
    assert report["summary"]["reported"] >= 1
    assert report["summary"]["applied"] == 0
