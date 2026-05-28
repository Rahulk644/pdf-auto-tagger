"""Tests for the repair-gating system (detect / classify / gate / repair).

Two layers:
  1. Pure gate control-flow on synthetic Findings (no PDF, always runs).
  2. Detector + apply coherence on PREP PDFs as real defect fixtures (skipped if
     the fixtures are absent). PREP's tagged outputs still carry the inherited
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


# ------------------------------------------------- detectors on PREP fixtures ---

_PREP = Path("/Users/rahulkhatri/Downloads/pdf_tag_tool/PREP PDF & Reports/PREP PDFs")
_OSTEO = _PREP / "Osteoarthritis.pdf"
_MISSOURI = _PREP / "Missouri State Epidemiological Profile July 2018.pdf"
_skip = pytest.mark.skipif(not _PREP.exists(), reason="PREP fixtures not present")


@_skip
def test_detect_cidsets_finds_missouri_defects():
    with pikepdf.open(str(_MISSOURI)) as pdf:
        fs = detect_cidsets(pdf)
    assert fs, "expected CIDSet findings on PREP Missouri"
    assert all(f.clause == "7.21.4.2" and f.repair_type == MODIFYING for f in fs)


@_skip
def test_detect_then_auto_apply_is_idempotent():
    """auto applies what detect finds; re-detect then finds nothing (coherence)."""
    with pikepdf.open(str(_OSTEO)) as pdf:
        nd = detect_notdef_refs(pdf)
        sp = detect_missing_space_refs(pdf)
        assert nd, "expected .notdef findings on PREP Osteo"
        assert sp, "expected missing-space findings on PREP Osteo"
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
