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
