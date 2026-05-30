"""Matterhorn 1.1 labelling layer over the ACT/PDF-UA audit."""
import os

import pytest

from tagger.audit.act_rules import audit_pdf
from tagger.audit.matterhorn import RULE_TO_MATTERHORN, to_matterhorn

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


def test_every_audit_rule_has_a_matterhorn_mapping(tmp_path):
    """Guard: if a new ACT/PDF-UA rule is added to act_rules, it must also get a
    Matterhorn mapping here (so the reporting layer stays complete)."""
    rep = audit_pdf(_tagged(tmp_path))
    audited = {r.rule_id for r in rep.results if r.rule_id != "io"}
    mapped = set(RULE_TO_MATTERHORN)
    assert audited <= mapped, f"unmapped audit rules: {audited - mapped}"


def test_status_propagates_to_failure_conditions(tmp_path):
    rep = audit_pdf(_tagged(tmp_path))
    rows = to_matterhorn(rep)
    assert rows, "expected Matterhorn rows"
    # Our own output should have no Matterhorn FAIL rows (Stage-8 enforcers).
    fails = [m for m in rows if m.status == "fail"]
    assert not fails, f"unexpected Matterhorn failures: {[(m.fc_id, m.source_rule) for m in fails]}"
    # Verified IDs we depend on are present.
    ids = {m.fc_id for m in rows}
    assert {"13-004", "11-001", "06-003", "07-001", "01-005"} <= ids
