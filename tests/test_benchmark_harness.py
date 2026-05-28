"""Unit 5 — harness loop, the hard-assert (failed=input not agreement-target),
and the 5-doc integration test (verdicts must match the hand-derivation)."""
import json
from pathlib import Path

import pytest

from tagger.benchmark.harness import (
    CHECKER,
    REMEDIATION,
    Result,
    checker_agreement,
    remediation_rate,
    run_checker,
)
from tagger.benchmark.loader import DocTask, load_benchmark
from tagger.benchmark.verdicts.base import Verdict


def _result(framing, criterion, expert_label, status):
    t = DocTask(openalex_id="x", criterion=criterion, expert_label=expert_label,
                pdf_path="x.pdf", is_tagged=True)
    return Result(t, framing, Verdict(status))


# ----------------------------------------------------------------- hard assert ---

def test_remediation_rate_rejects_checker_results():
    rs = [_result(CHECKER, "semantic_tagging", "failed", "passed")]
    with pytest.raises(AssertionError):
        remediation_rate(rs, "semantic_tagging")


def test_checker_agreement_rejects_remediation_results():
    rs = [_result(REMEDIATION, "semantic_tagging", "failed", "passed")]
    with pytest.raises(AssertionError):
        checker_agreement(rs, "semantic_tagging")


def test_remediation_rate_uses_failed_label_only_as_selector():
    # passed-labeled docs are NOT in the denominator; only failed-labeled inputs count.
    rs = [
        _result(REMEDIATION, "semantic_tagging", "failed", "passed"),   # remediated
        _result(REMEDIATION, "semantic_tagging", "failed", "failed"),   # not remediated
        _result(REMEDIATION, "semantic_tagging", "passed", "failed"),   # NOT an input
    ]
    m = remediation_rate(rs, "semantic_tagging")
    assert m["failed_inputs"] == 2 and m["remediated"] == 1 and m["rate"] == 0.5


def test_checker_agreement_excludes_np_ct_and_cannot_derive():
    rs = [
        _result(CHECKER, "semantic_tagging", "passed", "passed"),       # agree
        _result(CHECKER, "semantic_tagging", "failed", "passed"),       # disagree
        _result(CHECKER, "semantic_tagging", "not_present", "passed"),  # excluded
        _result(CHECKER, "semantic_tagging", "failed", "cannot_derive"),  # excluded
    ]
    m = checker_agreement(rs, "semantic_tagging")
    assert m["comparable"] == 2 and m["agree"] == 1 and m["agreement"] == 0.5


# ------------------------------------------- 5-doc integration vs hand-derivation ---

_SAMPLE = Path("/tmp/pdfa_sample")
_skip = pytest.mark.skipif(not _SAMPLE.exists(), reason="benchmark sample not present")


@_skip
def test_checker_loop_matches_hand_derivation(tmp_path):
    # mini benchmark tree pointing at the hand-validated sample docs
    expected = {
        ("sem_passed", "semantic_tagging"): "passed",
        ("sem_failed", "semantic_tagging"): "failed",
        ("table_passed", "table_structure"): "passed",
        ("links_passed", "functional_hyperlinks"): "passed",
        ("ro_W04_passed", "logical_reading_order"): "passed",
        ("ro_W04_failed", "logical_reading_order"): "failed",
    }
    tasks_def = {}
    for (stem, crit), status in expected.items():
        tasks_def.setdefault(crit, {}).setdefault(
            "passed" if status == "passed" else "failed", []).append(
            {"openalex_id": stem, "pdf_path": f"{stem}.pdf"})
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "dataset.json").write_text(json.dumps({"tasks": tasks_def}))
    # symlink sample pdfs under the tmp root
    for stem, _ in expected:
        (tmp_path / f"{stem}.pdf").symlink_to(_SAMPLE / f"{stem}.pdf")

    results = run_checker(load_benchmark(tmp_path))
    by = {(r.task.openalex_id, r.task.criterion): r.our_verdict.status for r in results}
    for key, want in expected.items():
        assert by[key] == want, f"{key}: got {by[key]} want {want}"
