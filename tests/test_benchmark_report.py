"""Unit 6 — scorecard aggregation + formatting."""
from tagger.benchmark.harness import CHECKER, REMEDIATION, Result
from tagger.benchmark.loader import DocTask
from tagger.benchmark.report import build_scorecard, format_scorecard
from tagger.benchmark.verdicts.base import Verdict


def _r(framing, crit, label, status, adobe6=None):
    t = DocTask(openalex_id="x", criterion=crit, expert_label=label,
                pdf_path="x.pdf", is_tagged=True, adobe6_compliance=adobe6)
    return Result(t, framing, Verdict(status))


def test_scorecard_checker_and_labels():
    checker = [
        _r(CHECKER, "semantic_tagging", "passed", "passed"),
        _r(CHECKER, "semantic_tagging", "failed", "failed"),
        _r(CHECKER, "semantic_tagging", "not_present", "cannot_derive"),
        _r(CHECKER, "semantic_tagging", "cannot_tell", "passed"),
    ]
    sc = build_scorecard(checker)
    e = sc["criteria"]["semantic_tagging"]
    assert e["checker_agreement"]["agreement"] == 1.0   # 2/2 comparable agree
    assert e["labels"] == {"total": 4, "addressed": 2, "attempted": 0,
                           "not_present": 1, "cannot_tell": 1}


def test_scorecard_remediation_rate_included():
    checker = [_r(CHECKER, "table_structure", "failed", "failed")]
    remediation = [
        _r(REMEDIATION, "table_structure", "failed", "passed"),  # remediated
        _r(REMEDIATION, "table_structure", "failed", "failed"),  # not
    ]
    sc = build_scorecard(checker, remediation)
    assert sc["criteria"]["table_structure"]["remediation_rate"]["rate"] == 0.5


def test_scorecard_adobe_triangulation():
    checker = [
        # expert passed; we say passed (agree); adobe says failed (disagree)
        _r(CHECKER, "semantic_tagging", "passed", "passed", adobe6=False),
        # expert failed; we say failed (agree); adobe says failed (agree)
        _r(CHECKER, "semantic_tagging", "failed", "failed", adobe6=False),
    ]
    ad = build_scorecard(checker)["criteria"]["semantic_tagging"]["adobe6"]
    assert ad == {"comparable": 2, "ours_agree": 2, "adobe6_agree": 1}


def test_format_scorecard_renders():
    checker = [_r(CHECKER, "semantic_tagging", "passed", "passed", adobe6=True)]
    out = format_scorecard(build_scorecard(checker))
    assert "scorecard" in out and "semantic_tagging" in out and "axes:" in out
