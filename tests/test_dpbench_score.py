"""dp-bench score aggregation (our score.py — not covered by ODL's golden tests).

The metric parity itself is proven by tests/dpbench_golden/ (ODL's own cases).
These cover the per-doc overall = mean(nid, teds, mhs) and the corpus aggregation,
including the table-less-doc case (teds=None excluded from overall + means).
"""
from pytest import approx

from tagger.benchmark.dpbench.score import (
    DocumentScores, aggregate_scores, score_document,
)


def test_identical_doc_scores_perfect():
    md = "# Title\nSome body text.\n\n<table><tr><td>a</td></tr></table>"
    s = score_document("d1", md, md)
    assert s.nid == approx(1.0)
    assert s.teds == approx(1.0)
    assert s.mhs == approx(1.0)
    assert s.overall == approx(1.0)


def test_table_less_doc_overall_excludes_teds():
    md = "# Title\nJust prose, no table here at all."
    s = score_document("d2", md, md)
    assert s.teds is None                      # GT has no table
    assert s.nid == approx(1.0) and s.mhs == approx(1.0)
    assert s.overall == approx(1.0)            # mean of (nid, mhs) only


def test_aggregate_means_and_counts():
    docs = [
        DocumentScores("a", overall=1.0, nid=1.0, nid_s=1.0, teds=0.5, teds_s=0.5,
                       mhs=1.0, mhs_s=1.0),
        DocumentScores("b", overall=0.5, nid=0.6, nid_s=0.6, teds=None, teds_s=None,
                       mhs=0.4, mhs_s=0.4, prediction_available=False),
    ]
    agg = aggregate_scores(docs)
    assert agg["score"]["nid_mean"] == approx(0.8)
    assert agg["score"]["teds_mean"] == approx(0.5)     # only doc a has a table
    assert agg["teds_count"] == 1
    assert agg["score"]["mhs_mean"] == approx(0.7)
    assert agg["missing_predictions"] == 1
    assert agg["document_count"] == 2
