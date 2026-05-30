"""Semantic judge — deterministic perception extractors (the no-LLM core).

The LLM call (judge(), Llama on Groq) needs GROQ_API_KEY and is exercised via
the pilot runner, not in CI. Here we test the two independent views it reasons over.
"""
import os

import pytest

from tagger.audit.semantic_judge import (
    physical_layout, tag_view, judge, candidates, judge_candidates,
)
import tagger.audit.semantic_judge as sj

FIXTURE = "tests/fixtures/conformance/native_scholarly.pdf"


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


def test_physical_layout_is_independent_signal(tmp_path):
    rows = physical_layout(_tagged(tmp_path))
    assert rows, "expected physical layout lines"
    r = rows[0]
    assert {"page", "text", "rel_size", "bold", "pos"} <= set(r)
    # carries a real, varying physical signal (size and/or weight), independent
    # of any tag — not a constant the tagger could have driven.
    assert any(x["bold"] for x in rows) or len({x["rel_size"] for x in rows}) > 1


def test_tag_view_returns_roles(tmp_path):
    rows = tag_view(_tagged(tmp_path))
    assert rows and {"role", "text"} <= set(rows[0])
    assert any(r["role"].startswith("H") for r in rows)  # at least one heading tagged


def test_judge_requires_key(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        judge(_tagged(tmp_path))


def test_candidates_are_wellformed(tmp_path):
    """Deterministic candidate-generation (no LLM) returns physical-vs-tag
    mismatches as structured dicts."""
    cands = candidates(_tagged(tmp_path))
    assert isinstance(cands, list)
    for c in cands:
        assert {"type", "text", "physical", "tag"} <= set(c)
        assert c["type"] in ("possible_missed_heading", "possible_over_heading")


def test_judge_candidates_no_llm_when_empty(tmp_path, monkeypatch):
    """The precision path must NOT call the LLM (no key needed) when the
    deterministic step finds zero candidates."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(sj, "candidates", lambda p: [])
    out = judge_candidates(_tagged(tmp_path))
    assert out == {"candidates": 0, "verdicts": []}
