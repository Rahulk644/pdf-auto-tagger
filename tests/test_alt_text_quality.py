"""Alt-text QUALITY checker — the deterministic floor from the McGowan guidelines."""
import pytest

from tagger.audit.alt_text_quality import (
    _APPEARANCE_PREFIX, _BARE_TYPE, _PLACEHOLDER, check_alt_quality)


def test_placeholder_alt_flagged():
    """Review-required placeholders are well-formed sentences that pass the format
    checks but convey nothing — the rubric must reject them (else placeholder alt
    scores 100% compliant, as the alt-quality scoreboard exposed)."""
    assert _PLACEHOLDER.search("Figure (description needed).")
    assert _PLACEHOLDER.search("[alt_text_placeholder] Figure needs descriptive alt text.")
    assert _PLACEHOLDER.search("Image to be described.")
    # real, informative alt must NOT trip it
    assert not _PLACEHOLDER.search("Bar chart of quarterly revenue rising from $2M to $5M.")
    assert not _PLACEHOLDER.search("Missouri Department of Education logo with a flame icon.")


def test_appearance_prefix_flagged():
    assert _APPEARANCE_PREFIX.match("Image of a cat on a mat")
    assert _APPEARANCE_PREFIX.match("A graphic showing revenue")
    assert _APPEARANCE_PREFIX.match("photograph of a bridge")
    # our value-safe template must NOT trip it
    assert not _APPEARANCE_PREFIX.match("Chart. Labelled: Muon Energy. Data values not detailed.")


def test_bare_type_flagged():
    assert _BARE_TYPE.match("Chart.")
    assert _BARE_TYPE.match("Figure")
    assert not _BARE_TYPE.match("Chart. Labelled: X, Y.")


def test_check_runs_on_tagged_fixture(tmp_path):
    from tagger.config import LAYOUT
    if LAYOUT.backend not in ("cpu", "picodet"):
        pytest.skip("requires a CPU layout backend")
    import os
    fx = "tests/fixtures/conformance/native_with_formulas.pdf"
    if not os.path.exists(fx):
        pytest.skip("fixture missing")
    from tagger.pipeline import AutoTaggerPipeline
    out = tmp_path / "t.pdf"
    AutoTaggerPipeline().run(input_pdf=fx, output_pdf=str(out),
                             report_path=str(out.with_suffix(".json")))
    rep = check_alt_quality(str(out))
    # Whatever figures exist, our own output must not trip the mechanical rules.
    bad = [i for i in rep.issues if i.rule in ("appearance_prefix", "bare_type", "redundant_with_caption")]
    assert not bad, f"alt-quality violations on our own output: {[(i.rule, i.alt) for i in bad]}"
