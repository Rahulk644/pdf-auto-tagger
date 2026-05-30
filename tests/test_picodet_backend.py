"""Picodet layout backend — contract tests that do NOT load the model.

The PP-DocLayout-V3 weights aren't fetched in CI, so these assert the drop-in
contract (Heron-compatible label vocabulary, graceful no-op without weights,
backend dispatch) rather than detection quality. The quality/speed A/B vs Heron
lives in scratch/run_picodet_ab.py; verdict recorded in project memory
(picodet lost the MHS gate + ran slower → not default).
"""
from tagger.stage3_layout import picodet_layout as pdl


def test_label_map_targets_are_heron_vocabulary():
    """Every PP-DocLayout label must map to a string the CPU detector's
    _HERON_LABEL_TO_CATEGORY knows — otherwise regions silently fall to TEXT."""
    from tagger.stage3_layout.cpu_layout_detector import _HERON_LABEL_TO_CATEGORY
    valid = set(_HERON_LABEL_TO_CATEGORY)
    for raw, heron in pdl._PP_TO_HERON.items():
        assert heron in valid, f"{raw!r} -> {heron!r} not in Heron vocabulary"


def test_heading_labels_are_reachable():
    """The headings the merge step unions (Title / Section-header) must be
    producible, or the picodet path can never contribute headings."""
    targets = set(pdl._PP_TO_HERON.values())
    assert "Title" in targets
    assert "Section-header" in targets
    assert "Table" in targets


def test_detect_is_graceful_noop_without_weights(monkeypatch):
    """If the model can't load, both public functions return [] (Stage 3 then
    degrades to the pdfplumber-only path) rather than raising."""
    monkeypatch.setattr(pdl, "_load", lambda: False)
    assert pdl.detect_all_regions("/nonexistent.pdf", 1) == []
    assert pdl.detect_tables("/nonexistent.pdf", 1) == []


def test_backend_dispatch_routes_to_picodet(monkeypatch):
    """cpu_layout_detector._region_detect_all routes to picodet when the active
    backend is 'picodet', and to Heron otherwise."""
    import types
    import tagger.stage3_layout.cpu_layout_detector as cld

    sentinel = [((0, 0, 1, 1), "Title")]
    monkeypatch.setattr(pdl, "detect_all_regions", lambda p, n: sentinel)
    # LAYOUT is a frozen dataclass — swap the module-level reference, not its attr.
    monkeypatch.setattr(cld, "LAYOUT", types.SimpleNamespace(backend="picodet"))
    assert cld._region_detect_all("x.pdf", 1) is sentinel
