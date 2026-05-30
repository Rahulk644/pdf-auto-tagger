"""Type-routed, OCR-grounded alt text: data-bearing figures name their labels
without ever claiming data values (the small-VLM hallucination failure mode)."""
from tagger.stage9_alttext.siglip_figure_classifier import (
    bucket_to_alt_text, figure_labels, _DATA_BEARING, _MAX_ALT_LEN,
)


def test_data_bearing_bucket_uses_labels_value_safely():
    alt = bucket_to_alt_text("chart", 0.9, labels=["Muon Energy Proxy", "log Event Weight"])
    assert "Muon Energy Proxy" in alt and "log Event Weight" in alt
    # value-safe: explicitly disclaims data values, never asserts numbers
    assert "Data values not detailed" in alt
    assert len(alt) <= _MAX_ALT_LEN


def test_data_bearing_without_labels_falls_back():
    alt = bucket_to_alt_text("chart", 0.9, labels=[])
    assert alt == "Chart. Refer to long description."


def test_non_data_bucket_ignores_labels():
    # A photograph has no axis labels to ground to — labels must not leak in.
    assert bucket_to_alt_text("photograph", 0.9, labels=["spurious"]) == "Photograph."


def test_decorative_still_none():
    assert bucket_to_alt_text("decorative", 0.9, labels=["x"]) is None


def test_data_bearing_set_covers_complex_types():
    assert {"chart", "diagram", "schematic", "map"} <= set(_DATA_BEARING)


def test_figure_labels_graceful_on_none():
    assert figure_labels(None) == []
