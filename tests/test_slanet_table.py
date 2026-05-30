"""SLANet table extractor — deterministic HTML->cells parsing (no model load)."""
from tagger.stage5_specialists.slanet_table_extractor import _parse_cells


def test_parse_cells_grid_and_header():
    html = ("<html><body><table>"
            "<tr><td>Year</td><td>Value</td></tr>"
            "<tr><td>2012</td><td>10%</td></tr>"
            "<tr><td>2013</td><td>6%</td></tr>"
            "</table></body></html>")
    cells = _parse_cells(html)
    assert len(cells) == 6
    # row/col indices positional
    assert {(c["row_idx"], c["col_idx"]) for c in cells} == {
        (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)}
    # row 0 is the header row (so the table always exposes TH)
    assert all(c["is_header"] for c in cells if c["row_idx"] == 0)
    assert not any(c["is_header"] for c in cells if c["row_idx"] > 0)
    # first column of body rows flagged as row-header
    assert any(c["is_row_header"] for c in cells if c["col_idx"] == 0 and c["row_idx"] > 0)
    # cells carry text but no MCID backing -> /ActualText path in Stage 10
    assert all(c["merged_from"] == [] for c in cells)
    assert next(c for c in cells if c["row_idx"] == 1 and c["col_idx"] == 0)["text"] == "2012"


def test_th_respected_when_present():
    html = "<table><tr><th>H</th></tr><tr><td>x</td></tr></table>"
    cells = _parse_cells(html)
    assert cells[0]["is_header"] and cells[0]["text"] == "H"


def test_engine_flag_default_is_tableformer():
    from tagger.config import TABLE
    # default must stay TableFormer (SLANet is opt-in until corpus-validated)
    import os
    if "TAGGER_TABLE_ENGINE" not in os.environ:
        assert TABLE.engine == "tableformer"
