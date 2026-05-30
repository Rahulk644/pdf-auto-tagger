"""rapid_table (SLANet/PP-Structure) extractor — HTML->cells parsing (no model)."""
from tagger.stage5_specialists.slanet_table_extractor import _parse_cells_in_order


def test_parse_cells_grid_and_header():
    html = ("<html><body><table>"
            "<tr><td>Year</td><td>Value</td></tr>"
            "<tr><td>2012</td><td>10%</td></tr>"
            "<tr><td>2013</td><td>6%</td></tr>"
            "</table></body></html>")
    cells = _parse_cells_in_order(html)
    assert len(cells) == 6
    # cells are in row-major order (aligned with rapid_table cell_bboxes)
    assert [(c["row_idx"], c["col_idx"]) for c in cells] == [
        (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
    # row 0 is the header row (so the table always exposes TH for AT)
    assert all(c["is_header"] for c in cells if c["row_idx"] == 0)
    assert not any(c["is_header"] for c in cells if c["row_idx"] > 0)
    # text/merged_from are filled later from NATIVE chars; ocr_text is the fallback
    assert all(c["text"] == "" and c["merged_from"] == [] for c in cells)
    assert next(c for c in cells if c["row_idx"] == 1 and c["col_idx"] == 0)["ocr_text"] == "2012"


def test_th_respected_when_present():
    html = "<table><tr><th>H</th></tr><tr><td>x</td></tr></table>"
    cells = _parse_cells_in_order(html)
    assert cells[0]["is_header"] and cells[0]["ocr_text"] == "H"


def test_engine_flag_default_is_tableformer():
    from tagger.config import TABLE
    # default = TableFormer: it won END-TO-END (0.567/0.721 vs PP-Structure
    # 0.552/0.717) even though PP-Structure won the misleading isolated-crop bench.
    import os
    if "TAGGER_TABLE_ENGINE" not in os.environ:
        assert TABLE.engine == "tableformer"
