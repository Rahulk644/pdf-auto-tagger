"""Table header-row detection (_is_header_row).

The old all-cells-non-empty rule classified financial/data header rows as TD
whenever the stub-head corner (top-left) was empty — the layout harness measured
TH recall at ~1%. These tests pin the relaxed rule: empty corner OK, numeric
column headers OK, but data/sparse rows are not headers.
"""
from tagger.stage5_specialists.table_extractor import _is_header_row


def test_empty_stub_head_corner_is_header():
    assert _is_header_row(["", "Governmental", "Business-Type"]) is True


def test_numeric_year_column_headers_are_header():
    # A "mostly non-numeric" rule would wrongly reject this; the corner rule keeps it.
    assert _is_header_row(["Item", "2017", "2016"]) is True


def test_fully_filled_first_row_still_header():
    assert _is_header_row(["A", "B", "C"]) is True


def test_all_empty_is_not_header():
    assert _is_header_row(["", "", ""]) is False


def test_single_filled_cell_is_not_header():
    assert _is_header_row(["", "", "X"]) is False


def test_empty_middle_cell_is_not_header():
    # Only the stub-head corner may be empty; an empty interior cell disqualifies.
    assert _is_header_row(["A", "", "C"]) is False


def test_single_column_is_not_header():
    assert _is_header_row(["X"]) is False


def test_empty_input_is_not_header():
    assert _is_header_row([]) is False
    assert _is_header_row(None) is False
