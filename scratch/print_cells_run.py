import pickle
from tagger.stage5_specialists.table_extractor import _is_numeric_content

cell_text = ""
merged_from = ["p1_c24"]
is_numeric = _is_numeric_content(cell_text) if cell_text else (not bool(merged_from))
print(f"is_numeric: {is_numeric}")
is_row_header = (
    0 == 0
    and not False
    and (bool(cell_text) or bool(merged_from))
    and not is_numeric
)
print(f"is_row_header: {is_row_header}")
