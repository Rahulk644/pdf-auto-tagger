import pdfplumber
from tagger.config import STANDARD_DPI, PDF_NATIVE_DPI

def _is_numeric_content(text: str) -> bool:
    if not text: return True
    cleaned = text.strip().lstrip("$").replace(",", "").replace("(", "").replace(")", "").replace("%", "").replace("-", "").strip()
    return not cleaned or cleaned.replace(".", "").isdigit()

with pdfplumber.open("miramar_untagged.pdf") as pdf:
    page = pdf.pages[0]
    # In 1000-normalized coords: [117, 433, 884, 774]
    # Page width=612, height=792 (72 DPI)
    # MinerU 1000-normalized -> 72 DPI directly:
    # x0 = 117 / 1000 * 612 = 71.6
    # y0 = 433 / 1000 * 792 = 342.9
    # x1 = 884 / 1000 * 612 = 541.0
    # y1 = 774 / 1000 * 792 = 613.0
    crop_box = (71.6 - 5, 342.9 - 5, 541.0 + 5, 613.0 + 5)
    
    cropped = page.within_bbox(crop_box)
    table_settings = {"vertical_strategy": "text", "horizontal_strategy": "text"}
    tables = cropped.find_tables(table_settings=table_settings)
    if not tables:
        print("No tables found in crop")
    else:
        t = tables[0]
        rows = t.extract()
        for row_idx, row in enumerate(t.rows):
            is_header_row = (row_idx == 0)
            for col_idx, cell_bbox in enumerate(row.cells):
                cell_text_val = rows[row_idx][col_idx] if row_idx < len(rows) and col_idx < len(rows[row_idx]) else None
                cell_text = str(cell_text_val).strip() if cell_text_val is not None else ""
                
                is_row_header = (
                    col_idx == 0
                    and not is_header_row
                    and bool(cell_text)
                    and not _is_numeric_content(cell_text)
                )
                if col_idx == 0:
                    print(f"R{row_idx} C{col_idx}: text={repr(cell_text)} row_header={is_row_header} numeric={_is_numeric_content(cell_text)}")
