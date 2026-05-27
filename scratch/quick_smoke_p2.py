import pdfplumber

def _is_numeric_content(text: str) -> bool:
    if not text: return True
    cleaned = text.strip().lstrip("$").replace(",", "").replace("(", "").replace(")", "").replace("%", "").replace("-", "").strip()
    return not cleaned or cleaned.replace(".", "").isdigit()

with pdfplumber.open("miramar_untagged.pdf") as pdf:
    for page_num in [1, 2, 3]:
        if page_num > len(pdf.pages):
            continue
        page = pdf.pages[page_num - 1]
        
        table_settings = {"vertical_strategy": "text", "horizontal_strategy": "text"}
        tables = page.find_tables(table_settings=table_settings)
        print(f"\n--- Page {page_num}: {len(tables)} tables ---")
        for i, t in enumerate(tables):
            rows = t.extract()
            has_header = False
            if rows:
                has_header = all(cell is not None and str(cell).strip() for cell in rows[0])
            
            row_headers = 0
            for row_idx, row in enumerate(t.rows):
                is_header_row = (has_header and row_idx == 0)
                for col_idx, cell_bbox in enumerate(row.cells):
                    cell_text_val = rows[row_idx][col_idx] if row_idx < len(rows) and col_idx < len(rows[row_idx]) else None
                    cell_text = str(cell_text_val).strip() if cell_text_val is not None else ""
                    
                    is_row_header = (
                        col_idx == 0
                        and not is_header_row
                        and bool(cell_text)
                        and not _is_numeric_content(cell_text)
                    )
                    if is_row_header:
                        row_headers += 1
            print(f"Table {i} has {row_headers} row headers detected.")
