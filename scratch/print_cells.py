import pdfplumber
from tagger.config import STANDARD_DPI, PDF_NATIVE_DPI

with pdfplumber.open("miramar_untagged.pdf") as pdf:
    for page_num in [1, 2, 3]:
        if page_num > len(pdf.pages):
            continue
        page = pdf.pages[page_num - 1]
        
        table_settings = {"vertical_strategy": "text", "horizontal_strategy": "text"}
        tables = page.find_tables(table_settings=table_settings)
        if tables:
            t = tables[0]
            rows = t.extract()
            print(f"--- Page {page_num} Table 0 ---")
            for r_idx, r in enumerate(rows):
                if r_idx < 5:
                    print(f"Row {r_idx}: {r}")
