import pdfplumber
from tagger.config import STANDARD_DPI, PDF_NATIVE_DPI

with pdfplumber.open("miramar_untagged.pdf") as pdf:
    page = pdf.pages[0]
    # MinerU crop_box for table on Page 1:
    crop_box = (71.6 - 5, 342.9 - 5, 541.0 + 5, 613.0 + 5)
    
    cropped = page.within_bbox(crop_box)
    table_settings = {"vertical_strategy": "text", "horizontal_strategy": "text"}
    tables = cropped.find_tables(table_settings=table_settings)
    if tables:
        t = tables[0]
        rows = t.extract()
        print(f"--- Page 1 Table ---")
        for r_idx, r in enumerate(rows):
            if r_idx < 10:
                print(f"Row {r_idx}: {r}")
