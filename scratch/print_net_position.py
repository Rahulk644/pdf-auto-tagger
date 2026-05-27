import pdfplumber

with pdfplumber.open("miramar_untagged.pdf") as pdf:
    page = pdf.pages[0]
    crop_box_72dpi = (50, 360, 550, 560)
    cropped = page.within_bbox(crop_box_72dpi)
    tables = cropped.find_tables({"vertical_strategy": "text", "horizontal_strategy": "text"})
    
    t = tables[0]
    rows = t.extract()
    for i, row in enumerate(rows):
        if row[0] and "Net" in row[0]:
            print(f"Row {i}:")
            for col_idx, cell_bbox in enumerate(t.rows[i].cells):
                print(f"  Col {col_idx}: '{row[col_idx]}' -> bbox: {cell_bbox}")
