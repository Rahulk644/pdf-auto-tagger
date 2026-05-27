import pickle
import pdfplumber


# We don't have page_elements easily accessible. 
# But let's check what `pdfplumber` cell_bbox values are!
with pdfplumber.open("output_modal/miramar_untagged.pdf") as pdf:
    page = pdf.pages[0]
    crop_box = (71.6 - 5, 342.9 - 5, 541.0 + 5, 613.0 + 5)
    cropped = page.within_bbox(crop_box)
    t = cropped.find_tables({"vertical_strategy": "text", "horizontal_strategy": "text"})[0]
    for r_idx, row in enumerate(t.rows):
        if r_idx < 2:
            print(f"Row {r_idx}")
            for c_idx, cell_bbox in enumerate(row.cells):
                print(f"  Col {c_idx}: {cell_bbox}")
