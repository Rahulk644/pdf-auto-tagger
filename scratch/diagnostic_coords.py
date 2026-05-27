import pdfplumber
import logging
from tagger.config import STANDARD_DPI, PDF_NATIVE_DPI
from tagger.stage1_extraction.native_extractor import _extract_page_chars

logging.basicConfig(level=logging.ERROR)

def run_diagnostic():
    print("Parsing PDF to get native PageElements...")
    with pdfplumber.open("miramar_untagged.pdf") as pdf:
        page_1_elements = _extract_page_chars(pdf.pages[0], 1)
    
    print(f"Extracted {len(page_1_elements)} elements from Page 1.")
    
    crop_box_150dpi = [117, 433, 884, 774]
    scale = STANDARD_DPI / PDF_NATIVE_DPI
    
    element_points = []
    for el in page_1_elements:
        cx = ((el.bbox[0] + el.bbox[2]) / 2.0) / scale
        cy = ((el.bbox[1] + el.bbox[3]) / 2.0) / scale
        element_points.append((el, cx, cy))

    crop_box_72dpi = (
        (crop_box_150dpi[0] / scale) - 5,
        (crop_box_150dpi[1] / scale) - 5,
        (crop_box_150dpi[2] / scale) + 5,
        (crop_box_150dpi[3] / scale) + 5,
    )
    
    print("\n--- Diagnostic Coordinate Table ---")
    print(f"PDFplumber CropBox (72 DPI): {crop_box_72dpi}")

    with pdfplumber.open("miramar_untagged.pdf") as pdf:
        page = pdf.pages[0]
        cropped = page.within_bbox(crop_box_72dpi)
        table_settings = {"vertical_strategy": "text", "horizontal_strategy": "text"}
        tables = cropped.find_tables(table_settings=table_settings)
        
        t = tables[0]
        rows = t.extract()
        
        print("\nRow 0 Analysis:")
        for col_idx, cell_bbox in enumerate(t.rows[0].cells):
            cell_text = str(rows[0][col_idx]).strip() if rows[0][col_idx] is not None else ""
            print(f"\nCol {col_idx} [Text: '{cell_text[:25]}...']")
            print(f"  pdfplumber cell_bbox (72 DPI): {cell_bbox}")
            
            cx0, cy0, cx1, cy1 = cell_bbox
            intersecting = []
            for el, cx, cy in element_points:
                if cx0 <= cx <= cx1 and cy0 <= cy <= cy1:
                    intersecting.append(el)
            
            if intersecting:
                print(f"  Result: {len(intersecting)} PageElements INTERSECTED (Success)")
            else:
                print("  Result: NO PageElements INTERSECTED (Failure)")
                words = cell_text.split()
                if not words: continue
                
                closest_el = None
                for el, cx, cy in element_points:
                    if words[0] in el.text:
                        closest_el = (el, cx, cy)
                        break
                        
                if closest_el:
                    el, cx, cy = closest_el
                    print(f"  Found matching text '{el.text}' outside cell:")
                    print(f"    Expected cx within ({cx0:.2f}, {cx1:.2f}), Actual cx={cx:.2f}")
                    print(f"    Expected cy within ({cy0:.2f}, {cy1:.2f}), Actual cy={cy:.2f}")

if __name__ == "__main__":
    run_diagnostic()
