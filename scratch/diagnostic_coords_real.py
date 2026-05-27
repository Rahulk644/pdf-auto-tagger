import pdfplumber
from tagger.config import STANDARD_DPI, PDF_NATIVE_DPI
from tagger.stage1_extraction.native_extractor import _extract_page_chars

def run_diagnostic():
    with pdfplumber.open("miramar_untagged.pdf") as pdf:
        page_1_elements = _extract_page_chars(pdf.pages[0], 1)
        
        # Manually define crop box for the REAL table
        # Table is from y=370 to y=550 roughly in 72 DPI
        crop_box_72dpi = (50, 360, 550, 560)
        
        scale = STANDARD_DPI / PDF_NATIVE_DPI
        
        element_points = []
        for el in page_1_elements:
            cx = ((el.bbox[0] + el.bbox[2]) / 2.0) / scale
            cy = ((el.bbox[1] + el.bbox[3]) / 2.0) / scale
            element_points.append((el, cx, cy))
        
        print("\n--- Diagnostic Coordinate Table (Real Table) ---")
        print(f"PDFplumber CropBox (72 DPI): {crop_box_72dpi}")

        page = pdf.pages[0]
        cropped = page.within_bbox(crop_box_72dpi)
        table_settings = {"vertical_strategy": "text", "horizontal_strategy": "text"}
        tables = cropped.find_tables(table_settings=table_settings)
        
        t = tables[0]
        rows = t.extract()
        
        # Find the row with "Current and other assets"
        target_row_idx = None
        for i, row in enumerate(rows):
            if row[0] and "Current" in row[0]:
                target_row_idx = i
                break
                
        if target_row_idx is None:
            return
            
        print(f"\nRow {target_row_idx} Analysis:")
        for col_idx, cell_bbox in enumerate(t.rows[target_row_idx].cells):
            cell_text = str(rows[target_row_idx][col_idx]).strip() if rows[target_row_idx][col_idx] is not None else ""
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
                    print(f"    Expected cx within ({cx0:.2f}, {cx1:.2f}), Actual cx={cx:.2f} (Delta: {min(abs(cx-cx0), abs(cx-cx1)):.2f})")
                    print(f"    Expected cy within ({cy0:.2f}, {cy1:.2f}), Actual cy={cy:.2f} (Delta: {min(abs(cy-cy0), abs(cy-cy1)):.2f})")

if __name__ == "__main__":
    run_diagnostic()
