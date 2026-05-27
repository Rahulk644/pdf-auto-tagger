import sys
from pathlib import Path
from tagger.pipeline import AutoTaggerPipeline

def run_local_smoke_test():
    p = AutoTaggerPipeline()
    doc_data = p.run(
        "miramar_untagged.pdf",
        output_pdf="output_local.pdf", page_range=(1, 1)
    )
    
    page_data = doc_data.pages.get(1)
    if not page_data:
        print("Failed to process page 1.")
        return

    table_found = False
    for el in page_data.tagged_elements:
        if el.pdf_tag.value == "Table":
            table_found = True
            cells = el.specialist_data.get("cells", [])
            print(f"\nFound Table! ID: {el.element_id}")
            print(f"Total cells extracted: {len(cells)}")
            if cells:
                print("First 15 cells:")
                for c in cells[:15]:
                    print(f"  Row {c['row_idx']}, Col {c['col_idx']}: merged_from={len(c.get('merged_from', []))} chars, header={c.get('is_header')}, row_header={c.get('is_row_header')}, text='{c.get('text', '')[:30]}...'")
            else:
                print("No cells extracted!")
            break

    if not table_found:
        print("No tables found on page 1.")

if __name__ == "__main__":
    run_local_smoke_test()
