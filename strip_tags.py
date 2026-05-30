import pikepdf
import os
from pathlib import Path

pdfs = [
    "/Users/rahulkhatri/Downloads/pdf_tag_tool/PDF & Reports/incumbent PDFs/CITY OF MIRAMAR, FLORIDA.pdf",
    "/Users/rahulkhatri/Downloads/pdf_tag_tool/PDF & Reports/incumbent PDFs/Missouri State Epidemiological Profile July 2018.pdf",
    "/Users/rahulkhatri/Downloads/pdf_tag_tool/PDF & Reports/incumbent PDFs/Osteoarthritis.pdf",
    "/Users/rahulkhatri/Downloads/pdf_tag_tool/PDF & Reports/incumbent PDFs/Summary of Revenues and Expenditures.pdf",
    "/Users/rahulkhatri/Downloads/pdf_tag_tool/PDF & Reports/incumbent PDFs/nyvra-factsheet.pdf"
]

out_dir = Path("/Users/rahulkhatri/Downloads/pdf_tag_tool/PDF & Reports/UNTAGGED PDFs")
out_dir.mkdir(parents=True, exist_ok=True)

for p in pdfs:
    try:
        pdf = pikepdf.Pdf.open(p)
        
        # Remove tag information
        if "/StructTreeRoot" in pdf.Root:
            del pdf.Root["/StructTreeRoot"]
        if "/MarkInfo" in pdf.Root:
            del pdf.Root["/MarkInfo"]
            
        out_path = out_dir / Path(p).name
        pdf.save(str(out_path))
        print(f"Stripped {Path(p).name}")
    except Exception as e:
        print(f"Error on {Path(p).name}: {e}")
