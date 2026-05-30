import time
import opendataloader_pdf

SRC = "/Users/rahulkhatri/Downloads/pdf_tag_tool/PDF & Reports/UNTAGGED PDFs"
OUT = "/Users/rahulkhatri/Tagger/output_odl"

inputs = [
    f"{SRC}/CITY OF MIRAMAR, FLORIDA.pdf",
    f"{SRC}/Missouri State Epidemiological Profile July 2018.pdf",
    f"{SRC}/Osteoarthritis.pdf",
    f"{SRC}/Summary of Revenues and Expenditures.pdf",
    f"{SRC}/nyvra-factsheet.pdf",
]

t0 = time.time()
opendataloader_pdf.convert(
    input_path=inputs,
    output_dir=OUT,
    format="tagged-pdf",
)
print(f"\nDONE in {time.time()-t0:.0f}s")
