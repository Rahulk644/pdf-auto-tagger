import os
from pathlib import Path
from tagger.pipeline import AutoTaggerPipeline

pdfs = [
    "/Users/rahulkhatri/Downloads/pdf_tag_tool/PREP PDF & Reports/PREP PDFs/CITY OF MIRAMAR, FLORIDA.pdf",
    "/Users/rahulkhatri/Downloads/pdf_tag_tool/PREP PDF & Reports/PREP PDFs/Missouri State Epidemiological Profile July 2018.pdf",
    "/Users/rahulkhatri/Downloads/pdf_tag_tool/PREP PDF & Reports/PREP PDFs/Osteoarthritis.pdf",
    "/Users/rahulkhatri/Downloads/pdf_tag_tool/PREP PDF & Reports/PREP PDFs/Summary of Revenues and Expenditures.pdf",
    "/Users/rahulkhatri/Downloads/pdf_tag_tool/PREP PDF & Reports/PREP PDFs/nyvra-factsheet.pdf"
]

out_dir = Path("/Users/rahulkhatri/Tagger/output_batch")
out_dir.mkdir(parents=True, exist_ok=True)

pipeline = AutoTaggerPipeline()

for pdf in pdfs:
    pdf_path = Path(pdf)
    if not pdf_path.exists():
        print(f"Skipping {pdf_path.name}, not found.")
        continue
    out_pdf = out_dir / f"{pdf_path.stem}_tagged.pdf"
    out_report = out_dir / f"{pdf_path.stem}_report.json"
    
    print(f"Processing {pdf_path.name}...")
    try:
        pipeline.run(
            input_pdf=str(pdf_path),
            output_pdf=str(out_pdf),
            report_path=str(out_report)
        )
        print(f"Successfully processed {pdf_path.name}")
    except Exception as e:
        print(f"Error processing {pdf_path.name}: {e}")
