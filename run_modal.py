import modal
import os
from pathlib import Path

# Build an image with all the necessary dependencies for the pipeline
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch",
        "torchvision",
        "PyMuPDF",
        "pdfplumber",
        "pillow",
        "pikepdf",
        "tqdm",
    )
    .pip_install("mineru-vl-utils[transformers]")
    .pip_install("transformers>=4.45.0", "accelerate", "qwen-vl-utils")
    .add_local_dir("/Users/rahulkhatri/Tagger/tagger", remote_path="/root/tagger")
)

app = modal.App("pdf-auto-tagger")

@app.function(
    image=image,
    gpu="A10G", # Allocate a cloud GPU
    timeout=3600
)
def run_tagger_remotely(pdf_bytes: bytes, filename: str) -> bytes:
    import sys
    sys.path.append("/root")
    
    from tagger.pipeline import AutoTaggerPipeline
    
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, filename)
        output_path = os.path.join(tmpdir, f"tagged_{filename}")
        report_path = os.path.join(tmpdir, f"report_{filename}.json")
        
        with open(input_path, "wb") as f:
            f.write(pdf_bytes)
            
        pipeline = AutoTaggerPipeline()
        # This will use the GPU in the cloud
        pipeline.run(input_pdf=input_path, output_pdf=output_path, report_path=report_path)
        
        with open(output_path, "rb") as f:
            out_bytes = f.read()
            
    return out_bytes

@app.local_entrypoint()
def main():
    pdfs = [
        "/Users/rahulkhatri/Downloads/pdf_tag_tool/PREP PDF & Reports/PREP PDFs/CITY OF MIRAMAR, FLORIDA.pdf",
        "/Users/rahulkhatri/Downloads/pdf_tag_tool/PREP PDF & Reports/PREP PDFs/Missouri State Epidemiological Profile July 2018.pdf",
        "/Users/rahulkhatri/Downloads/pdf_tag_tool/PREP PDF & Reports/PREP PDFs/Osteoarthritis.pdf",
        "/Users/rahulkhatri/Downloads/pdf_tag_tool/PREP PDF & Reports/PREP PDFs/Summary of Revenues and Expenditures.pdf",
        "/Users/rahulkhatri/Downloads/pdf_tag_tool/PREP PDF & Reports/PREP PDFs/nyvra-factsheet.pdf"
    ]
    
    out_dir = Path("/Users/rahulkhatri/Tagger/output_modal")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for pdf_path in pdfs:
        p = Path(pdf_path)
        if not p.exists():
            print(f"Skipping {p.name}, not found.")
            continue
            
        print(f"Deploying {p.name} to Modal GPU...")
        with open(p, "rb") as f:
            pdf_bytes = f.read()
            
        try:
            tagged_bytes = run_tagger_remotely.remote(pdf_bytes, p.name)
            out_file = out_dir / p.name
            with open(out_file, "wb") as f:
                f.write(tagged_bytes)
            print(f"✅ Successfully processed and saved {out_file}")
        except Exception as e:
            print(f"❌ Failed processing {p.name}: {e}")
