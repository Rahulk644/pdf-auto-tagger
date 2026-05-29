"""Strip+V2 remediation regen for the PDF-A-B benchmark (Modal A10G).

Runs the full pipeline on the 25 expert-FAILED docs (stripped to untagged in
input_benchmark_v2/ — 20 were tagged/V1, 5 already untagged) -> output_benchmark_v2/.
This is the substrate's REMEDIATION axis: can our V2 pipeline bring expert-failed
docs to passing per the verdict contracts. Run:  modal run scratch/run_benchmark_v2.py
"""
import os
from pathlib import Path

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install("torch", "torchvision", "PyMuPDF", "pdfplumber", "pillow", "pikepdf", "tqdm")
    .pip_install("mineru-vl-utils[transformers]")
    .pip_install("transformers>=4.45.0", "accelerate", "qwen-vl-utils")
    .add_local_dir("/Users/rahulkhatri/Tagger/tagger", remote_path="/root/tagger")
)

app = modal.App("pdf-auto-tagger-benchmark-v2")


@app.function(image=image, gpu="A10G", timeout=3600)
def run_tagger_remotely(pdf_bytes: bytes, filename: str) -> tuple[bytes, bytes]:
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
        pipeline.run(input_pdf=input_path, output_pdf=output_path, report_path=report_path)
        with open(output_path, "rb") as f:
            out_bytes = f.read()
        report_bytes = b""
        if os.path.exists(report_path):
            with open(report_path, "rb") as f:
                report_bytes = f.read()
    return out_bytes, report_bytes


@app.local_entrypoint()
def main():
    in_dir = Path("/Users/rahulkhatri/Tagger/input_benchmark_v2")
    out_dir = Path("/Users/rahulkhatri/Tagger/output_benchmark_v2")
    out_dir.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(in_dir.glob("*.pdf"))
    print(f"Remediation regen: {len(pdfs)} stripped failed docs -> {out_dir}")
    for pdf_path in pdfs:
        try:
            tagged, report = run_tagger_remotely.remote(pdf_path.read_bytes(), pdf_path.name)
            (out_dir / pdf_path.name).write_bytes(tagged)
            if report:
                (out_dir / f"{pdf_path.stem}_report.json").write_bytes(report)
            print(f"OK {pdf_path.name}")
        except Exception as e:
            print(f"FAIL {pdf_path.name}: {e}")
