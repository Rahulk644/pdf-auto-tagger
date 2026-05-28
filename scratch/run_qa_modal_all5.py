"""
QA runner for my-pipeline re-tagged outputs (Stage 8 + false-positive fix).
Reuses audit_pdf() from run_qa_modal. Reads all 5 corpus docs from output_modal/,
saves to qa_results_modal_stage8/.

Run via: modal run scratch/run_qa_modal_all5.py
"""
import sys
import os
import json
from pathlib import Path

import modal

sys.path.insert(0, os.path.dirname(__file__))
from run_qa_modal import audit_pdf

app = modal.App("qa-runner-all5")


@app.local_entrypoint()
def main():
    OUT = Path("/Users/rahulkhatri/Tagger/output_modal")
    RES = Path("/Users/rahulkhatri/PREP QA Tool/scratch/qa_results_modal_stage8")
    RES.mkdir(parents=True, exist_ok=True)

    pdfs = [
        OUT / "miramar_untagged.pdf",
        OUT / "Missouri State Epidemiological Profile July 2018.pdf",
        OUT / "Osteoarthritis.pdf",
        OUT / "Summary of Revenues and Expenditures.pdf",
        OUT / "nyvra-factsheet.pdf",
    ]

    for pdf_path in pdfs:
        if not pdf_path.exists():
            print(f"Skipping (missing): {pdf_path.name}", flush=True)
            continue

        print(f"\nRunning QA on {pdf_path.name}...", flush=True)
        pdf_bytes = pdf_path.read_bytes()
        result = audit_pdf(pdf_bytes, pdf_path.name)

        out_file = RES / f"qa_{pdf_path.stem}.json"
        out_file.write_text(json.dumps(result, indent=2))
        print(f"Saved: {out_file}", flush=True)
        print(f"Elements: {result['total_elements']}  Accuracy: {result['accuracy']:.1%}", flush=True)
