"""
QA runner for OpenDataLoader-tagged PDFs — reuses audit_pdf() from run_qa_modal.
Spawns all chunks in parallel to Modal Gemma, saves to qa_results_odl/.

Run via: modal run scratch/run_qa_odl.py
"""
import sys
import os
import json
from pathlib import Path

import modal

sys.path.insert(0, os.path.dirname(__file__))
from run_qa_modal import audit_pdf

app = modal.App("qa-runner-odl")


@app.local_entrypoint()
def main():
    OUT = Path("/Users/rahulkhatri/Tagger/output_odl")
    RES = Path("/Users/rahulkhatri/QA Tool/scratch/qa_results_odl")
    RES.mkdir(parents=True, exist_ok=True)

    pdfs = [
        OUT / "CITY OF MIRAMAR, FLORIDA_tagged.pdf",
        OUT / "Missouri State Epidemiological Profile July 2018_tagged.pdf",
        OUT / "Osteoarthritis_tagged.pdf",
        OUT / "Summary of Revenues and Expenditures_tagged.pdf",
        OUT / "nyvra-factsheet_tagged.pdf",
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
