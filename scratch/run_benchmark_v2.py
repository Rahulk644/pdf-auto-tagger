"""Strip+V2 remediation regen for the PDF-A-B benchmark (Modal A10G).

Runs the full pipeline on the 25 expert-FAILED docs (stripped to untagged in
input_benchmark_v2/ — 20 were tagged/V1, 5 already untagged) -> output_benchmark_v2/.
This is the substrate's REMEDIATION axis: can our V2 pipeline bring expert-failed
docs to passing per the verdict contracts. Run:  modal run scratch/run_benchmark_v2.py

Large tagged outputs (e.g. Missouri, the functional_hyperlinks docs) exceed Modal's
function-return-value transport and fail with "BlobGet not implemented". So the
remote function writes outputs to a modal.Volume and returns only their names; the
local entrypoint streams them back from the Volume. No large value crosses the wire.
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

# Outputs flow through this Volume instead of the function return value, so
# arbitrarily large tagged PDFs never hit Modal's return-value blob transport.
out_vol = modal.Volume.from_name("pdf-auto-tagger-benchmark-v2-out", create_if_missing=True)


@app.function(image=image, gpu="A10G", timeout=3600, volumes={"/outputs": out_vol},
              max_containers=10)
def run_tagger_remotely(pdf_bytes: bytes, filename: str) -> dict:
    import sys
    sys.path.append("/root")
    from tagger.pipeline import AutoTaggerPipeline
    import tempfile

    stem = Path(filename).stem
    out_name = filename
    report_name = f"{stem}_report.json"

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, filename)
        output_path = os.path.join(tmpdir, f"tagged_{filename}")
        report_path = os.path.join(tmpdir, f"report_{filename}.json")
        with open(input_path, "wb") as f:
            f.write(pdf_bytes)
        pipeline = AutoTaggerPipeline()
        pipeline.run(input_pdf=input_path, output_pdf=output_path, report_path=report_path)

        with open(output_path, "rb") as src, open(f"/outputs/{out_name}", "wb") as dst:
            dst.write(src.read())
        has_report = os.path.exists(report_path)
        if has_report:
            with open(report_path, "rb") as src, open(f"/outputs/{report_name}", "wb") as dst:
                dst.write(src.read())

    out_vol.commit()
    return {"out_name": out_name, "report_name": report_name if has_report else None}


@app.local_entrypoint()
def main():
    in_dir = Path("/Users/rahulkhatri/Tagger/input_benchmark_v2")
    out_dir = Path("/Users/rahulkhatri/Tagger/output_benchmark_v2")
    out_dir.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(in_dir.glob("*.pdf"))
    names = [p.name for p in pdfs]
    payloads = [p.read_bytes() for p in pdfs]
    print(f"Remediation regen: {len(pdfs)} stripped failed docs -> {out_dir} (fan-out)")
    # Fan out across containers; .map() preserves input order so zip(names, ...)
    # aligns. return_exceptions keeps one failed doc from aborting the batch.
    for name, res in zip(
        names,
        run_tagger_remotely.map(
            payloads, names, return_exceptions=True, wrap_returned_exceptions=False
        ),
    ):
        if isinstance(res, Exception):
            print(f"FAIL {name}: {res}")
            continue
        try:
            (out_dir / res["out_name"]).write_bytes(b"".join(out_vol.read_file(res["out_name"])))
            if res["report_name"]:
                (out_dir / res["report_name"]).write_bytes(
                    b"".join(out_vol.read_file(res["report_name"]))
                )
            print(f"OK {name}")
        except Exception as e:
            print(f"FAIL {name} (readback): {e}")
