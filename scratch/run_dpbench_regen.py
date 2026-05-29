"""dp-bench V2 regen on Modal A10G (PILOT scope by default).

Runs the full pipeline on dp-bench PDFs (input_dpbench_pilot/ -> output_dpbench_pilot/),
saving tagged {id}.pdf + {id}_report.json (the report carries stage_timings, so we
can measure MinerU per-page cost before scaling). Same image/GPU as run_benchmark_v2.

Run:  /Users/rahulkhatri/Library/Python/3.9/bin/modal run scratch/run_dpbench_regen.py
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

app = modal.App("pdf-auto-tagger-dpbench")

# Dirs configurable for pilot vs full corpus:
#   DPBENCH_IN=~/benchmarks/opendataloader-bench/pdfs DPBENCH_OUT=output_dpbench_full
IN_DIR = os.environ.get("DPBENCH_IN", "/Users/rahulkhatri/Tagger/input_dpbench_pilot")
OUT_DIR = os.environ.get("DPBENCH_OUT", "/Users/rahulkhatri/Tagger/output_dpbench_pilot")


@app.function(image=image, gpu="A10G", timeout=3600, max_containers=10)
def run_tagger_remotely(pdf_bytes: bytes, filename: str) -> tuple[bytes, bytes]:
    import sys, tempfile, time
    sys.path.append("/root")
    from tagger.pipeline import AutoTaggerPipeline

    with tempfile.TemporaryDirectory() as tmp:
        ip = os.path.join(tmp, filename)
        op = os.path.join(tmp, f"tagged_{filename}")
        rp = os.path.join(tmp, f"report_{filename}.json")
        with open(ip, "wb") as f:
            f.write(pdf_bytes)
        t0 = time.time()
        AutoTaggerPipeline().run(input_pdf=ip, output_pdf=op, report_path=rp)
        print(f"[{filename}] end-to-end {time.time() - t0:.1f}s")
        out = open(op, "rb").read()
        rep = open(rp, "rb").read() if os.path.exists(rp) else b""
    return out, rep


@app.local_entrypoint()
def main():
    in_dir = Path(IN_DIR)
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(in_dir.glob("*.pdf"))
    print(f"dp-bench regen: {len(pdfs)} docs -> {out_dir}")
    names = [p.name for p in pdfs]
    payloads = [p.read_bytes() for p in pdfs]
    # Fan out across containers (near cost-neutral, ~cuts wall-clock); skip docs
    # that raise rather than aborting the batch.
    ok = fail = 0
    for name, result in zip(
        names,
        run_tagger_remotely.map(payloads, names, return_exceptions=True),
    ):
        if isinstance(result, Exception):
            fail += 1
            print(f"FAIL {name}: {result}")
            continue
        tagged, report = result
        (out_dir / name).write_bytes(tagged)
        if report:
            (out_dir / f"{Path(name).stem}_report.json").write_bytes(report)
        ok += 1
        print(f"OK {name}")
    print(f"done: ok={ok} fail={fail}")
