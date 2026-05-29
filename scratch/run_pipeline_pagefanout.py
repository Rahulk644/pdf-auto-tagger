"""Unit 3 — page-level cross-container fan-out for the tagging pipeline.

Document-level fan-out load-balances badly when page counts range 1..100: a
100-page doc pins one container while 1-page docs idle others. This decouples the
ONLY GPU stage (Stage 3 / MinerU layout) from the CPU stages via the pipeline seam
(prep_through_merge / render_layout_pages / inject_layout / finish_from_route):

  process_doc (CPU, one per doc, concurrent):
    Stages 0-2  ->  render pages  ->  fan THIS doc's pages to the shared GPU pool
    ->  inject regions  ->  Stages 4-10  ->  write tagged PDF to the Volume

  detect_pages (GPU pool, bounded, SHARED across all docs):
    loads MinerU2.5 (transformers, layout_detect) once per container, runs a chunk.

Every doc submits its page-chunks to the SAME bounded GPU pool, so a 100-page doc's
chunks interleave with everyone else's — pages, not docs, are the unit of work.
DocumentData never crosses the wire (prep+finish share one process per doc); only
page images go out and regions come back.

(Uses transformers layout_detect, the Unit 2 path — vLLM is degenerate for MinerU2.5
layout and LMDeploy has a dep conflict; both parked.)  Run: modal run scratch/run_pipeline_pagefanout.py
"""
import io
import os
import tempfile
from pathlib import Path

import modal

PAGES_PER_CHUNK = 3          # a doc's pages are chunked this fine, then fanned out
GPU_POOL = 10                # max concurrent GPU containers in the shared layout pool

# CPU stages (0-2, 4-10): no MinerU. add_local_dir must be the LAST build step.
cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("PyMuPDF", "pdfplumber", "pillow", "pikepdf", "numpy")
    .add_local_dir("/Users/rahulkhatri/Tagger/tagger", remote_path="/root/tagger")
)

# GPU layout pool: MinerU2.5 via transformers (layout_detect, the Unit 2 path).
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install("torch", "torchvision", "PyMuPDF", "pillow", "pikepdf")
    .pip_install("mineru-vl-utils[transformers]")
    .pip_install("transformers>=4.45.0", "accelerate", "qwen-vl-utils")
    .add_local_dir("/Users/rahulkhatri/Tagger/tagger", remote_path="/root/tagger")
)

app = modal.App("pipeline-pagefanout")
out_vol = modal.Volume.from_name("pipeline-pagefanout-out", create_if_missing=True)

_DET = None  # per-container MinerU detector, loaded once and reused across .map inputs


@app.function(image=gpu_image, gpu="A10G", timeout=2400, max_containers=GPU_POOL)
def detect_pages(chunk: tuple) -> dict:
    """chunk = (page_nums: list[int], pngs: list[bytes]) for ONE doc's sub-chunk.
    Returns {page_num: list[LayoutRegion]}. MinerU loads once per container (reuse)."""
    import sys
    sys.path.append("/root")
    from PIL import Image
    from tagger.stage3_layout.layout_detector import MinerULayoutDetector

    page_nums, pngs = chunk
    global _DET
    if _DET is None:
        _DET = MinerULayoutDetector()   # transformers backend (default), layout_detect
        _DET.load()

    out = {}
    for page_num, png in zip(page_nums, pngs):
        img = Image.open(io.BytesIO(png)).convert("RGB")
        out[page_num] = _DET.detect(img, page_num)
    return out


@app.function(image=cpu_image, timeout=3600, volumes={"/outputs": out_vol})
def process_doc(pdf_bytes: bytes, filename: str) -> dict:
    import sys
    sys.path.append("/root")
    from tagger.pipeline import AutoTaggerPipeline
    from tagger.models.data_types import DocumentData

    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, filename)
        out_path = os.path.join(tmp, f"tagged_{filename}")
        with open(in_path, "wb") as f:
            f.write(pdf_bytes)

        pipeline = AutoTaggerPipeline()
        doc_data = DocumentData(input_path=in_path, num_pages=0)
        pipeline.prep_through_merge(in_path, doc_data)               # Stages 0-2 (CPU)
        imgs, page_nums = pipeline.render_layout_pages(in_path, doc_data)

        # Chunk THIS doc's pages and fan them into the shared GPU pool.
        chunks = []
        for i in range(0, len(imgs), PAGES_PER_CHUNK):
            pngs = []
            for im in imgs[i:i + PAGES_PER_CHUNK]:
                buf = io.BytesIO()
                im.save(buf, format="PNG")
                pngs.append(buf.getvalue())
            chunks.append((page_nums[i:i + PAGES_PER_CHUNK], pngs))

        regions_by_page: dict = {}
        if chunks:
            for part in detect_pages.map(chunks):
                regions_by_page.update(part)

        pipeline.inject_layout(in_path, doc_data, regions_by_page)   # Stage 3 results in
        pipeline.finish_from_route(in_path, out_path, doc_data)      # Stages 4-10 (CPU)

        with open(out_path, "rb") as src, open(f"/outputs/{filename}", "wb") as dst:
            dst.write(src.read())
    out_vol.commit()
    return {"out_name": filename}


@app.local_entrypoint()
def main():
    in_dir = Path("/Users/rahulkhatri/Tagger/input_benchmark_v2")
    out_dir = Path("/Users/rahulkhatri/Tagger/output_pagefanout")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Validation subset; widen to sorted(in_dir.glob("*.pdf")) for a full regen.
    picks = [
        "table_structure__W2296421107.pdf",
        "functional_hyperlinks__W2893185172.pdf",
        "logical_reading_order__W2953207266.pdf",
    ]
    pdfs = [(n, (in_dir / n)) for n in picks if (in_dir / n).exists()]
    names = [n for n, _ in pdfs]
    payloads = [p.read_bytes() for _, p in pdfs]
    print(f"page-fanout: {len(names)} docs, {PAGES_PER_CHUNK}p/chunk, GPU pool {GPU_POOL}")
    for name, res in zip(
        names, process_doc.map(payloads, names, return_exceptions=True, wrap_returned_exceptions=False)
    ):
        if isinstance(res, Exception):
            print(f"FAIL {name}: {res}")
            continue
        (out_dir / res["out_name"]).write_bytes(b"".join(out_vol.read_file(res["out_name"])))
        print(f"OK {name}")
