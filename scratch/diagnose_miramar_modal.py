"""
Runs the pipeline on miramar_untagged.pdf on Modal A10G and prints
per-stage element counts:
  - Stage 3: MinerU region count + categories per page
  - Stage 4/5: tagged element count + tag distribution per page
  - Stage 10 input: surviving elements per page + empty merged_from count
  - Stage 10 output: BDC/EMC counts per page
"""
import modal
import os
from pathlib import Path

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

app = modal.App("miramar-diagnose")


@app.function(image=image, gpu="A10G", timeout=3600)
def diagnose_miramar(pdf_bytes: bytes, filename: str) -> str:
    import sys
    import tempfile
    from collections import Counter
    sys.path.append("/root")

    from tagger.pipeline import AutoTaggerPipeline

    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    # ── Monkey-patch stages ──────────────────────────────────────────────

    original_stage3 = AutoTaggerPipeline._stage3_layout

    def patched_stage3(self, input_pdf, doc_data):
        original_stage3(self, input_pdf, doc_data)
        log("\n=== STAGE 3: MinerU Layout Regions ===")
        for page_num, page_data in sorted(doc_data.pages.items()):
            regions = page_data.layout_regions or []
            cats = Counter(
                r.category.value if hasattr(r.category, "value") else str(r.category)
                for r in regions
            )
            log(f"  Page {page_num}: {len(regions)} regions  {dict(cats)}")

    AutoTaggerPipeline._stage3_layout = patched_stage3

    original_stage45 = AutoTaggerPipeline._stage4_5_route_extract

    def patched_stage45(self, doc_data):
        original_stage45(self, doc_data)
        log("\n=== STAGE 4/5: Tagged Elements (after route + table specialist) ===")
        for page_num, page_data in sorted(doc_data.pages.items()):
            elems = page_data.tagged_elements or []
            tags = Counter(
                e.pdf_tag.value if hasattr(e.pdf_tag, "value") else str(e.pdf_tag)
                for e in elems
            )
            empty_mf = sum(1 for e in elems if len(e.merged_from) == 0)
            log(f"  Page {page_num}: {len(elems)} elements  tags={dict(tags)}  empty_merged_from={empty_mf}")

    AutoTaggerPipeline._stage4_5_route_extract = patched_stage45

    original_stage10 = AutoTaggerPipeline._stage10_write

    def patched_stage10(self, input_pdf, output_pdf, doc_data):
        log("\n=== STAGE 10 INPUT: Surviving elements per page ===")
        for page_num, page_data in sorted(doc_data.pages.items()):
            elems = page_data.tagged_elements or []
            tags = Counter(
                e.pdf_tag.value if hasattr(e.pdf_tag, "value") else str(e.pdf_tag)
                for e in elems
            )
            empty_mf = sum(1 for e in elems if len(e.merged_from) == 0)
            log(f"  Page {page_num}: {len(elems)} elements  tags={dict(tags)}  empty_merged_from={empty_mf}")

        original_stage10(self, input_pdf, output_pdf, doc_data)

        log("\n=== STAGE 10 OUTPUT: BDC/EMC per page ===")
        import pikepdf
        pdf = pikepdf.open(output_pdf)
        for i, page in enumerate(pdf.pages):
            cs = pikepdf.parse_content_stream(page)
            bdc = sum(1 for op in cs if str(op.operator) == "BDC")
            emc = sum(1 for op in cs if str(op.operator) == "EMC")
            log(f"  Page {i+1}: BDC={bdc} EMC={emc} balanced={bdc==emc}")

    AutoTaggerPipeline._stage10_write = patched_stage10

    # ── Run ──────────────────────────────────────────────────────────────

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, filename)
        output_path = os.path.join(tmpdir, f"tagged_{filename}")

        with open(input_path, "wb") as f:
            f.write(pdf_bytes)

        log(f"Running pipeline on {filename}...")
        pipeline = AutoTaggerPipeline()
        pipeline.run(input_pdf=input_path, output_pdf=output_path)

    return "\n".join(lines)


@app.local_entrypoint()
def main():
    pdf_path = Path("/Users/rahulkhatri/Tagger/miramar_untagged.pdf")
    print(f"Sending {pdf_path.name} to Modal A10G for diagnosis...")
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    result = diagnose_miramar.remote(pdf_bytes, pdf_path.name)
    print(result)

    out = Path("/Users/rahulkhatri/Tagger/scratch/miramar_diag_output.txt")
    out.write_text(result)
    print(f"\nSaved to {out}")
