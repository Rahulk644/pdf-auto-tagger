"""Full PDF-A-B remediation regen: strip+tag ALL 35 unique benchmark docs.

The 35 unique source PDFs carry the full 125 (doc x criterion) task set. The
remediation policy is strip+V2: remove any existing structure, then tag from
scratch with the current CPU pipeline. Outputs land in out_dir/<openalex_id>.pdf
for scratch/run_benchmark.py --remediation-dir to score against expert labels.
"""
import sys
import tempfile
import time
from pathlib import Path

import pikepdf

from tagger.benchmark.loader import load_benchmark
from tagger.pipeline import AutoTaggerPipeline

ROOT = "/private/tmp/pdf-bench"
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "output_benchmark_full_cpu")
OUT.mkdir(parents=True, exist_ok=True)


def strip(in_path: str, out_path: str) -> None:
    """Remove existing struct tree / marks so we re-tag from a clean slate
    (orphan BDC in content streams is stripped by Stage 10)."""
    pdf = pikepdf.open(in_path)
    try:
        root = pdf.Root
        for k in ("/StructTreeRoot", "/MarkInfo"):
            if k in root:
                del root[k]
        for page in pdf.pages:
            for k in ("/StructParents", "/Tabs"):
                if k in page.obj:
                    del page.obj[k]
        pdf.save(out_path)
    finally:
        pdf.close()


uniq: dict[str, str] = {}
for t in load_benchmark(ROOT):
    uniq.setdefault(t.openalex_id, t.pdf_path)

pipe = AutoTaggerPipeline()
print(f"strip+tag {len(uniq)} unique docs -> {OUT}", flush=True)
tot = 0.0
for i, (oid, src) in enumerate(sorted(uniq.items()), 1):
    out_pdf = OUT / f"{oid}.pdf"
    t0 = time.time()
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
            stripped = tf.name
        strip(src, stripped)
        pipe.run(input_pdf=stripped, output_pdf=str(out_pdf),
                 report_path=str(out_pdf.with_suffix(".json")))
        dt = time.time() - t0
        tot += dt
        print(f"[{i}/{len(uniq)}] {oid}  {dt:.1f}s", flush=True)
    except Exception as e:
        print(f"[{i}/{len(uniq)}] {oid}  FAILED: {e}", flush=True)
    finally:
        Path(stripped).unlink(missing_ok=True)
print(f"DONE {len(uniq)} docs in {tot:.1f}s", flush=True)
