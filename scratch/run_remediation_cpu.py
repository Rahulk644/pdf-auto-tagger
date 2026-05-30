"""Re-tag the stripped PDF-A-B remediation inputs with the CURRENT CPU pipeline.

The prior remediation set lives in input_benchmark_v2/ as
<criterion>__<openalex_id>.pdf (already stripped to untagged). The benchmark
remediation harness wants outputs named <openalex_id>.pdf. We tag each input
locally on the CPU backend (no Modal/GPU) into out_dir/<openalex_id>.pdf, then
run_benchmark.py --remediation-dir out_dir scores them against the expert labels.
"""
import sys
import time
from pathlib import Path

from tagger.pipeline import AutoTaggerPipeline

IN = Path("input_benchmark_v2")
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "output_benchmark_v2_cpu")
OUT.mkdir(parents=True, exist_ok=True)

pdfs = sorted(IN.glob("*.pdf"))
pipe = AutoTaggerPipeline()
print(f"re-tagging {len(pdfs)} remediation inputs -> {OUT}", flush=True)
tot = 0.0
for i, p in enumerate(pdfs, 1):
    oid = p.stem.split("__")[-1]  # <criterion>__<openalex_id> -> openalex_id
    out_pdf = OUT / f"{oid}.pdf"
    t0 = time.time()
    try:
        pipe.run(input_pdf=str(p), output_pdf=str(out_pdf),
                 report_path=str(out_pdf.with_suffix(".json")))
        dt = time.time() - t0
        tot += dt
        print(f"[{i}/{len(pdfs)}] {oid}  {dt:.1f}s", flush=True)
    except Exception as e:
        print(f"[{i}/{len(pdfs)}] {oid}  FAILED: {e}", flush=True)
print(f"DONE {len(pdfs)} docs in {tot:.1f}s", flush=True)
