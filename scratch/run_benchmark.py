"""Entrypoint for the PDF-Accessibility-Benchmark substrate.

Checker framing is CPU/free and runs on all 125 docs locally:
  PYTHONPATH=. .venv3/bin/python scratch/run_benchmark.py <benchmark_root>

Remediation framing needs the strip+V2 tagged outputs (the one Modal regen);
pass --remediation-dir DIR where DIR/<openalex_id>.pdf is the re-tagged output.
"""
import argparse
from pathlib import Path

from tagger.benchmark.harness import run_checker, run_remediation
from tagger.benchmark.loader import ADDRESSED_CRITERIA, load_benchmark
from tagger.benchmark.report import build_scorecard, format_scorecard

# Criteria with a deterministic verdict fn (font_embedding is an adjacent axis).
SCORED = set(ADDRESSED_CRITERIA) | {"font_embedding"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark_root")
    ap.add_argument("--remediation-dir", default=None,
                    help="dir of strip+V2 outputs named <openalex_id>.pdf")
    args = ap.parse_args()

    tasks = list(load_benchmark(args.benchmark_root))
    checker = run_checker(tasks, criteria=SCORED)

    remediation = None
    if args.remediation_dir:
        rdir = Path(args.remediation_dir)

        def output_for(task):
            p = rdir / f"{task.openalex_id}.pdf"
            return str(p) if p.exists() else None

        remediation = run_remediation(tasks, output_for, criteria=SCORED)

    print(format_scorecard(build_scorecard(checker, remediation)))


if __name__ == "__main__":
    main()
