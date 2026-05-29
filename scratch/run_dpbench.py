"""Local dp-bench scorecard (CPU-only, free) — score our tagged outputs vs ODL GT.

Pairs ground-truth markdown with our prediction PDFs (produced by a separate V2
regen), runs the replicated evaluator, and prints the scorecard alongside ODL's
published per-engine numbers. The V2 regen that produces the prediction PDFs is a
separate, GPU-gated step (run a 1-2 doc pilot first — see project memory).

Usage:
  PYTHONPATH=. python scratch/run_dpbench.py \
      --gt-dir   ~/benchmarks/opendataloader-bench/ground-truth/markdown \
      --pred-dir output_dpbench \
      [--odl-repo ~/benchmarks/opendataloader-bench] [--doc-ids id1 id2] [--out card.json]
"""
import argparse
import json
from pathlib import Path

from tagger.benchmark.dpbench.harness import run_dpbench
from tagger.benchmark.dpbench.report import format_scorecard, load_published


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--odl-repo", default=None,
                    help="ODL clone to refresh published numbers from prediction/*/evaluation.json")
    ap.add_argument("--doc-ids", nargs="*", default=None)
    ap.add_argument("--out", default=None, help="optional path to write the full JSON result")
    args = ap.parse_args()

    result = run_dpbench(args.gt_dir, args.pred_dir, doc_ids=args.doc_ids)
    published = load_published(args.odl_repo) if args.odl_repo else None
    print(format_scorecard(result.aggregate, published))

    if args.out:
        Path(args.out).write_text(json.dumps({
            "aggregate": result.aggregate,
            "documents": [d.to_json() for d in result.documents],
        }, indent=2, ensure_ascii=False))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
