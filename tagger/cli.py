"""
CLI entry point for the PDF auto-tagger.

Usage:
    python -m tagger.cli input.pdf -o output.pdf --report report.json
    python -m tagger.cli input.pdf --classify-only
    python -m tagger.cli --serve  # Start Flask API
"""

from __future__ import annotations

import argparse
import logging
import sys


def main():
    parser = argparse.ArgumentParser(
        description="PDF Auto-Tagger — local-first pipeline for tagging PDFs",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # --- tag command ---
    tag_parser = subparsers.add_parser("tag", help="Run full auto-tagging pipeline")
    tag_parser.add_argument("input", help="Input PDF file path")
    tag_parser.add_argument("-o", "--output", help="Output tagged PDF path")
    tag_parser.add_argument("--report", help="Output JSON report path")
    tag_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    # --- classify command ---
    classify_parser = subparsers.add_parser("classify", help="Classify pages only (Stage 0)")
    classify_parser.add_argument("input", help="Input PDF file path")
    classify_parser.add_argument("-v", "--verbose", action="store_true")

    # --- extract command ---
    extract_parser = subparsers.add_parser("extract", help="Extract + merge text (Stages 0-2)")
    extract_parser.add_argument("input", help="Input PDF file path")
    extract_parser.add_argument("--report", help="Output JSON report path")
    extract_parser.add_argument("-v", "--verbose", action="store_true")

    # --- serve command ---
    serve_parser = subparsers.add_parser("serve", help="Start Flask API server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    serve_parser.add_argument("--port", type=int, default=5002, help="Port to bind")
    serve_parser.add_argument("--debug", action="store_true", help="Debug mode")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Configure logging
    log_level = logging.DEBUG if getattr(args, "verbose", False) else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command == "tag":
        from tagger.pipeline import AutoTaggerPipeline

        pipeline = AutoTaggerPipeline()
        report = pipeline.run(
            input_pdf=args.input,
            output_pdf=args.output,
            report_path=args.report,
        )

        # Print summary
        summary = report.get("summary", {})
        print(f"\n{'='*50}")
        print(f"  Total elements: {summary.get('total_elements', 0)}")
        print(f"  Needs review:   {summary.get('needs_review', 0)}")
        print(f"  Review rate:    {summary.get('review_rate_percent', 0)}%")
        print(f"  Time:           {summary.get('total_time_seconds', 0)}s")
        print(f"{'='*50}")

    elif args.command == "classify":
        from tagger.stage0_classifier.page_classifier import classify_pages

        classifications = classify_pages(args.input)

        print(f"\nPage classifications for: {args.input}")
        print(f"{'Page':<6} {'Type':<10} {'Conf':<6} {'Chars':<8} {'Img%':<8} {'Unicode%':<10}")
        print("-" * 50)
        for c in classifications:
            print(
                f"{c.page_num:<6} {c.page_type.value:<10} {c.confidence:<6.2f} "
                f"{c.char_count:<8} {c.image_coverage:<8.2f} {c.unicode_validity:<10.3f}"
            )

    elif args.command == "extract":
        import json
        from tagger.stage0_classifier.page_classifier import classify_pages
        from tagger.stage1_extraction.native_extractor import extract_native_pages
        from tagger.stage2_merger.text_merger import merge_page_elements

        classifications = classify_pages(args.input)
        raw = extract_native_pages(args.input, classifications)

        total_chars = sum(len(v) for v in raw.values())
        total_merged = 0

        for page_num, chars in raw.items():
            merged = merge_page_elements(chars, page_num)
            total_merged += len(merged)

            print(f"\nPage {page_num} ({len(chars)} chars → {len(merged)} elements):")
            for el in merged[:10]:  # Show first 10
                print(f"  [{el.font_size or '?'}pt] {el.text[:80]}")
            if len(merged) > 10:
                print(f"  ... and {len(merged) - 10} more")

        print(f"\nTotal: {total_chars} chars → {total_merged} merged elements")

    elif args.command == "serve":
        from tagger.api import create_app

        app = create_app()
        print(f"\n🏷️  PDF Auto-Tagger API starting on http://{args.host}:{args.port}")
        app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
