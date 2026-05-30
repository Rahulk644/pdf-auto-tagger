#!/usr/bin/env python3
"""veraPDF PDF/UA-1 conformance gate.

Runs the auto-tagger on a set of fixture PDFs and pipes every tagged output
through the veraPDF CLI (PDF/UA-1 profile). Exits NON-ZERO if any output is
not compliant. This is the deterministic line the "PDF/UA compliant" claim
rests on: it does not matter how good the tagging looks — if veraPDF says the
structure tree fails, the gate fails and CI goes red.

Usage:
    TAGGER_LAYOUT_BACKEND=cpu python scripts/verapdf_gate.py [input.pdf ...]

With no arguments it tags every PDF in tests/fixtures/. Each input is run
through the full pipeline to a temp file, then validated.

veraPDF discovery (first match wins):
    $VERAPDF_CLI  →  `verapdf` on PATH  →  ~/verapdf/verapdf
If veraPDF is not found the gate SKIPS (exit 0) locally, but set
VERAPDF_REQUIRED=1 (CI does) to turn "not found" into a hard failure.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def find_verapdf() -> str | None:
    cand = os.environ.get("VERAPDF_CLI")
    if cand and Path(cand).is_file():
        return cand
    onpath = shutil.which("verapdf")
    if onpath:
        return onpath
    home = Path.home() / "verapdf" / "verapdf"
    if home.is_file():
        return str(home)
    return None


def validate_ua1(verapdf: str, pdf: Path) -> tuple[bool, int, str]:
    """Return (is_compliant, failed_checks, detail). Parses the veraPDF XML."""
    try:
        out = subprocess.run(
            [verapdf, "-f", "ua1", str(pdf)],
            capture_output=True, text=True, timeout=300,
        ).stdout
    except (subprocess.TimeoutExpired, OSError) as e:
        return False, -1, f"veraPDF invocation failed: {e}"
    # Lightweight XML scrape — avoids a parser dependency for one attribute.
    compliant = 'isCompliant="true"' in out
    failed = 0
    import re
    m = re.search(r'failedChecks="(\d+)"', out)
    if m:
        failed = int(m.group(1))
    return compliant, failed, "" if compliant else f"failedChecks={failed}"


def tag(pdf: Path, out_dir: Path) -> Path:
    """Run the full pipeline on `pdf`, returning the tagged output path."""
    from tagger.pipeline import AutoTaggerPipeline
    out_pdf = out_dir / pdf.name
    AutoTaggerPipeline().run(
        input_pdf=str(pdf), output_pdf=str(out_pdf),
        report_path=str(out_pdf.with_suffix(".json")))
    return out_pdf


def main() -> int:
    verapdf = find_verapdf()
    if verapdf is None:
        msg = "veraPDF CLI not found (set $VERAPDF_CLI, add to PATH, or ~/verapdf/verapdf)"
        if os.environ.get("VERAPDF_REQUIRED") == "1":
            print(f"FAIL: {msg}", file=sys.stderr)
            return 2
        print(f"SKIP: {msg}")
        return 0

    inputs = [Path(a) for a in sys.argv[1:]]
    if not inputs:
        inputs = sorted((REPO / "tests" / "fixtures" / "conformance").glob("*.pdf"))
    if not inputs:
        print("FAIL: no input PDFs", file=sys.stderr)
        return 2

    print(f"veraPDF: {verapdf}")
    print(f"gating {len(inputs)} document(s) against PDF/UA-1\n")
    failures = 0
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td)
        for pdf in inputs:
            try:
                tagged = tag(pdf, out_dir)
            except Exception as e:
                print(f"  [ERROR] {pdf.name}: pipeline failed: {e}")
                failures += 1
                continue
            ok, failed, detail = validate_ua1(verapdf, tagged)
            mark = "PASS" if ok else "FAIL"
            print(f"  [{mark}] {pdf.name}  {detail}".rstrip())
            if not ok:
                failures += 1

    print()
    if failures:
        print(f"GATE FAILED: {failures}/{len(inputs)} document(s) not PDF/UA-1 compliant",
              file=sys.stderr)
        return 1
    print(f"GATE PASSED: all {len(inputs)} document(s) PDF/UA-1 compliant")
    return 0


if __name__ == "__main__":
    sys.exit(main())
