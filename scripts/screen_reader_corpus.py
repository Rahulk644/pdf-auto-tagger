#!/usr/bin/env python3
"""Run the screen-reader linearizer across a corpus of tagged PDFs.

For every PDF in a directory it computes the deterministic AT "smell test" and
aggregates the issues a screen-reader user would hit — so you can sweep a whole
output folder (e.g. a benchmark regen) and see, in one table, where the tagged
output reads badly: graphics with no alt text, empty headings announced, tables
with no header row, non-descriptive links, untagged docs.

Usage:
    python scripts/screen_reader_corpus.py <dir-or-glob> [--json out.json] [--transcripts]

Exit code is non-zero if any document has issues, so it can gate a regen the way
the veraPDF gate gates conformance — except this gates the *reading experience*,
not just structural validity.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tagger.audit.screen_reader import linearize  # noqa: E402


def _inputs(arg: str) -> list[Path]:
    p = Path(arg)
    if p.is_dir():
        return sorted(p.glob("*.pdf"))
    return sorted(Path().glob(arg))


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not args:
        print("usage: screen_reader_corpus.py <dir-or-glob> [--json out] [--transcripts]",
              file=sys.stderr)
        return 2
    pdfs = _inputs(args[0])
    if not pdfs:
        print(f"no PDFs under {args[0]}", file=sys.stderr)
        return 2
    want_json = "--json" in sys.argv
    json_path = None
    if want_json:
        i = sys.argv.index("--json")
        json_path = sys.argv[i + 1] if i + 1 < len(sys.argv) else "screen_reader_corpus.json"
    dump_transcripts = "--transcripts" in sys.argv

    issue_kinds: Counter = Counter()
    docs_with_issues = 0
    per_doc = []
    for pdf in pdfs:
        t = linearize(str(pdf))
        issues = t.issues
        if issues:
            docs_with_issues += 1
        for a in issues:
            issue_kinds[a.issue] += 1
        per_doc.append({
            "doc": pdf.name,
            "announcements": len(t.announcements),
            "issues": [{"role": a.role, "issue": a.issue} for a in issues],
        })
        if dump_transcripts:
            print(f"\n===== {pdf.name} =====")
            print(t.as_text())

    print(f"\nScreen-reader corpus sweep: {len(pdfs)} docs, "
          f"{docs_with_issues} with issues ({len(pdfs) - docs_with_issues} clean)")
    if issue_kinds:
        print("\n  issue                                                     count")
        for kind, n in issue_kinds.most_common():
            print(f"  {kind:<55} {n:>5}")
        print("\n  worst docs:")
        for d in sorted(per_doc, key=lambda d: -len(d["issues"]))[:10]:
            if d["issues"]:
                print(f"    {d['doc']:<40} {len(d['issues'])} issue(s)")
    else:
        print("  no screen-reader issues across the corpus")

    if json_path:
        Path(json_path).write_text(json.dumps({
            "summary": {"docs": len(pdfs), "docs_with_issues": docs_with_issues,
                        "issue_kinds": dict(issue_kinds)},
            "documents": per_doc,
        }, indent=2))
        print(f"\nwrote {json_path}")

    return 1 if docs_with_issues else 0


if __name__ == "__main__":
    sys.exit(main())
