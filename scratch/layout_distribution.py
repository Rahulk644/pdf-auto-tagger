"""Layout-accuracy harness — phase 0: struct-tag distribution, ours vs PREP.

The cheapest tier of the layout-accuracy harness (the third eval axis: did each
region get the *right* tag, vs veraPDF's "is it tagged at all" and Gemma's noisy
"is the tag defensible"). A pure struct-tree walk — no token alignment, no GPU —
it tabulates struct-element tag counts for each (PREP ground-truth, our output)
pair and surfaces systematic divergences (figures dropped, headings
over-promoted, lists not built, table grids over-segmented) in seconds. Run it
as a regression guard after any structural change.

PREP is a STRONG REFERENCE, not gold: it fails veraPDF on 3/5 corpus docs and
self-reports ~82% accuracy. Read deltas as "where we systematically diverge",
NOT "our errors" — adjudicating who is right needs the hand-labeled gold subset.

Usage:
  .venv3/bin/python scratch/layout_distribution.py                  # PREP corpus
  .venv3/bin/python scratch/layout_distribution.py PREP.pdf OURS.pdf  # one pair
"""
import sys
from collections import Counter
from pathlib import Path

import pikepdf
from pikepdf import Array, Dictionary, Name

_PREP = Path("/Users/rahulkhatri/Downloads/pdf_tag_tool/PREP PDF & Reports/PREP PDFs")
_OURS = Path("/Users/rahulkhatri/Tagger/output_modal")

# (label, PREP ground-truth filename, our output path). Miramar's fresh output is
# miramar_untagged.pdf (the same-named file in output_modal is a stale old run).
DEFAULT_PAIRS = [
    ("nyvra", "nyvra-factsheet.pdf", "nyvra-factsheet.pdf"),
    ("Osteo", "Osteoarthritis.pdf", "Osteoarthritis.pdf"),
    ("Missouri", "Missouri State Epidemiological Profile July 2018.pdf",
     "Missouri State Epidemiological Profile July 2018.pdf"),
    ("Summary", "Summary of Revenues and Expenditures.pdf",
     "Summary of Revenues and Expenditures.pdf"),
    ("Miramar", "CITY OF MIRAMAR, FLORIDA.pdf", "miramar_untagged.pdf"),
]

COLUMNS = [
    "/H1", "/H2", "/H3", "/P", "/Figure", "/Table", "/TR", "/TH", "/TD",
    "/L", "/LI", "/Lbl", "/LBody", "/TOC", "/TOCI", "/Link", "/Caption", "/Annot",
]


def tag_counts(path: str | Path) -> Counter | None:
    """Count struct-element /S tags in a PDF's struct tree (None if untagged)."""
    with pikepdf.open(str(path)) as pdf:
        sr = pdf.Root.get("/StructTreeRoot")
        if sr is None:
            return None
        counts: Counter = Counter()

        def walk(node):
            if not isinstance(node, Dictionary):
                return
            if node.get("/Type") == Name.StructElem and node.get("/S") is not None:
                counts[str(node.get("/S"))] += 1
            k = node.get("/K")
            for c in (k if isinstance(k, Array) else [k] if k is not None else []):
                walk(c)

        walk(sr.get("/K"))
        return counts


def _row(label: str, src: str, counts: Counter) -> str:
    cells = " ".join(f"{counts.get(k, 0):5d}" for k in COLUMNS)
    return f"{label:9s} {src:4s} {cells}"


def compare(pairs) -> None:
    header = " ".join(f"{k.replace('/', ''):>5s}" for k in COLUMNS)
    print(f"{'doc':9s} {'src':4s} {header}")
    totals = {"PREP": Counter(), "OURS": Counter()}
    for label, prep_path, our_path in pairs:
        cp = tag_counts(prep_path) if Path(prep_path).exists() else None
        co = tag_counts(our_path) if Path(our_path).exists() else None
        if cp is not None:
            print(_row(label, "PREP", cp)); totals["PREP"].update(cp)
        if co is not None:
            print(_row(label, "OURS", co)); totals["OURS"].update(co)
        print()
    print(_row("TOTAL", "PREP", totals["PREP"]))
    print(_row("TOTAL", "OURS", totals["OURS"]))
    # Headline divergences worth watching.
    fp, fo = totals["PREP"].get("/Figure", 0), totals["OURS"].get("/Figure", 0)
    print(f"\nFigure coverage (corpus): ours {fo} vs PREP {fp}"
          f"  ({fo / fp:.0%} of reference)" if fp else "")


def main(argv) -> None:
    if len(argv) == 2:
        pairs = [("pair", argv[0], argv[1])]
    else:
        pairs = [(lbl, str(_PREP / pf), str(_OURS / op)) for lbl, pf, op in DEFAULT_PAIRS]
    compare(pairs)


if __name__ == "__main__":
    main(sys.argv[1:])
