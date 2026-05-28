"""Layout-accuracy harness — phase 2b/3: token tag-agreement + figure-IoU coverage.

Builds on the phase-2a normalizer (scratch/layout_tokens.py). Because PREP's tagged
PDF and ours tag the SAME source document, tokens align near-exactly by position,
so we can compare the tag each side assigned token-by-token:

  - Tag-agreement confusion matrix (PREP tag x our tag) over positionally-aligned
    word tokens → which tags we get right and where we systematically diverge
    (e.g. real content we hide as Artifact, headings we flatten to P).
  - Figure-IoU coverage: of PREP's Figure-tagged images, how many we also tag
    Figure (IoU-matched), vs leave as Artifact.

PREP is a STRONG REFERENCE, not gold — read disagreements as "where we diverge",
adjudicate with the 4-page gold subset (see [[project-layout-harness]]).

Usage:
  .venv3/bin/python scratch/layout_agreement.py                 # PREP corpus
  .venv3/bin/python scratch/layout_agreement.py PREP.pdf OURS.pdf
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pdfplumber
import pikepdf

from layout_distribution import DEFAULT_PAIRS, _OURS, _PREP
from layout_tokens import extract_tokens, mcid_tag_map, strip_tag


def _center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def align(prep_tokens, our_tokens, tol=3.0):
    """Positionally align tokens (same source → same positions). Yields (prep_tag,
    our_tag) for matched pairs; returns also unaligned counts both directions."""
    by_page = defaultdict(list)
    for t in prep_tokens:
        by_page[t["page"]].append(t)
    used = set()
    pairs = []
    our_unaligned = 0
    for ot in our_tokens:
        oc = _center(ot["bbox"])
        best, best_d = None, tol
        for i, pt in enumerate(by_page[ot["page"]]):
            if (ot["page"], i) in used:
                continue
            pc = _center(pt["bbox"])
            d = abs(oc[0] - pc[0]) + abs(oc[1] - pc[1])
            if d < best_d:
                best_d, best = d, i
        if best is None:
            our_unaligned += 1
        else:
            used.add((ot["page"], best))
            pairs.append((by_page[ot["page"]][best]["tag"], ot["tag"]))
    prep_total = len(prep_tokens)
    prep_unaligned = prep_total - len(pairs)
    return pairs, our_unaligned, prep_unaligned


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def figure_images(pdf_path):
    """(page, bbox) of every image tagged /Figure in the struct tree."""
    with pikepdf.open(pdf_path) as pk:
        tmap = mcid_tag_map(pk)
    out = []
    with pdfplumber.open(pdf_path) as pl:
        for pidx, page in enumerate(pl.pages):
            for im in page.images:
                m = im.get("mcid")
                tag = tmap.get((pidx, int(m))) if m is not None else None
                if strip_tag(tag) == "Figure":
                    out.append((pidx, (im["x0"], im["top"], im["x1"], im["bottom"])))
    return out


def figure_coverage(prep_pdf, our_pdf, thr=0.5):
    """Fraction of PREP figure-images IoU-matched by an our figure-image."""
    pf, of = figure_images(prep_pdf), figure_images(our_pdf)
    covered = 0
    for pp, pb in pf:
        if any(op == pp and _iou(pb, ob) >= thr for op, ob in of):
            covered += 1
    return covered, len(pf), len(of)


def report(pairs):
    cm = Counter(pairs)
    prep_tags = sorted({p for p, _ in pairs})
    our_tags = sorted({o for _, o in pairs})
    total = len(pairs)
    agree = sum(n for (p, o), n in cm.items() if p == o)
    print(f"  aligned tokens: {total}   agreement: {agree/total:.1%}" if total else "  no aligned tokens")
    print(f"  {'PREP|ours':12s}" + "".join(f"{o:>9s}" for o in our_tags) + f"{'recall':>9s}")
    for p in prep_tags:
        row = sum(cm.get((p, o), 0) for o in our_tags)
        cells = "".join(f"{cm.get((p, o), 0):9d}" for o in our_tags)
        rec = cm.get((p, p), 0) / row if row else 0.0
        print(f"  {p:12s}{cells}{rec:9.0%}")


def run(pairs):
    all_pairs = []
    for label, prep_path, our_path in pairs:
        if not (Path(prep_path).exists() and Path(our_path).exists()):
            continue
        pp = extract_tokens(prep_path)
        op = extract_tokens(our_path)
        matched, ou, pu = align(pp, op)
        all_pairs += matched
        cov, npf, nof = figure_coverage(prep_path, our_path)
        agree = sum(1 for a, b in matched if a == b)
        print(f"\n=== {label} ===")
        print(f"  tokens PREP={len(pp)} ours={len(op)}; aligned={len(matched)} "
              f"(ours_unaligned={ou}, prep_unaligned={pu})")
        print(f"  token agreement: {agree/len(matched):.1%}" if matched else "  no tokens")
        print(f"  figure-image coverage: {cov}/{npf} PREP figures matched "
              f"(ours has {nof} figure-images)")
    print("\n=== CORPUS confusion matrix (PREP tag -> our tag) ===")
    report(all_pairs)


def main(argv):
    if len(argv) == 2:
        run([("pair", argv[0], argv[1])])
    else:
        run([(lbl, str(_PREP / pf), str(_OURS / op)) for lbl, pf, op in DEFAULT_PAIRS])


if __name__ == "__main__":
    main(sys.argv[1:])
