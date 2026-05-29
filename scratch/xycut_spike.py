"""XY-Cut++ spike (CPU/no-GPU): is a geometric recursive projection-cut a viable
zero-compute reading-order primitive?

Context: Stage 10 builds the struct tree in MinerU's incoming order; an earlier
naive (top, left) geometric re-sort was REMOVED because it interleaves multi-column
text. XY-Cut++ is the smart geometric alternative. This spike implements the core
recursive XY-cut (largest-clean-gap selection — handles header+columns) and asks:
  1. (correctness) does it order synthetic multi-column layouts right where (top,left) fails?
  2. (value) on real dp-bench docs, does it beat the naive (top,left) order on NID?

Run:  PYTHONPATH=. .venv3/bin/python scratch/xycut_spike.py
"""
import re
from pathlib import Path

from rapidfuzz import fuzz


# ---------------------------------------------------------------- algorithm
def _gaps(intervals):
    """Maximal empty gaps in a 1-D projection: [(gap_size, cut_coord), ...]."""
    ints = sorted(intervals)
    merged = [list(ints[0])]
    for lo, hi in ints[1:]:
        if lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [(merged[i + 1][0] - merged[i][1], (merged[i][1] + merged[i + 1][0]) / 2.0)
            for i in range(len(merged) - 1)]


def xycut_order(boxes):
    """Reading order via recursive XY-cut. boxes: list of (x0, y0, x1, y1) with
    y growing downward. Returns indices in reading order. At each node cut along
    the axis with the LARGEST clean gap (Y-gap -> top then bottom; X-gap -> left
    then right); ties prefer horizontal (top-to-bottom). No gap on either axis ->
    fallback sort by (top, left)."""
    order = []

    def rec(idxs):
        if len(idxs) <= 1:
            order.extend(idxs)
            return
        xg = _gaps([(boxes[i][0], boxes[i][2]) for i in idxs])
        yg = _gaps([(boxes[i][1], boxes[i][3]) for i in idxs])
        bx = max(xg, default=(0.0, None))
        by = max(yg, default=(0.0, None))
        if bx[0] <= 0 and by[0] <= 0:
            order.extend(sorted(idxs, key=lambda i: (boxes[i][1], boxes[i][0])))
            return
        if by[0] >= bx[0]:                      # horizontal cut: top, then bottom
            c = by[1]
            rec([i for i in idxs if (boxes[i][1] + boxes[i][3]) / 2 < c])
            rec([i for i in idxs if (boxes[i][1] + boxes[i][3]) / 2 >= c])
        else:                                   # vertical cut: left, then right
            c = bx[1]
            rec([i for i in idxs if (boxes[i][0] + boxes[i][2]) / 2 < c])
            rec([i for i in idxs if (boxes[i][0] + boxes[i][2]) / 2 >= c])

    rec(list(range(len(boxes))))
    return order


def naive_order(boxes):
    return sorted(range(len(boxes)), key=lambda i: (boxes[i][1], boxes[i][0]))


# ---------------------------------------------------------------- correctness
def _synthetic_tests():
    # single column: 3 stacked lines
    sc = [(0, 0, 100, 10), (0, 20, 100, 30), (0, 40, 100, 50)]
    assert xycut_order(sc) == [0, 1, 2], xycut_order(sc)

    # two columns (no header): left L0,L1,L2 ; right R0,R1,R2 (interleaved in y)
    #   idx: 0=L0 1=R0 2=L1 3=R1 4=L2 5=R2
    tc = [(0, 0, 90, 10), (110, 0, 200, 10),
          (0, 20, 90, 30), (110, 20, 200, 30),
          (0, 40, 90, 50), (110, 40, 200, 50)]
    # correct reading: left col top->bottom (0,2,4) then right col (1,3,5)
    assert xycut_order(tc) == [0, 2, 4, 1, 3, 5], xycut_order(tc)
    # the naive sort interleaves the columns — this is the bug XY-cut fixes
    assert naive_order(tc) == [0, 1, 2, 3, 4, 5]

    # header + two columns: 0=header(full width) ; then 2 cols
    hc = [(0, 0, 200, 10),
          (0, 30, 90, 40), (110, 30, 200, 40),
          (0, 50, 90, 60), (110, 50, 200, 60)]
    # correct: header, then left col (1,3), then right col (2,4)
    assert xycut_order(hc) == [0, 1, 3, 2, 4], xycut_order(hc)
    print("synthetic correctness: PASS (single-col, 2-col, header+2-col)")


# ---------------------------------------------------------------- real eval
def _norm(t):
    return re.sub(r"\s+", " ", t).strip()


def _nid(pred, gt):
    return fuzz.ratio(_norm(pred), _norm(gt)) / 100.0


def _doc_text(pdf_path, orderer):
    import pdfplumber
    parts = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            lines = page.extract_text_lines()
            if not lines:
                continue
            boxes = [(ln["x0"], ln["top"], ln["x1"], ln["bottom"]) for ln in lines]
            for i in orderer(boxes):
                parts.append(lines[i]["text"])
    return "\n".join(parts)


def _real_eval(n=199):
    import json
    gt_dir = Path.home() / "benchmarks/opendataloader-bench/ground-truth/markdown"
    src = Path.home() / "benchmarks/opendataloader-bench/pdfs"
    card = {d["document_id"]: d["scores"].get("nid")
            for d in json.load(open("/tmp/dpbench_card.json"))["documents"]}
    ids = sorted(p.stem for p in src.glob("*.pdf"))[:n]
    rows = []
    for did in ids:
        gt = (gt_dir / f"{did}.md")
        if not gt.exists():
            continue
        gt_txt = gt.read_text()
        try:
            nv = _nid(_doc_text(src / f"{did}.pdf", naive_order), gt_txt)
            xc = _nid(_doc_text(src / f"{did}.pdf", xycut_order), gt_txt)
        except Exception as e:
            print(f"  {did}: {e}")
            continue
        rows.append((did, nv, xc, card.get(did)))
    import statistics
    nvs = [r[1] for r in rows]
    xcs = [r[2] for r in rows]
    wins = [r for r in rows if r[2] > r[1] + 0.02]
    loss = [r for r in rows if r[2] < r[1] - 0.02]
    print(f"\nreal eval on {len(rows)} dp-bench docs (pdfplumber lines):")
    print(f"  naive(top,left) NID mean = {statistics.mean(nvs):.3f}")
    print(f"  xycut           NID mean = {statistics.mean(xcs):.3f}")
    print(f"  xycut beats naive (>+0.02): {len(wins)} docs | worse: {len(loss)} docs")
    print("  biggest xycut wins (likely multi-column):")
    for did, nv, xc, pn in sorted(wins, key=lambda r: r[1] - r[2])[:6]:
        print(f"    {did}: naive {nv:.3f} -> xycut {xc:.3f} (+{xc-nv:.3f})  [pipeline nid {pn}]")
    if loss:
        print("  regressions:")
        for did, nv, xc, pn in sorted(loss, key=lambda r: r[2] - r[1])[:6]:
            print(f"    {did}: naive {nv:.3f} -> xycut {xc:.3f} ({xc-nv:+.3f})")


if __name__ == "__main__":
    _synthetic_tests()
    _real_eval()
