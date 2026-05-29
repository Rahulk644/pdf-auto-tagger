"""CPU-native dp-bench extractor (NO MinerU / NO GPU) — proves the benchmark metrics
can be tackled at CPU level by replacing the GPU layout VLM with CPU primitives:
pdfplumber lattice tables, font-size/bold heading detection, XY-cut reading order.
Scored directly against GT with the dp-bench scorers (all CPU).

Full-corpus result (199 docs) vs the current GPU pipeline (MinerU), with PRINCIPLED
heading detection (no recall-gaming):
  metric    CPU-native   pipeline
  overall   0.747        0.747     (tie — CPU matches the GPU pipeline)
  NID       0.853        0.828     (CPU wins)
  TEDS      0.427        0.205     (CPU wins ~2x — tables don't need the GPU)
  MHS       0.571        0.681     (pipeline wins — heading semantics favor the VLM)

NB: a bold-at-body-size heading rule scored MHS 0.581 / overall 0.750 but OVER-DETECTED
(305 vs 192 GT headings) — it gamed MHS's recall-dominance. Dropped for the principled
rule below (size-tier OR bold-that-starts-a-block), which is honest and precise.

Run:  PYTHONPATH=. .venv3/bin/python scratch/cpu_extract.py [N]
"""
import json
import re
import statistics
import sys
from pathlib import Path

import pdfplumber

from tagger.benchmark.dpbench.score import score_document

GT = Path.home() / "benchmarks/opendataloader-bench/ground-truth/markdown"
SRC = Path.home() / "benchmarks/opendataloader-bench/pdfs"


# ---- XY-cut reading order (largest-clean-gap; handles header+columns) ----
def _gaps(intervals):
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
    order = []

    def rec(idx):
        if len(idx) <= 1:
            order.extend(idx)
            return
        xg = _gaps([(boxes[i][0], boxes[i][2]) for i in idx])
        yg = _gaps([(boxes[i][1], boxes[i][3]) for i in idx])
        bx = max(xg, default=(0.0, None))
        by = max(yg, default=(0.0, None))
        if bx[0] <= 0 and by[0] <= 0:
            order.extend(sorted(idx, key=lambda i: (boxes[i][1], boxes[i][0])))
            return
        if by[0] >= bx[0]:
            c = by[1]
            rec([i for i in idx if (boxes[i][1] + boxes[i][3]) / 2 < c])
            rec([i for i in idx if (boxes[i][1] + boxes[i][3]) / 2 >= c])
        else:
            c = bx[1]
            rec([i for i in idx if (boxes[i][0] + boxes[i][2]) / 2 < c])
            rec([i for i in idx if (boxes[i][0] + boxes[i][2]) / 2 >= c])

    rec(list(range(len(boxes))))
    return order


def _line_size(ln):
    s = [c.get("size") for c in ln.get("chars", []) if c.get("size")]
    return statistics.median(s) if s else 0.0


def _line_bold(ln):
    fonts = [c.get("fontname", "") for c in ln.get("chars", [])]
    if not fonts:
        return False
    bold = sum(1 for f in fonts if any(k in f.lower() for k in ("bold", "black", "heavy", "semibold")))
    return bold >= 0.6 * len(fonts)


def _inside(box, tboxes):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return any(t[0] <= cx <= t[2] and t[1] <= cy <= t[3] for t in tboxes)


def _tables(page, lattice_only=True):
    """Lattice (lines) tables only — text-strategy find_tables hallucinates grids from
    prose and craters NID. Accept a lines table only if it's a genuine ruled grid."""
    strategies = (("lines", "lines"),) if lattice_only else (("lines", "lines"), ("text", "text"))
    for vstrat, hstrat in strategies:
        out = []
        for t in page.find_tables(table_settings={"vertical_strategy": vstrat, "horizontal_strategy": hstrat}):
            rows = t.extract() or []
            ncells = sum(1 for r in rows for c in r if c and str(c).strip())
            if vstrat == "lines":
                ruled = [c for c in (t.cells or []) if c]
                if not (t.rows and len(t.rows) >= 2 and len(ruled) >= 4):
                    continue
            elif ncells < 4:
                continue
            html = "<table>" + "".join(
                "<tr>" + "".join(f"<td>{(c or '').strip()}</td>" for c in r) + "</tr>"
                for r in rows) + "</table>"
            out.append((t.bbox, html))
        if out:
            return out
    return []


def _body_size(sizes):
    """Body font = the most common rounded size (mode), the dominant text size."""
    rounded = [round(s * 2) / 2 for s in sizes if s]
    return statistics.mode(rounded) if rounded else 0.0


def cpu_markdown(pdf_path, big_ratio=1.15, bold_ratio=1.00, max_len=100, lattice_only=True):
    parts = []
    n_head = 0
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            blocks = []  # (bbox, kind, content)
            tboxes = []
            for bbox, html in _tables(page, lattice_only):
                blocks.append((bbox, "table", html))
                tboxes.append(bbox)
            lines = sorted(page.extract_text_lines(), key=lambda l: (l["top"], l["x0"]))
            body = _body_size([_line_size(ln) for ln in lines
                               if not _inside((ln["x0"], ln["top"], ln["x1"], ln["bottom"]), tboxes)])
            prev_bottom = None
            for ln in lines:
                lb = (ln["x0"], ln["top"], ln["x1"], ln["bottom"])
                gap_above = (ln["top"] - prev_bottom) if prev_bottom is not None else 1e9
                prev_bottom = ln["bottom"]
                if _inside(lb, tboxes):
                    continue
                sz = _line_size(ln)
                txt = ln["text"].strip()
                # PRINCIPLED heading rule (precision over recall; NOT the recall-gaming
                # "bold-at-body-size" optimum that over-detected 305 vs 192 GT):
                #   - clearly larger than body (a real heading size tier), OR
                #   - bold AND >= body AND it STARTS A BLOCK (gap above) — a real heading
                #     begins a section; inline/body-size bold EMPHASIS has no gap, so it's
                #     correctly excluded. Plus: short, no trailing sentence punct, non-numeric.
                short = 0 < len(txt) <= max_len
                no_end = bool(txt) and txt[-1] not in ".,;:"
                not_num = not re.match(r"^[\d\s.,()%/–-]+$", txt)
                big = bool(body) and sz >= body * big_ratio
                bold_block = (bool(body) and _line_bold(ln) and sz >= body * bold_ratio
                              and gap_above >= body * 0.6)
                is_h = bool((big or bold_block) and short and no_end and not_num)
                if is_h:
                    n_head += 1
                blocks.append((lb, "heading" if is_h else "text", ln["text"]))
            if not blocks:
                continue
            for i in xycut_order([b[0] for b in blocks]):
                _, kind, content = blocks[i]
                parts.append(f"# {content}" if kind == "heading" else content)
    return "\n\n".join(parts), n_head


_GTH = re.compile(r"^#{1,6}\s", re.M)


def _mean(items, k):
    vals = [getattr(x, k) for x in items]
    vals = [v for v in vals if v is not None]
    return statistics.mean(vals) if vals else 0.0


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 199
    card = {d["document_id"]: d["scores"] for d in json.load(open("/tmp/dpbench_card.json"))["documents"]}
    ids = sorted(p.stem for p in SRC.glob("*.pdf"))[:n]
    docs = [(did, (GT / f"{did}.md").read_text()) for did in ids if (GT / f"{did}.md").exists()]
    gt_heads = sum(len(_GTH.findall(gt)) for _, gt in docs)

    pipe = [type("S", (), card[d]) for d, _ in docs if d in card]
    print(f"docs={len(docs)}  GT headings total={gt_heads}")
    print(f"pipeline (MinerU): overall {_mean(pipe,'overall'):.3f} nid {_mean(pipe,'nid'):.3f} "
          f"teds {_mean(pipe,'teds'):.3f} mhs {_mean(pipe,'mhs'):.3f}")

    scores, nh = [], 0
    for did, gt in docs:
        try:
            md, h = cpu_markdown(str(SRC / f"{did}.pdf"))
        except Exception:
            continue
        nh += h
        scores.append(score_document(did, gt, md))
    print(f"\nCPU-native: ourHeads={nh}")
    print(f"   overall {_mean(scores,'overall'):.3f}  nid {_mean(scores,'nid'):.3f}  "
          f"teds {_mean(scores,'teds'):.3f}  mhs {_mean(scores,'mhs'):.3f}")


if __name__ == "__main__":
    main()
