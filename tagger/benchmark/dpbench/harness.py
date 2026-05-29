"""dp-bench harness: score our tagged-PDF outputs against ODL ground-truth markdown.

Pairs each GT markdown (``{id}.md``) with our prediction PDF (``{id}.pdf``), runs
the struct-tree -> markdown adapter on the prediction, and scores with the
replicated metrics. Missing prediction -> scored against empty markdown (mirrors
ODL's _read_text("") behavior), so a doc we failed to produce still counts.
CPU-only; the prediction PDFs are produced by a separate (Modal) V2 regen.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from tagger.benchmark.dpbench.adapter import pdf_to_markdown
from tagger.benchmark.dpbench.score import (
    DocumentScores, aggregate_scores, score_document,
)


@dataclass
class DpbenchResult:
    aggregate: Dict[str, Any]
    documents: List[DocumentScores]


def run_dpbench(
    gt_dir: str | Path,
    pred_pdf_dir: str | Path,
    *,
    doc_ids: Optional[Sequence[str]] = None,
) -> DpbenchResult:
    """Score our predictions in ``pred_pdf_dir`` against GT markdown in ``gt_dir``.

    Args:
        gt_dir: directory of ``{id}.md`` ground-truth files (ODL ground-truth/markdown).
        pred_pdf_dir: directory of our tagged ``{id}.pdf`` outputs (V2 regen).
        doc_ids: optional subset of ids to score (for pilot/sample phases).
    """
    gt_dir = Path(gt_dir)
    pred_pdf_dir = Path(pred_pdf_dir)
    wanted = set(doc_ids) if doc_ids is not None else None

    documents: List[DocumentScores] = []
    for gt_path in sorted(gt_dir.glob("*.md")):
        doc_id = gt_path.stem
        if wanted is not None and doc_id not in wanted:
            continue
        gt_md = gt_path.read_text(encoding="utf-8")
        pred_pdf = pred_pdf_dir / f"{doc_id}.pdf"
        if pred_pdf.is_file():
            try:
                pred_md = pdf_to_markdown(str(pred_pdf))
                available = True
            except Exception:
                pred_md, available = "", False
        else:
            pred_md, available = "", False
        documents.append(score_document(doc_id, gt_md, pred_md, available))

    return DpbenchResult(aggregate=aggregate_scores(documents), documents=documents)


__all__ = ["DpbenchResult", "run_dpbench"]
