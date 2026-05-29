"""Per-document + aggregate dp-bench scoring (replicated from opendataloader-bench).

overall = unweighted mean of the three primary scores (nid, teds, mhs) that are
present for a doc; teds is None when the GT has no table, so table-less docs ride
on nid+mhs. Aggregate = mean of each metric across docs (Nones excluded). The
file-walking / corpus loading lives in harness.py (later phase); this module is
pure computation so it can be unit-tested against ODL's golden cases.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Any, Dict, Iterable, List, Optional

from tagger.benchmark.dpbench.heading import evaluate_heading_level
from tagger.benchmark.dpbench.reading_order import evaluate_reading_order
from tagger.benchmark.dpbench.table import evaluate_table


def _safe_mean(values: Iterable[float]) -> Optional[float]:
    values = [v for v in values if v is not None]
    return fmean(values) if values else None


@dataclass
class DocumentScores:
    document_id: str
    overall: Optional[float]
    nid: Optional[float]
    nid_s: Optional[float]
    teds: Optional[float]
    teds_s: Optional[float]
    mhs: Optional[float]
    mhs_s: Optional[float]
    prediction_available: bool = True

    def to_json(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "scores": {
                "overall": self.overall, "nid": self.nid, "nid_s": self.nid_s,
                "teds": self.teds, "teds_s": self.teds_s,
                "mhs": self.mhs, "mhs_s": self.mhs_s,
            },
            "prediction_available": self.prediction_available,
        }


def score_document(
    document_id: str, gt_markdown: str, pred_markdown: str,
    prediction_available: bool = True,
) -> DocumentScores:
    nid, nid_s = evaluate_reading_order(gt_markdown, pred_markdown)
    teds, teds_s = evaluate_table(gt_markdown, pred_markdown)
    mhs, mhs_s = evaluate_heading_level(gt_markdown, pred_markdown)
    overall = _safe_mean([nid, teds, mhs])
    return DocumentScores(
        document_id=document_id, overall=overall,
        nid=nid, nid_s=nid_s, teds=teds, teds_s=teds_s, mhs=mhs, mhs_s=mhs_s,
        prediction_available=prediction_available,
    )


def aggregate_scores(documents: List[DocumentScores]) -> Dict[str, Any]:
    return {
        "score": {
            "overall_mean": _safe_mean(d.overall for d in documents),
            "nid_mean": _safe_mean(d.nid for d in documents),
            "nid_s_mean": _safe_mean(d.nid_s for d in documents),
            "teds_mean": _safe_mean(d.teds for d in documents),
            "teds_s_mean": _safe_mean(d.teds_s for d in documents),
            "mhs_mean": _safe_mean(d.mhs for d in documents),
            "mhs_s_mean": _safe_mean(d.mhs_s for d in documents),
        },
        "nid_count": sum(1 for d in documents if d.nid is not None),
        "teds_count": sum(1 for d in documents if d.teds is not None),
        "mhs_count": sum(1 for d in documents if d.mhs is not None),
        "missing_predictions": sum(1 for d in documents if not d.prediction_available),
        "document_count": len(documents),
    }


__all__ = ["DocumentScores", "score_document", "aggregate_scores"]
