"""dp-bench scorecard: our numbers alongside ODL's published per-engine numbers.

PUBLISHED is a dated snapshot of opendataloader-bench's prediction/*/evaluation.json
(read 2026-05-29). load_published() refreshes it from a local ODL repo clone if the
prediction/ dir is present. The deterministic ODL "Fast" engine is the bar our table
(TEDS) number is read against for the grid-topology initiative.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

# Snapshot of opendataloader-bench prediction/*/evaluation.json metrics.score
# (overall/nid/teds/teds_s/mhs). Read 2026-05-29. Refresh via load_published().
PUBLISHED: Dict[str, Dict[str, float]] = {
    "opendataloader-hybrid": {"overall": 0.907, "nid": 0.934, "teds": 0.928, "teds_s": 0.945, "mhs": 0.821},
    "docling":               {"overall": 0.882, "nid": 0.898, "teds": 0.887, "teds_s": 0.901, "mhs": 0.824},
    "marker":                {"overall": 0.861, "nid": 0.890, "teds": 0.808, "teds_s": 0.834, "mhs": 0.796},
    "mineru":                {"overall": 0.831, "nid": 0.857, "teds": 0.873, "teds_s": 0.904, "mhs": 0.743},
    "opendataloader (Fast)": {"overall": 0.831, "nid": 0.902, "teds": 0.489, "teds_s": 0.513, "mhs": 0.739},
    "markitdown":            {"overall": 0.589, "nid": 0.844, "teds": 0.273, "teds_s": 0.328, "mhs": 0.000},
}

_ENGINE_DIR_LABELS = {
    "opendataloader": "opendataloader (Fast)",
    "opendataloader-hybrid": "opendataloader-hybrid",
}
_COLS = ["overall", "nid", "nid_s", "teds", "teds_s", "mhs", "mhs_s"]


def load_published(odl_repo: str | Path) -> Dict[str, Dict[str, float]]:
    """Refresh published numbers from a local ODL clone's prediction/*/evaluation.json."""
    pred_root = Path(odl_repo) / "prediction"
    out: Dict[str, Dict[str, float]] = {}
    if not pred_root.is_dir():
        return dict(PUBLISHED)
    for eval_path in sorted(pred_root.glob("*/evaluation.json")):
        try:
            score = json.loads(eval_path.read_text())["metrics"]["score"]
        except (json.JSONDecodeError, OSError, KeyError):
            continue
        label = _ENGINE_DIR_LABELS.get(eval_path.parent.name, eval_path.parent.name)
        out[label] = {
            "overall": score.get("overall_mean"), "nid": score.get("nid_mean"),
            "teds": score.get("teds_mean"), "teds_s": score.get("teds_s_mean"),
            "mhs": score.get("mhs_mean"),
        }
    return out or dict(PUBLISHED)


def _fmt(v: Optional[float]) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) else "  -  "


def format_scorecard(
    aggregate: Dict[str, Any],
    published: Optional[Dict[str, Dict[str, float]]] = None,
    our_label: str = "auto-tagger (V2)",
) -> str:
    """Render a markdown table: our aggregate row + the published reference rows."""
    published = published if published is not None else PUBLISHED
    score = aggregate.get("score", {})
    ours = {
        "overall": score.get("overall_mean"), "nid": score.get("nid_mean"),
        "nid_s": score.get("nid_s_mean"), "teds": score.get("teds_mean"),
        "teds_s": score.get("teds_s_mean"), "mhs": score.get("mhs_mean"),
        "mhs_s": score.get("mhs_s_mean"),
    }

    header = "| engine | " + " | ".join(_COLS) + " |"
    sep = "|" + "---|" * (len(_COLS) + 1)
    rows = [header, sep]
    # our row first (highlighted)
    rows.append("| **" + our_label + "** | " + " | ".join(_fmt(ours.get(c)) for c in _COLS) + " |")
    # reference engines, sorted by overall desc
    for name, m in sorted(published.items(), key=lambda kv: kv[1].get("overall") or 0, reverse=True):
        rows.append("| " + name + " | " + " | ".join(_fmt(m.get(c)) for c in _COLS) + " |")

    n = aggregate.get("document_count", len(aggregate.get("documents", []) or []))
    footer = (
        f"\n_n={n} docs scored; tables in {aggregate.get('teds_count', 0)}, "
        f"headings in {aggregate.get('mhs_count', 0)}; "
        f"missing predictions {aggregate.get('missing_predictions', 0)}._\n"
        "_Reference numbers: opendataloader-bench published snapshot (2026-05-29). "
        "ODL 'Fast' = the deterministic table bar (TEDS 0.489)._"
    )
    return "\n".join(rows) + "\n" + footer


__all__ = ["PUBLISHED", "load_published", "format_scorecard"]
