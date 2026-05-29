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

# Snapshot of the opendataloader.org/docs/benchmark public leaderboard (read 2026-05-29,
# full 13-engine field). The leaderboard column names map to our metric keys as:
# Reading Order = nid, Table = teds, Heading = mhs. teds_s (structure-only TEDS) is only
# available for the 6 engines carried from prediction/*/evaluation.json; the leaderboard
# itself doesn't publish it, so the newer engines have no teds_s. speed = s/page (lower is
# better), license carried for the acquisition-positioning read. Refresh via load_published().
PUBLISHED: Dict[str, Dict[str, Any]] = {
    "opendataloader-hybrid": {"overall": 0.907, "nid": 0.934, "teds": 0.928, "teds_s": 0.945, "mhs": 0.821, "speed": 0.463, "license": "Apache-2.0"},
    "nutrient":              {"overall": 0.885, "nid": 0.925, "teds": 0.708, "mhs": 0.819, "speed": 0.008, "license": "Commercial"},
    "docling":               {"overall": 0.882, "nid": 0.898, "teds": 0.887, "teds_s": 0.901, "mhs": 0.824, "speed": 0.762, "license": "MIT"},
    "marker":                {"overall": 0.861, "nid": 0.890, "teds": 0.808, "teds_s": 0.834, "mhs": 0.796, "speed": 53.932, "license": "GPL-3.0"},
    "unstructured [hi_res]": {"overall": 0.841, "nid": 0.904, "teds": 0.588, "mhs": 0.749, "speed": 3.008, "license": "Apache-2.0"},
    "edgeparse":             {"overall": 0.837, "nid": 0.894, "teds": 0.717, "mhs": 0.706, "speed": 0.036, "license": "Apache-2.0"},
    "opendataloader (Fast)": {"overall": 0.831, "nid": 0.902, "teds": 0.489, "teds_s": 0.513, "mhs": 0.739, "speed": 0.015, "license": "Apache-2.0"},
    "mineru":                {"overall": 0.831, "nid": 0.857, "teds": 0.873, "teds_s": 0.904, "mhs": 0.743, "speed": 5.962, "license": "AGPL-3.0"},
    "pymupdf4llm":           {"overall": 0.732, "nid": 0.885, "teds": 0.401, "mhs": 0.412, "speed": 0.091, "license": "AGPL-3.0"},
    "unstructured":          {"overall": 0.686, "nid": 0.882, "teds": 0.000, "mhs": 0.388, "speed": 0.077, "license": "Apache-2.0"},
    "markitdown":            {"overall": 0.589, "nid": 0.844, "teds": 0.273, "teds_s": 0.328, "mhs": 0.000, "speed": 0.114, "license": "MIT"},
    "liteparse":             {"overall": 0.576, "nid": 0.866, "teds": 0.000, "mhs": 0.000, "speed": 1.061, "license": "Apache-2.0"},
}

_ENGINE_DIR_LABELS = {
    "opendataloader": "opendataloader (Fast)",
    "opendataloader-hybrid": "opendataloader-hybrid",
}
_COLS = ["overall", "nid", "nid_s", "teds", "teds_s", "mhs", "mhs_s"]
_EXTRA_COLS = ["speed", "license"]


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

    all_cols = _COLS + _EXTRA_COLS
    header = "| engine | " + " | ".join(all_cols) + " |"
    sep = "|" + "---|" * (len(all_cols) + 1)
    rows = [header, sep]

    def _cell(m: Dict[str, Any], c: str) -> str:
        v = m.get(c)
        if c == "license":
            return v if isinstance(v, str) else "  -  "
        return _fmt(v)

    # our row first (highlighted); we have no speed/license self-measurement yet
    rows.append("| **" + our_label + "** | " + " | ".join(_cell(ours, c) for c in all_cols) + " |")
    # reference engines, sorted by overall desc
    for name, m in sorted(published.items(), key=lambda kv: kv[1].get("overall") or 0, reverse=True):
        rows.append("| " + name + " | " + " | ".join(_cell(m, c) for c in all_cols) + " |")

    n = aggregate.get("document_count", len(aggregate.get("documents", []) or []))
    footer = (
        f"\n_n={n} docs scored; tables in {aggregate.get('teds_count', 0)}, "
        f"headings in {aggregate.get('mhs_count', 0)}; "
        f"missing predictions {aggregate.get('missing_predictions', 0)}._\n"
        "_Reference: opendataloader.org/docs/benchmark public leaderboard (2026-05-29, "
        "full 13-engine field). Cols nid/teds/mhs = their Reading Order/Table/Heading; "
        "speed = s/page (lower better). ODL 'Fast' = the deterministic table bar (TEDS 0.489)._"
    )
    return "\n".join(rows) + "\n" + footer


__all__ = ["PUBLISHED", "load_published", "format_scorecard"]
