"""Unit 6 — scorecard aggregation + formatting.

Per criterion: checker-agreement (vs expert), remediation-rate (if remediation
results supplied), an addressed/attempted/NP/CT label breakdown (so sample-size
context is preserved), and an Adobe-checker triangulation (does our verdict agree
with the expert more/less than adobe6_compliance). Header carries the three-axis
framing so a reader never confuses WCAG accessibility with ISO conformance.
"""
from __future__ import annotations

from tagger.benchmark.harness import Result, checker_agreement, remediation_rate

THREE_AXES = (
    "veraPDF = ISO/PDF-UA conformance | "
    "PDF-A-B = WCAG/expert accessibility (this scorecard) | "
    "DocLayNet = tag accuracy"
)


def _label_breakdown(rs: list[Result], criterion: str) -> dict:
    rs = [r for r in rs if r.task.criterion == criterion]
    pf = [r for r in rs if r.task.expert_label in ("passed", "failed")]
    return {
        "total": len(rs),
        "addressed": sum(1 for r in pf if r.our_verdict.status in ("passed", "failed")),
        "attempted": sum(1 for r in pf if r.our_verdict.status == "cannot_derive"),
        "not_present": sum(1 for r in rs if r.task.expert_label == "not_present"),
        "cannot_tell": sum(1 for r in rs if r.task.expert_label == "cannot_tell"),
    }


def _adobe_triangulation(rs: list[Result], criterion: str) -> dict:
    comp = [r for r in rs
            if r.task.criterion == criterion
            and r.task.expert_label in ("passed", "failed")
            and r.task.adobe6_compliance is not None
            and r.our_verdict.status in ("passed", "failed")]
    ours = sum(1 for r in comp if r.our_verdict.status == r.task.expert_label)
    adobe = sum(1 for r in comp
                if ("passed" if r.task.adobe6_compliance else "failed") == r.task.expert_label)
    return {"comparable": len(comp), "ours_agree": ours, "adobe6_agree": adobe}


def build_scorecard(
    checker_results: list[Result],
    remediation_results: list[Result] | None = None,
) -> dict:
    criteria = sorted({r.task.criterion for r in checker_results})
    per: dict = {}
    for c in criteria:
        entry = {
            "checker_agreement": checker_agreement(checker_results, c),
            "labels": _label_breakdown(checker_results, c),
            "adobe6": _adobe_triangulation(checker_results, c),
        }
        if remediation_results is not None:
            entry["remediation_rate"] = remediation_rate(remediation_results, c)
        per[c] = entry
    return {"axes": THREE_AXES, "criteria": per}


def _pct(x):
    return "  n/a" if x is None else f"{x:5.0%}"


def format_scorecard(scorecard: dict) -> str:
    lines = ["PDF-Accessibility-Benchmark scorecard",
             f"  axes: {scorecard['axes']}", ""]
    hdr = f"  {'criterion':24s} {'checkAgr':>9s} {'remed':>7s} {'addr':>5s} {'att':>4s} {'NP':>3s} {'CT':>3s}  adobe(us/ad)"
    lines.append(hdr)
    for c, e in scorecard["criteria"].items():
        ca = e["checker_agreement"]
        lb = e["labels"]
        ad = e["adobe6"]
        rem = e.get("remediation_rate", {}).get("rate") if "remediation_rate" in e else None
        lines.append(
            f"  {c:24s} {_pct(ca['agreement'])}({ca['comparable']:>2d}) "
            f"{_pct(rem)} {lb['addressed']:>5d} {lb['attempted']:>4d} "
            f"{lb['not_present']:>3d} {lb['cannot_tell']:>3d}   "
            f"{ad['ours_agree']}/{ad['adobe6_agree']} of {ad['comparable']}"
        )
    return "\n".join(lines)
