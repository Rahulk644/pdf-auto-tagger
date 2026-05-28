"""Unit 5 — the eval loop + the two metric primitives, with the hard-assert.

Two framings, kept structurally un-mixable via Result.framing:
  - CHECKER: derive a verdict on the ORIGINAL doc and compare to the expert label
    (does our judgment match the expert?). expert_label is a comparison target here.
  - REMEDIATION: derive a verdict on our STRIP+V2 OUTPUT; the expert `failed` label
    is the INPUT CONDITION ONLY (selects the real failures) — it is NEVER compared
    to the output verdict as "agreement". remediation_rate asserts its inputs are
    remediation-framed and only ever uses expert_label as a `== "failed"` selector.

This makes the metric semantics impossible to silently corrupt in later refactors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from tagger.benchmark.loader import DocTask
from tagger.benchmark.verdicts.base import CannotDeriveReason, Verdict, derive_verdict

CHECKER = "checker"
REMEDIATION = "remediation"


@dataclass
class Result:
    task: DocTask
    framing: str           # CHECKER | REMEDIATION
    our_verdict: Verdict


def run_checker(tasks: Iterable[DocTask], criteria: set | None = None) -> list[Result]:
    """Derive verdicts on the ORIGINAL docs (CPU, no pipeline). Checker framing."""
    out = []
    for t in tasks:
        if criteria is not None and t.criterion not in criteria:
            continue
        out.append(Result(t, CHECKER, derive_verdict(t.pdf_path, t.criterion)))
    return out


def run_remediation(
    tasks: Iterable[DocTask],
    output_for: Callable[[DocTask], str | None],
    criteria: set | None = None,
) -> list[Result]:
    """Derive verdicts on our STRIP+V2 OUTPUT (output_for(task) -> path or None)."""
    out = []
    for t in tasks:
        if criteria is not None and t.criterion not in criteria:
            continue
        op = output_for(t)
        if op is None:
            v = Verdict.cannot(CannotDeriveReason.PipelineError, note="no remediated output")
        else:
            v = derive_verdict(op, t.criterion)
        out.append(Result(t, REMEDIATION, v))
    return out


def remediation_rate(results: list[Result], criterion: str) -> dict:
    """Of docs expert-labeled `failed` on `criterion`, what fraction does our
    OUTPUT bring to `passed`. expert_label is the INPUT selector ONLY.
    """
    # HARD ASSERT: never compute remediation from checker-framed (original) verdicts.
    assert all(r.framing == REMEDIATION for r in results), \
        "remediation_rate requires remediation-framed results (output verdicts)"
    failed_inputs = [r for r in results
                     if r.task.criterion == criterion and r.task.expert_label == "failed"]
    remediated = [r for r in failed_inputs if r.our_verdict.status == "passed"]
    return {
        "criterion": criterion,
        "failed_inputs": len(failed_inputs),
        "remediated": len(remediated),
        "rate": (len(remediated) / len(failed_inputs)) if failed_inputs else None,
    }


def checker_agreement(results: list[Result], criterion: str) -> dict:
    """Agreement between our verdict on the ORIGINAL and the expert label
    (passed/failed only; NP/CT and cannot_derive excluded from the rate).
    """
    # HARD ASSERT: only compare to the expert label when judging the ORIGINAL.
    assert all(r.framing == CHECKER for r in results), \
        "checker_agreement requires checker-framed results (original verdicts)"
    comparable = [r for r in results
                  if r.task.criterion == criterion
                  and r.task.expert_label in ("passed", "failed")
                  and r.our_verdict.status in ("passed", "failed")]
    agree = sum(1 for r in comparable if r.our_verdict.status == r.task.expert_label)
    return {
        "criterion": criterion,
        "comparable": len(comparable),
        "agree": agree,
        "agreement": (agree / len(comparable)) if comparable else None,
    }
