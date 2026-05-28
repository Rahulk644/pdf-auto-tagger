"""Contract 2 — table_structure (WCAG 1.3.1).

This is a table-criterion doc, so a table is expected: no /Table struct -> failed
(we did not tag the table). passed = a Table with TR>TD and >=1 TH (header cells).
Tags are RoleMap-resolved. NOTE: our WEAK axis (grid topology) — expect a low
number; that's the honest signal driving the deferred grid fix, do not soften it.
Cell-level accuracy is DocLayNet's job, not this contract.
"""
from tagger.benchmark.struct_utils import tag_counts_open
from tagger.benchmark.verdicts.base import Verdict


def verdict(pdf, pdf_path) -> Verdict:
    counts = tag_counts_open(pdf)
    if not counts:
        return Verdict.failed(note="no struct tree / untagged")

    n_table = counts.get("/Table", 0)
    n_th = counts.get("/TH", 0)
    n_td = counts.get("/TD", 0)
    detail = dict(tables=n_table, th=n_th, td=n_td)

    if n_table == 0:
        return Verdict.failed(**detail, note="no Table struct (table not tagged)")
    if n_th == 0:
        return Verdict.failed(**detail, note="table has no header cells (TH)")
    return Verdict.passed(**detail)
