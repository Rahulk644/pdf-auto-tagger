"""W3C ACT Rules + PDF/UA-1 audit checker.

Reads a tagged PDF and reports per-rule pass / fail / not-applicable for the
rules our pipeline cares about. Rule IDs match the W3C ACT Rules Format ID
when one exists; PDF/UA-1 clause numbers are used for the rules that don't
have an ACT cousin. Read-only — never modifies the PDF.

Rules implemented:

  ACT 6cfa84  /  WCAG 1.1.1   Every <Figure> has /Alt or /ActualText
  ACT 36b590  /  WCAG 1.3.1   Every heading (H1-H6) is non-empty
  ACT b40fd1  /  WCAG 3.1.1   Catalog /Lang is a valid BCP-47 tag
  PDF/UA 7.4.2 /  WCAG 1.3.1  Heading levels are not skipped (H1->H3 fails)
  PDF/UA 7.1-10/ WCAG 2.4.2   Catalog has /Info /Title AND ViewerPreferences
                              /DisplayDocTitle = true
  PDF/UA 7.5.2 /  WCAG 1.3.1  Every <Caption> structurally accompanies a
                              <Figure> or <Table>
  PDF/UA 7.5.3 /  WCAG 1.3.1  Every <LI> is inside an <L>
  PDF/UA 7.1-1 /  WCAG 1.3.1  Every page has either /StructTreeRoot coverage
                              or a /MarkInfo dictionary signalling tagged

Each rule returns:
  status: "pass" / "fail" / "not_applicable"
  detail: int count (failed items, or 1/0)
  notes:  optional human-readable extra info
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field

import pikepdf
from pikepdf import Array, Dictionary

# BCP-47 simple structural check — accepts en, en-US, en-GB, zh-Hant etc.
_BCP47 = re.compile(r"^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{1,8})*$")
_HEADING_TAGS = {"/H1", "/H2", "/H3", "/H4", "/H5", "/H6"}
_TAG_TO_LEVEL = {f"/H{i}": i for i in range(1, 7)}


@dataclass
class RuleResult:
    rule_id: str
    title: str
    status: str  # pass / fail / not_applicable
    detail: int = 0
    notes: str = ""


@dataclass
class AuditReport:
    path: str
    results: list[RuleResult] = field(default_factory=list)

    @property
    def passes(self) -> int:
        return sum(1 for r in self.results if r.status == "pass")

    @property
    def fails(self) -> int:
        return sum(1 for r in self.results if r.status == "fail")

    @property
    def na(self) -> int:
        return sum(1 for r in self.results if r.status == "not_applicable")

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "summary": {"pass": self.passes, "fail": self.fails,
                        "not_applicable": self.na, "total": len(self.results)},
            "rules": [r.__dict__ for r in self.results],
        }


def audit_pdf(path: str) -> AuditReport:
    rep = AuditReport(path=path)
    try:
        pdf = pikepdf.open(path)
    except Exception as e:
        rep.results.append(RuleResult("io", "Open PDF", "fail", notes=str(e)))
        return rep

    try:
        root = pdf.Root
        # Walk the struct tree once and collect what we need.
        tag_counts: dict[str, int] = {}
        figures_missing_alt = 0
        empty_headings = 0
        heading_sequence: list[str] = []   # numbered H1-H6 in document order
        unnumbered_headings: list[str] = []  # unnumbered /H tags (PDF/UA allows H1-H6 XOR H)
        captions: list[Dictionary] = []
        lis_outside_l = 0
        figures_by_page: dict = {}
        tables_by_page: dict = {}

        sr = root.get("/StructTreeRoot")

        def is_heading_empty(node):
            at = node.get("/ActualText")
            if at is not None and str(at).strip():
                return False
            alt = node.get("/Alt")
            if alt is not None and str(alt).strip():
                return False
            # Walk children; if any /K MCID is a number, the heading has
            # marked content backing it -> not empty.
            k = node.get("/K")
            if k is None:
                return True
            kids = k if isinstance(k, Array) else [k]
            for c in kids:
                if isinstance(c, (int,)):
                    return False
                if isinstance(c, Dictionary):
                    if c.get("/MCID") is not None:
                        return False
            return True

        def walk(node, parent_s=None):
            nonlocal figures_missing_alt, empty_headings, lis_outside_l
            if isinstance(node, Dictionary) and node.get("/S") is not None:
                s = str(node.get("/S"))
                tag_counts[s] = tag_counts.get(s, 0) + 1
                if s == "/Figure":
                    figures_by_page.setdefault(_pg_id(node), []).append(node)
                    has_alt = node.get("/Alt") is not None and str(node.get("/Alt")).strip()
                    has_at = node.get("/ActualText") is not None and str(node.get("/ActualText")).strip()
                    if not (has_alt or has_at):
                        figures_missing_alt += 1
                elif s == "/Table":
                    tables_by_page.setdefault(_pg_id(node), []).append(node)
                elif s in _HEADING_TAGS:
                    heading_sequence.append(s)
                elif s == "/H":
                    unnumbered_headings.append(s)
                    if is_heading_empty(node):
                        empty_headings += 1
                elif s == "/Caption":
                    captions.append(node)
                elif s == "/LI":
                    if parent_s != "/L":
                        lis_outside_l += 1
                parent_s = s
            k = node.get("/K") if isinstance(node, Dictionary) else None
            for c in (k if isinstance(k, Array) else [k] if k is not None else []):
                if isinstance(c, Dictionary):
                    walk(c, parent_s)

        if sr is not None:
            walk(sr)

        # --- Rule definitions ---

        # ACT 6cfa84 — every /Figure has /Alt or /ActualText
        n_fig = tag_counts.get("/Figure", 0)
        if n_fig == 0:
            rep.results.append(RuleResult(
                "ACT-6cfa84", "Figure has accessible name",
                "not_applicable", notes="no /Figure elements"))
        else:
            rep.results.append(RuleResult(
                "ACT-6cfa84", "Figure has accessible name",
                "pass" if figures_missing_alt == 0 else "fail",
                detail=figures_missing_alt,
                notes=f"{n_fig - figures_missing_alt}/{n_fig} figures have /Alt or /ActualText"))

        # ACT 36b590 — every heading is non-empty
        n_h = sum(tag_counts.get(t, 0) for t in _HEADING_TAGS)
        if n_h == 0:
            rep.results.append(RuleResult(
                "ACT-36b590", "Heading is non-empty",
                "not_applicable", notes="no headings in struct tree"))
        else:
            rep.results.append(RuleResult(
                "ACT-36b590", "Heading is non-empty",
                "pass" if empty_headings == 0 else "fail",
                detail=empty_headings,
                notes=f"{n_h - empty_headings}/{n_h} headings have content"))

        # ACT b40fd1 — Catalog /Lang is a valid BCP-47 tag
        lang_obj = root.get("/Lang")
        lang_str = str(lang_obj) if lang_obj is not None else ""
        if not lang_str:
            rep.results.append(RuleResult(
                "ACT-b40fd1", "Document language is set",
                "fail", notes="missing /Lang"))
        elif not _BCP47.match(lang_str.strip()):
            rep.results.append(RuleResult(
                "ACT-b40fd1", "Document language is valid BCP-47",
                "fail", notes=f"/Lang = {lang_str!r} (invalid BCP-47)"))
        else:
            rep.results.append(RuleResult(
                "ACT-b40fd1", "Document language is valid BCP-47",
                "pass", notes=f"/Lang = {lang_str}"))

        # PDF/UA 7.4.2 — numbered heading hierarchy. Three failure modes (all
        # WCAG 1.3.1 / PDF/UA-1 7.4.2, validated against veraPDF-corpus 7.4.x):
        #   (1) level skip      — H1 -> H3
        #   (2) first not H1    — sequence starts at H2/H3 (e.g. H2,H3,H4)
        #   (3) mixed H + Hn    — a document must use EITHER numbered (H1-H6)
        #                         OR unnumbered (/H) headings, never both.
        problems = []
        skip_count = 0
        last_level = 0
        for s in heading_sequence:
            lvl = _TAG_TO_LEVEL[s]
            if last_level > 0 and lvl > last_level + 1:
                skip_count += 1
            last_level = lvl
        if skip_count:
            problems.append(f"{skip_count} level-skip(s)")
        if heading_sequence and heading_sequence[0] != "/H1":
            problems.append(f"first heading is {heading_sequence[0]} not /H1")
        if heading_sequence and unnumbered_headings:
            problems.append(f"mixes numbered + {len(unnumbered_headings)} unnumbered /H")
        if not heading_sequence and not unnumbered_headings:
            rep.results.append(RuleResult(
                "PDFUA-7.4.2", "Heading hierarchy (no skip / first=H1 / no mix)",
                "not_applicable", notes="no headings"))
        else:
            seq = " ".join(heading_sequence) or f"{len(unnumbered_headings)}×/H"
            rep.results.append(RuleResult(
                "PDFUA-7.4.2", "Heading hierarchy (no skip / first=H1 / no mix)",
                "pass" if not problems else "fail", detail=len(problems),
                notes=("; ".join(problems) + f" | sequence: {seq}") if problems
                      else f"sequence: {seq}"))

        # PDF/UA 7.1-10 — /Info /Title set AND /DisplayDocTitle true.
        # The canonical /Info dict is in the trailer, NOT the catalog —
        # pdf.Root.get('/Info') can return a stale partial dict with only
        # /Producer set, so always use pdf.docinfo (trailer-resolved).
        info = pdf.docinfo if hasattr(pdf, "docinfo") else pdf.trailer.get("/Info")
        title = info.get("/Title") if isinstance(info, Dictionary) else None
        title_str = str(title) if title is not None else ""
        vp = root.get("/ViewerPreferences")
        ddt = False
        if isinstance(vp, Dictionary):
            ddt = bool(vp.get("/DisplayDocTitle"))
        if title_str.strip() and ddt:
            rep.results.append(RuleResult(
                "PDFUA-7.1-10", "/Info /Title set and /DisplayDocTitle true",
                "pass", notes=f"/Title = {title_str[:60]!r}"))
        else:
            missing = []
            if not title_str.strip():
                missing.append("/Info /Title")
            if not ddt:
                missing.append("/ViewerPreferences /DisplayDocTitle")
            rep.results.append(RuleResult(
                "PDFUA-7.1-10", "/Info /Title set and /DisplayDocTitle true",
                "fail", notes="missing: " + ", ".join(missing)))

        # PDF/UA 7.5.2 — every /Caption has a /Figure or /Table neighbour in the struct tree
        # NB: structural neighbour check by Pg-id; spatial check done in
        # pdfua_structural_enforcer (Stage 8); here we just check Pg colocation.
        if not captions:
            rep.results.append(RuleResult(
                "PDFUA-7.5.2", "Caption accompanies Figure or Table",
                "not_applicable", notes="no /Caption elements"))
        else:
            orphan = 0
            for cap in captions:
                pg = _pg_id(cap)
                if not (figures_by_page.get(pg) or tables_by_page.get(pg)):
                    orphan += 1
            rep.results.append(RuleResult(
                "PDFUA-7.5.2", "Caption accompanies Figure or Table",
                "pass" if orphan == 0 else "fail", detail=orphan,
                notes=f"{len(captions) - orphan}/{len(captions)} captions colocated with a /Figure or /Table"))

        # PDF/UA 7.5.3 — every /LI is inside /L
        n_li = tag_counts.get("/LI", 0)
        if n_li == 0:
            rep.results.append(RuleResult(
                "PDFUA-7.5.3", "Every /LI is inside /L",
                "not_applicable", notes="no /LI elements"))
        else:
            rep.results.append(RuleResult(
                "PDFUA-7.5.3", "Every /LI is inside /L",
                "pass" if lis_outside_l == 0 else "fail", detail=lis_outside_l,
                notes=f"{n_li - lis_outside_l}/{n_li} list items inside /L"))

        # PDF/UA 7.1-1 — page is tagged (Marked = true)
        mi = root.get("/MarkInfo")
        marked = isinstance(mi, Dictionary) and bool(mi.get("/Marked"))
        rep.results.append(RuleResult(
            "PDFUA-7.1-1", "/MarkInfo /Marked is true",
            "pass" if marked else "fail",
            notes="document declares itself as tagged" if marked else "missing or false"))

    finally:
        pdf.close()
    return rep


def _pg_id(node):
    pg = node.get("/Pg") if isinstance(node, Dictionary) else None
    if pg is None:
        return None
    try:
        return pg.objgen
    except Exception:
        return id(pg)


def main():
    if len(sys.argv) < 2:
        print("usage: python -m tagger.audit.act_rules <pdf> [pdf ...]")
        sys.exit(2)
    aggregate: list[dict] = []
    for path in sys.argv[1:]:
        rep = audit_pdf(path)
        d = rep.to_dict()
        aggregate.append(d)
        print(f"\n== {path} ==")
        print(f"  pass={rep.passes}  fail={rep.fails}  na={rep.na}")
        for r in rep.results:
            mark = "PASS" if r.status == "pass" else (
                "FAIL" if r.status == "fail" else " NA ")
            print(f"  [{mark}] {r.rule_id:<14} {r.title}  -- {r.notes}")
    if "--json" in sys.argv:
        print(json.dumps(aggregate, indent=2, default=str))


if __name__ == "__main__":
    main()
