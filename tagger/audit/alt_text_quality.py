"""Alt-text QUALITY checker — operationalizes the McGowan "Alt Text Writing
Guidelines" into the mechanically-checkable rules (the deterministic floor of
quality, no model needed). What it CANNOT judge — "does the alt convey the
image's purpose?" — is the semantic half that needs the LLM judge; this catches
the violations a rubric can decide without seeing the image.

Guideline → rule:
  - "A screen reader announces 'graphic' then reads the alt" → alt must NOT start
    with "image of / picture of / graphic of / photo of ..." (redundant).
  - "Be concise" → short alt within the length cap; flag the empty and the bloated.
  - "Do not be redundant" → alt must not merely duplicate an adjacent /Caption.
  - "Explain the information the image conveys, not its appearance" → a bare type
    word ("Chart.", "Figure.") conveys no information → low quality.
  - Decorative images take NO alt (handled upstream as /Artifact).

Read-only; returns per-figure issues + an aggregate. Pairs with the benchmark's
alt_text_quality expert axis (the 0% hole) as the deterministic component.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pikepdf
from pikepdf import Array, Dictionary

_MAX_SHORT_ALT = 150  # guideline short-alt cap (long description carries the rest)
_APPEARANCE_PREFIX = re.compile(
    r"^\s*(an?\s+)?(image|picture|photo|photograph|graphic|illustration|icon)\s+(of|showing|depicting|that shows)\b",
    re.IGNORECASE)
_BARE_TYPE = re.compile(
    r"^\s*(figure|image|chart|graph|diagram|schematic|map|photo|photograph|table|graphic)\s*[.:]?\s*$",
    re.IGNORECASE)


@dataclass
class AltIssue:
    page: object
    rule: str
    detail: str
    alt: str = ""


@dataclass
class AltQualityReport:
    path: str
    figures: int = 0
    with_alt: int = 0
    issues: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"path": self.path, "figures": self.figures, "with_alt": self.with_alt,
                "issues": [i.__dict__ for i in self.issues]}


def _txt(node, key):
    v = node.get(key) if isinstance(node, Dictionary) else None
    return str(v).strip() if v is not None else ""


def check_alt_quality(path: str) -> AltQualityReport:
    rep = AltQualityReport(path=path)
    try:
        pdf = pikepdf.open(path)
    except Exception as e:
        rep.issues.append(AltIssue(None, "io", str(e)))
        return rep
    try:
        sr = pdf.Root.get("/StructTreeRoot")
        if sr is None:
            return rep

        def kids(n):
            k = n.get("/K") if isinstance(n, Dictionary) else None
            return [c for c in (k if isinstance(k, Array) else [k] if k is not None else [])
                    if isinstance(c, Dictionary)]

        def pg(n):
            p = n.get("/Pg")
            try:
                return p.objgen[0] if p is not None else None
            except Exception:
                return None

        # collect captions per page for redundancy check
        captions_by_pg: dict = {}

        def collect(n):
            if str(n.get("/S")) == "/Caption":
                captions_by_pg.setdefault(pg(n), []).append(_txt(n, "/ActualText"))
            for c in kids(n):
                collect(c)
        collect(sr)

        def walk(n):
            if str(n.get("/S")) == "/Figure":
                rep.figures += 1
                alt = _txt(n, "/Alt") or _txt(n, "/ActualText")
                if not alt:
                    rep.issues.append(AltIssue(pg(n), "empty", "Figure has no /Alt"))
                else:
                    rep.with_alt += 1
                    if _APPEARANCE_PREFIX.match(alt):
                        rep.issues.append(AltIssue(pg(n), "appearance_prefix",
                            "starts with 'image/picture of' — reader already says 'graphic'", alt))
                    if len(alt) > _MAX_SHORT_ALT:
                        rep.issues.append(AltIssue(pg(n), "too_long",
                            f"{len(alt)} chars > {_MAX_SHORT_ALT}; move detail to a long description", alt))
                    if _BARE_TYPE.match(alt):
                        rep.issues.append(AltIssue(pg(n), "bare_type",
                            "only a type word — conveys no information", alt))
                    for cap in captions_by_pg.get(pg(n), []):
                        if cap and _norm(cap) == _norm(alt):
                            rep.issues.append(AltIssue(pg(n), "redundant_with_caption",
                                "alt duplicates the adjacent caption", alt))
                            break
            for c in kids(n):
                walk(c)
        walk(sr)
    finally:
        pdf.close()
    return rep


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def main() -> None:
    import sys
    if len(sys.argv) < 2:
        print("usage: python -m tagger.audit.alt_text_quality <pdf> [...]")
        sys.exit(2)
    for path in sys.argv[1:]:
        rep = check_alt_quality(path)
        print(f"\n== {path} ==  figures={rep.figures} with_alt={rep.with_alt} issues={len(rep.issues)}")
        for i in rep.issues:
            print(f"  [{i.rule}] p{i.page}: {i.detail}" + (f"  | {i.alt[:60]!r}" if i.alt else ""))


if __name__ == "__main__":
    main()
