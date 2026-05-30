"""Deterministic screen-reader linearizer — a cross-platform AT simulation.

Real screen readers are OS-bound: NVDA/JAWS are Windows-only, VoiceOver is macOS
with no clean automation API. So an *in-pipeline* accessibility check can't drive
a real reader. Instead we simulate the one thing they all do: walk the tagged
PDF's structure tree in LOGICAL reading order and emit, per element, what the
reader would announce — heading levels, figure alternate text, table dimensions,
list items — while SILENCING artifacts (page furniture a reader must skip).

Our Stage 10 builds the struct tree in reading order and writes /ActualText on
text elements + /Alt on figures, so the tree walk reproduces the AT transcript
without needing the content stream. This gives two things on a Mac:
  - `transcript(pdf)` — the readable "what a screen reader says" stream, to eyeball
    reading order and announcements.
  - `smell_test(pdf)` — deterministic issues a reader would hit: a graphic with no
    description, an empty heading announced, a table with no header row, etc.

The real NVDA/JAWS (Windows) and VoiceOver (macOS) runs remain valuable as an
out-of-process verification job; this is the part that runs anywhere, every build.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pikepdf
from pikepdf import Array, Dictionary

_HEADINGS = {f"/H{i}": i for i in range(1, 7)}


@dataclass
class Announcement:
    role: str            # "heading" / "figure" / "table" / "list" / "link" / "text" / ...
    text: str
    level: int = 0       # heading level, else 0
    issue: str = ""      # non-empty if a reader would hit a problem here


@dataclass
class Transcript:
    path: str
    announcements: list[Announcement] = field(default_factory=list)

    @property
    def issues(self) -> list[Announcement]:
        return [a for a in self.announcements if a.issue]

    def as_text(self) -> str:
        lines = []
        for a in self.announcements:
            if a.role == "heading":
                lines.append(f"Heading level {a.level}: {a.text}")
            elif a.role == "figure":
                lines.append(f"Graphic: {a.text}" if a.text else "Graphic, no description")
            elif a.role == "formula":
                lines.append(f"Formula: {a.text}" if a.text else "Formula, no description")
            elif a.role == "table":
                lines.append(a.text)  # pre-rendered "Table, R rows by C columns"
            elif a.role == "list":
                lines.append(a.text)
            elif a.role == "link":
                lines.append(f"Link: {a.text}")
            else:
                lines.append(a.text)
        return "\n".join(lines)


def _txt(node) -> str:
    for key in ("/ActualText", "/Alt"):
        v = node.get(key)
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""


def linearize(path: str) -> Transcript:
    """Walk the struct tree in order and produce the AT announcement stream."""
    t = Transcript(path=path)
    try:
        pdf = pikepdf.open(path)
    except Exception as e:
        t.announcements.append(Announcement("error", str(e), issue="cannot open PDF"))
        return t
    try:
        sr = pdf.Root.get("/StructTreeRoot")
        if sr is None:
            t.announcements.append(Announcement(
                "error", "", issue="no StructTreeRoot — document is untagged, a reader gets nothing"))
            return t

        def kids(node):
            k = node.get("/K") if isinstance(node, Dictionary) else None
            return [c for c in (k if isinstance(k, Array) else [k] if k is not None else [])
                    if isinstance(c, Dictionary)]

        def count(node, s):
            return sum(1 for c in kids(node) if str(c.get("/S")) == s)

        def walk(node):
            if not isinstance(node, Dictionary):
                return
            s = str(node.get("/S")) if node.get("/S") is not None else ""

            if s == "/Artifact":
                return  # a screen reader skips artifacts entirely — silence them
            if s in _HEADINGS:
                txt = _txt(node)
                t.announcements.append(Announcement(
                    "heading", txt, level=_HEADINGS[s],
                    issue="" if txt else "empty heading announced"))
                return
            if s == "/Figure":
                txt = _txt(node)
                t.announcements.append(Announcement(
                    "figure", txt, issue="" if txt else "graphic with no alternate text"))
                return
            if s == "/Formula":
                txt = _txt(node)
                t.announcements.append(Announcement(
                    "formula", txt, issue="" if txt else "formula with no text equivalent"))
                return
            if s == "/Table":
                rows = count(node, "/TR")
                # rows can also live under THead/TBody/TFoot
                for c in kids(node):
                    if str(c.get("/S")) in ("/THead", "/TBody", "/TFoot"):
                        rows += count(c, "/TR")
                has_th = _has_descendant(node, "/TH")
                t.announcements.append(Announcement(
                    "table", f"Table, {rows} rows",
                    issue="" if has_th else "table has no header cells (TH) — columns unlabelled"))
                for c in kids(node):
                    walk(c)
                return
            if s == "/L":
                n = count(node, "/LI")
                t.announcements.append(Announcement("list", f"List, {n} items"))
                for c in kids(node):
                    walk(c)
                return
            if s == "/Link":
                # Link visible text usually lives in MCID-backed content, which
                # this /ActualText-based walk can't see — so DON'T treat empty
                # text as a defect (that's a linearizer blind spot, not a real
                # one). Only flag text we CAN see that's a known-bad phrase.
                txt = _txt(node)
                bad = txt.strip().lower() in ("click here", "here", "link", "read more", "more")
                t.announcements.append(Announcement(
                    "link", txt, issue="non-descriptive link text" if bad else ""))
                return
            if s in ("/P", "/Caption", "/Note", "/LI", "/LBody", "/Lbl", "/TD", "/TH",
                     "/BlockQuote", "/Quote", "/Reference", "/Code", "/TOCI"):
                txt = _txt(node)
                if txt:
                    t.announcements.append(Announcement("text", txt))
                for c in kids(node):
                    walk(c)
                return
            # generic container (Document, Sect, Div, THead/TBody, ...) — descend
            for c in kids(node):
                walk(c)

        walk(sr)
        if not t.announcements:
            t.announcements.append(Announcement(
                "error", "", issue="struct tree present but nothing announceable"))
    finally:
        pdf.close()
    return t


def _has_descendant(node, s_value) -> bool:
    if not isinstance(node, Dictionary):
        return False
    if str(node.get("/S")) == s_value:
        return True
    k = node.get("/K")
    for c in (k if isinstance(k, Array) else [k] if k is not None else []):
        if isinstance(c, Dictionary) and _has_descendant(c, s_value):
            return True
    return False


def smell_test(path: str) -> list[Announcement]:
    """The issues a screen-reader user would actually hit (subset of announcements
    flagged with `issue`)."""
    return linearize(path).issues


def main() -> None:
    import sys
    if len(sys.argv) < 2:
        print("usage: python -m tagger.audit.screen_reader <pdf> [--issues]")
        sys.exit(2)
    issues_only = "--issues" in sys.argv
    for path in [a for a in sys.argv[1:] if not a.startswith("-")]:
        t = linearize(path)
        print(f"\n== {path} ==")
        if issues_only:
            for a in t.issues:
                print(f"  [ISSUE] {a.role}: {a.issue}")
            if not t.issues:
                print("  no screen-reader issues found")
        else:
            print(t.as_text())
            print(f"\n  ({len(t.announcements)} announcements, {len(t.issues)} issues)")


if __name__ == "__main__":
    main()
