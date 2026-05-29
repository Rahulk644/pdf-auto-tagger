"""Adapter: our tagged-PDF struct tree -> dp-bench ground-truth-convention markdown.

Reads the OUTPUT PDF's struct tree (so it validates the full V2 path incl. Stage 10
reading-order writeback) and emits markdown matching ODL's GT conventions
(generate_groundtruth_markdown.py): elements in reading (struct-tree) order;
heading -> ``# text`` (level-agnostic per MHS); list item -> ``- text``; table ->
``<table>`` from TR/cell structure; everything else -> plain text; figures and
artifacts skipped. Text comes from each element's ``/ActualText`` (set by Stage 10,
properly spaced).
"""
from __future__ import annotations

from html import escape as html_escape
from typing import List

import pikepdf
from pikepdf import Array, Dictionary

from tagger.benchmark.struct_utils import role_resolver

_HEADINGS = {"/H1", "/H2", "/H3", "/H4", "/H5", "/H6"}
_SKIP = {"/Artifact", "/Figure", "/Form"}


def _actual_text(node: Dictionary) -> str:
    val = node.get("/ActualText")
    return str(val).strip() if val is not None else ""


def _struct_children(node: Dictionary):
    """Child nodes of ``node`` that are themselves struct elements (have /S)."""
    k = node.get("/K")
    for c in (k if isinstance(k, Array) else [k] if k is not None else []):
        if isinstance(c, Dictionary) and c.get("/S") is not None:
            yield c


def _table_html(table_node: Dictionary, resolve) -> str:
    rows: List[str] = []
    for tr in _struct_children(table_node):
        if resolve(str(tr.get("/S"))) != "/TR":
            continue
        cells: List[str] = []
        for cell in _struct_children(tr):
            ctag = resolve(str(cell.get("/S")))
            if ctag not in ("/TD", "/TH"):
                continue
            tag = "th" if ctag == "/TH" else "td"
            cells.append(f"<{tag}>{html_escape(_actual_text(cell))}</{tag}>")
        if cells:
            rows.append("<tr>" + "".join(cells) + "</tr>")
    return "<table>" + "".join(rows) + "</table>"


def _emit(node: Dictionary, resolve, out: List[str]) -> None:
    tag = resolve(str(node.get("/S")))
    if tag in _SKIP:
        return
    if tag in _HEADINGS:
        text = _actual_text(node)
        if text:
            out.append(f"# {text}")
        return
    if tag == "/L":
        for li in _struct_children(node):
            if resolve(str(li.get("/S"))) != "/LI":
                continue
            text = _actual_text(li) or " ".join(
                _actual_text(c) for c in _struct_children(li)
            ).strip()
            if text:
                out.append(f"- {text}")
        return
    if tag == "/Table":
        html = _table_html(node, resolve)
        if "<tr>" in html:
            out.append(html)
        return
    if tag == "/TOC":
        for toci in _struct_children(node):
            if resolve(str(toci.get("/S"))) == "/TOCI":
                text = _actual_text(toci)
                if text:
                    out.append(text)
        return
    # Container element (Sect/Part/Document nesting) with no own text -> recurse.
    children = list(_struct_children(node))
    own_text = _actual_text(node)
    if children and not own_text:
        for c in children:
            _emit(c, resolve, out)
        return
    if own_text:
        out.append(own_text)


def pdf_to_markdown(pdf_path: str) -> str:
    """Render a tagged PDF's struct tree to dp-bench-convention markdown."""
    with pikepdf.open(pdf_path) as pdf:
        sr = pdf.Root.get("/StructTreeRoot")
        if sr is None:
            return ""
        resolve = role_resolver(sr)
        out: List[str] = []
        top = sr.get("/K")
        roots = top if isinstance(top, Array) else [top] if top is not None else []
        for root in roots:
            if not isinstance(root, Dictionary) or root.get("/S") is None:
                continue
            if resolve(str(root.get("/S"))) == "/Document":
                for child in _struct_children(root):
                    _emit(child, resolve, out)
            else:
                _emit(root, resolve, out)
    return "\n\n".join(out)


__all__ = ["pdf_to_markdown"]
