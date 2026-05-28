"""Struct-tree reading utilities shared by the benchmark substrate + layout harness.

Single source of truth for parsing a tagged PDF's structure tree. Two hard-won
rules (see project memory project-benchmark-pdfa-design hand-validation):
  - struct elements are identified by /S PRESENCE, NOT /Type /StructElem (that
    key is OPTIONAL per ISO 32000 and absent in many real PDFs);
  - every /S is resolved through /RoleMap to its standard structure type (real
    PDFs use custom tag names like /HEAD_42, /Story mapped to standard types).
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pikepdf
from pikepdf import Array, Dictionary, Name

ARTIFACT = "Artifact"


def role_resolver(sr):
    """Resolve a raw /S tag through /RoleMap to its standard structure type.

    Follows chains (custom -> custom -> standard), cycle-safe. Returns a callable
    str -> str. No RoleMap -> identity.
    """
    rm = sr.get("/RoleMap") if sr is not None else None
    rm = rm if isinstance(rm, Dictionary) else None

    def resolve(s):
        if rm is None:
            return s
        seen = set()
        while s in rm and s not in seen:
            seen.add(s)
            s = str(rm[s])
        return s

    return resolve


def mcid_tag_map(pdf: pikepdf.Pdf) -> dict:
    """{(page_index, page-local MCID): standard /S tag} from the struct tree."""
    sr = pdf.Root.get("/StructTreeRoot")
    if sr is None:
        return {}
    resolve = role_resolver(sr)
    page_index = {p.obj.objgen: i for i, p in enumerate(pdf.pages)}
    out: dict = {}

    def walk(node, pg_inherited):
        if not isinstance(node, Dictionary) or node.get("/S") is None:
            return
        pg = node.get("/Pg", pg_inherited)
        tag = resolve(str(node.get("/S")))
        pidx = page_index.get(pg.objgen) if pg is not None else None

        def scan(v):
            if isinstance(v, int):
                if pidx is not None:
                    out[(pidx, int(v))] = tag
            elif isinstance(v, Dictionary):
                if v.get("/S") is not None:           # child struct elem (Type optional)
                    walk(v, pg)
                elif v.get("/Type") == Name.OBJR:
                    return
                elif v.get("/MCID") is not None and pidx is not None:  # MCR
                    out[(pidx, int(v.get("/MCID")))] = tag
            elif isinstance(v, Array):
                for it in v:
                    scan(it)

        scan(node.get("/K"))

    top = sr.get("/K")
    for node in (top if isinstance(top, Array) else [top] if top is not None else []):
        walk(node, None)
    return out


def tag_counts(path: str | Path) -> Counter | None:
    """Count RoleMap-resolved struct-element /S tags in a PDF (None if untagged)."""
    with pikepdf.open(str(path)) as pdf:
        sr = pdf.Root.get("/StructTreeRoot")
        if sr is None:
            return None
        resolve = role_resolver(sr)
        counts: Counter = Counter()

        def walk(node):
            if not isinstance(node, Dictionary):
                return
            if node.get("/S") is not None:
                counts[resolve(str(node.get("/S")))] += 1
            k = node.get("/K")
            for c in (k if isinstance(k, Array) else [k] if k is not None else []):
                walk(c)

        walk(sr.get("/K"))
        return counts


def strip_tag(tag: str | None) -> str:
    """'/H1' -> 'H1'; None/unmapped -> 'Artifact'."""
    if not tag:
        return ARTIFACT
    return tag[1:] if tag.startswith("/") else tag
