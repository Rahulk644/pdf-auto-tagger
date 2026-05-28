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


def tag_counts_open(pdf: pikepdf.Pdf) -> Counter | None:
    """tag_counts on an already-open Pdf (None if untagged)."""
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


def tag_counts(path: str | Path) -> Counter | None:
    """Count RoleMap-resolved struct-element /S tags in a PDF (None if untagged)."""
    with pikepdf.open(str(path)) as pdf:
        return tag_counts_open(pdf)


def objr_referenced_objgens(pdf: pikepdf.Pdf, tag: str = "/Link") -> set:
    """objgens of annotations referenced via OBJR from struct elems of `tag`.

    The struct-tree side of "is this annotation tagged?": an annotation is tagged
    iff some `tag` struct element (RoleMap-resolved) holds an OBJR pointing at it.
    """
    sr = pdf.Root.get("/StructTreeRoot")
    if sr is None:
        return set()
    resolve = role_resolver(sr)
    out: set = set()

    def walk(node):
        if not isinstance(node, Dictionary) or node.get("/S") is None:
            return
        if resolve(str(node.get("/S"))) == tag:
            k = node.get("/K")
            for o in (k if isinstance(k, Array) else [k] if k is not None else []):
                if isinstance(o, Dictionary) and o.get("/Type") == Name.OBJR \
                        and o.get("/Obj") is not None:
                    try:
                        out.add(o.get("/Obj").objgen)
                    except Exception:
                        pass
        k = node.get("/K")
        for c in (k if isinstance(k, Array) else [k] if k is not None else []):
            walk(c)

    top = sr.get("/K")
    for node in (top if isinstance(top, Array) else [top] if top is not None else []):
        walk(node)
    return out


def iter_struct_elems(pdf: pikepdf.Pdf):
    """Yield (resolved_tag, node) for every struct element (identified by /S)."""
    sr = pdf.Root.get("/StructTreeRoot")
    if sr is None:
        return
    resolve = role_resolver(sr)

    def walk(node):
        if not isinstance(node, Dictionary) or node.get("/S") is None:
            return
        yield (resolve(str(node.get("/S"))), node)
        k = node.get("/K")
        for c in (k if isinstance(k, Array) else [k] if k is not None else []):
            yield from walk(c)

    top = sr.get("/K")
    for node in (top if isinstance(top, Array) else [top] if top is not None else []):
        yield from walk(node)


def struct_order_mcids(pdf: pikepdf.Pdf) -> list:
    """[(page_index, mcid)] in struct-tree traversal (= assistive reading) order."""
    sr = pdf.Root.get("/StructTreeRoot")
    if sr is None:
        return []
    page_index = {p.obj.objgen: i for i, p in enumerate(pdf.pages)}
    seq: list = []

    def walk(node, pg_inherited):
        if not isinstance(node, Dictionary) or node.get("/S") is None:
            return
        pg = node.get("/Pg", pg_inherited)
        pidx = page_index.get(pg.objgen) if pg is not None else None

        def scan(v):
            if isinstance(v, int):
                if pidx is not None:
                    seq.append((pidx, int(v)))
            elif isinstance(v, Dictionary):
                if v.get("/S") is not None:
                    walk(v, pg)
                elif v.get("/Type") == Name.OBJR:
                    return
                elif v.get("/MCID") is not None and pidx is not None:
                    seq.append((pidx, int(v.get("/MCID"))))
            elif isinstance(v, Array):
                for it in v:
                    scan(it)

        scan(node.get("/K"))

    top = sr.get("/K")
    for node in (top if isinstance(top, Array) else [top] if top is not None else []):
        walk(node, None)
    return seq


def mcid_bbox_map(pdf_path: str) -> dict:
    """{(page_index, mcid): (top, x0)} reading-start point per MCID (pdfplumber)."""
    import pdfplumber
    from collections import defaultdict

    pos: dict = {}
    with pdfplumber.open(pdf_path) as pl:
        for pidx, page in enumerate(pl.pages):
            agg: dict = defaultdict(lambda: [1e9, 1e9])
            for ch in page.chars:
                m = ch.get("mcid")
                if m is None:
                    continue
                a = agg[(pidx, int(m))]
                a[0] = min(a[0], ch["top"])
                a[1] = min(a[1], ch["x0"])
            for k, v in agg.items():
                pos[k] = (v[0], v[1])
    return pos


def reading_monotonicity(pdf: pikepdf.Pdf, pdf_path: str, linetol: float = 10.0):
    """Fraction of consecutive same-page content items that follow reading geometry.

    Walks content in struct-tree (assistive reading) order; a consecutive pair is
    "in order" if the next item starts clearly below (top > cur+linetol) or on the
    same line to the right. Geometric PROXY for logical order — solid for
    single-column, weak for multi-column (see Contract 4 caveat). None if no pairs.
    """
    from collections import defaultdict

    seq = struct_order_mcids(pdf)
    pos = mcid_bbox_map(pdf_path)
    bypage: dict = defaultdict(list)
    for pidx, m in seq:
        if (pidx, m) in pos:
            bypage[pidx].append(pos[(pidx, m)])
    inorder = tot = 0
    for pts in bypage.values():
        for a, b in zip(pts, pts[1:]):
            tot += 1
            if (b[0] > a[0] + linetol) or (abs(b[0] - a[0]) <= linetol and b[1] >= a[1] - 1.0):
                inorder += 1
    return (inorder / tot) if tot else None


def strip_tag(tag: str | None) -> str:
    """'/H1' -> 'H1'; None/unmapped -> 'Artifact'."""
    if not tag:
        return ARTIFACT
    return tag[1:] if tag.startswith("/") else tag
