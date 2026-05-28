"""Layout-accuracy harness — token-extraction normalizer (shared substrate).

Normalizes a tagged PDF into a flat list of word tokens, each carrying the
struct-tree tag of the marked content it belongs to:
    {page, text, bbox(72-DPI top-left), tag}

How: walk the struct tree to build (page_index, page-local MCID) -> /S tag, then
read pdfplumber words with their mcid and look the tag up. Words whose mcid is
None or unmapped are tagged "Artifact" (marked artifact or untagged content).

Because PREP's tagged PDF and ours tag the SAME source document, tokens align
near-exactly by position across the two — the substrate for phase-2 token-level
tag agreement (see scratch/layout_agreement.py) and figure-IoU coverage.

Usage:
  .venv3/bin/python scratch/layout_tokens.py <tagged.pdf> [page_index]   # dump
"""
import sys
from collections import Counter

import pdfplumber
import pikepdf
from pikepdf import Array, Dictionary, Name

ARTIFACT = "Artifact"


def role_resolver(sr):
    """Resolve a raw /S tag through /RoleMap to its standard structure type.

    Real PDFs use custom tag names (e.g. /HEAD_42, /Story, /StyleSpan) mapped to
    standard types via the struct tree's /RoleMap; without resolution they look
    unknown. Follows chains (custom -> custom -> standard), cycle-safe.
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
    """{(page_index, page-local MCID): standard /S tag} from the struct tree.

    Identifies struct elements by /S PRESENCE — the /Type /StructElem key is
    OPTIONAL per ISO 32000 and absent in many real PDFs — and resolves each /S
    through /RoleMap to its standard type.
    """
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


def strip_tag(tag: str | None) -> str:
    """'/H1' -> 'H1'; None/unmapped -> 'Artifact'."""
    if not tag:
        return ARTIFACT
    return tag[1:] if tag.startswith("/") else tag


def extract_tokens(pdf_path: str) -> list[dict]:
    """Flat word tokens with their struct tag, in reading order per page."""
    with pikepdf.open(pdf_path) as pk:
        tmap = mcid_tag_map(pk)
    tokens: list[dict] = []
    with pdfplumber.open(pdf_path) as plumb:
        for pidx, page in enumerate(plumb.pages):
            for w in page.extract_words(extra_attrs=["mcid"]):
                m = w.get("mcid")
                tag = tmap.get((pidx, int(m))) if m is not None else None
                tokens.append({
                    "page": pidx,
                    "text": w["text"],
                    "bbox": (w["x0"], w["top"], w["x1"], w["bottom"]),
                    "tag": strip_tag(tag),
                })
    return tokens


def _dump(pdf_path, page_idx=None):
    toks = extract_tokens(pdf_path)
    if page_idx is not None:
        toks = [t for t in toks if t["page"] == page_idx]
    dist = Counter(t["tag"] for t in toks)
    print(f"{pdf_path}  tokens={len(toks)}  tag dist={dict(dist)}")
    for t in toks[:60]:
        print(f"  p{t['page']} {t['tag']:9s} {t['text'][:50]}")


if __name__ == "__main__":
    _dump(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None)
