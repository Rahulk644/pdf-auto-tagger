"""Per-document IO caches: rasterized page images + pdfplumber handles.

Several stages independently rasterize and re-open the SAME PDF: Stage 3 (Heron
region detection), Stage 5 (TableFormer, formula recogniser) and Stage 1 (native
extraction) each `fitz.open`/`pdfplumber.open` the file and re-render/re-parse
pages. These two small caches let a page be rendered once (shared by Heron +
TableFormer + the formula renderer) and the PDF parsed once (shared by the
native extractor + layout detector + table extractor).

The image cache is BOUNDED (maxsize 8) on purpose — the gold target is an M1
8 GB box, and a 150-DPI letter page is ~6 MB, so an unbounded cache would blow
memory on long documents. Bounded means full cross-stage reuse on the common
small docs while a long doc degrades gracefully to re-rendering, never to an OOM.

The pipeline calls clear_document_caches() in its run() finally so neither
handles nor images outlive the document.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from functools import lru_cache

from tagger.config import STANDARD_DPI

logger = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def _render(path: str, page_num: int, dpi: int):
    import fitz
    from PIL import Image

    with fitz.open(path) as doc:
        if not (0 <= page_num - 1 < len(doc)):
            return None
        pix = doc[page_num - 1].get_pixmap(dpi=dpi)
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def render_page(pdf_path, page_num: int, dpi: int = STANDARD_DPI):
    """Return a (default 150-DPI) RGB PIL image of the page, fitz-rendered once
    and cached. Callers MUST treat the result as read-only — copy before any
    mutation (np.array() / .crop() already return copies). None on failure or
    out-of-range page."""
    try:
        return _render(str(pdf_path), int(page_num), int(dpi))
    except Exception as e:
        logger.debug("render_page failed (%s p%d): %s", pdf_path, page_num, e)
        return None


_pdf_cache: dict = {}  # path -> pdfplumber.PDF (opened once per document)


def get_pdf(pdf_path):
    """Return a cached pdfplumber.PDF for this path, opened once per document.
    Do NOT close it or use it in a `with` block — its lifecycle is owned by
    clear_document_caches()."""
    import pdfplumber

    key = str(pdf_path)
    pdf = _pdf_cache.get(key)
    if pdf is None:
        pdf = pdfplumber.open(key)
        _pdf_cache[key] = pdf
    return pdf


@contextmanager
def open_pdf(pdf_path):
    """Drop-in for `with pdfplumber.open(path) as pdf:` that yields the cached
    handle and does NOT close it on exit (lifecycle owned by
    clear_document_caches). Lets call sites change a single line."""
    yield get_pdf(pdf_path)


def clear_document_caches() -> None:
    """Release cached pdfplumber handles + rendered images. Idempotent; the
    pipeline calls this in run()'s finally so caches don't outlive a document."""
    _render.cache_clear()
    for pdf in _pdf_cache.values():
        try:
            pdf.close()
        except Exception:
            pass
    _pdf_cache.clear()
