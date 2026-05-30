"""LaTeX → MathML emitter for PDF/UA-2 formula accessibility.

PDF/UA-2 (and the Matterhorn 1.1 maths checkpoints) want every `/Formula`
structure element to carry a machine-readable MathML representation, embedded as
a PDF 2.0 *Associated File* (`/AF`, relationship `/Supplement`) so assistive
technology can read the equation rather than guessing from glyphs.

This module is the pure-Python, no-ML half: it turns a LaTeX string into a
MathML document via `latex2mathml` (MIT). The LaTeX itself comes from Stage 5's
formula extractor — the cheap path is the born-digital text layer; the optional
image→LaTeX recogniser (pix2tex) is a separable accuracy upgrade layered on top.

Returns None when there's nothing usable to emit (empty / unparseable LaTeX), so
callers simply skip the `/AF` attachment and fall back to `/Alt` text only.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def latex_to_mathml(latex: str, is_inline: bool = False) -> str | None:
    """Convert a LaTeX string to a MathML document string.

    `is_inline` sets the MathML `display` attribute ("inline" vs "block"), which
    AT uses to decide phrasing. Returns None on empty input or any conversion
    error (the formula then ships with /Alt text only — still PDF/UA-1 valid)."""
    if not latex or not latex.strip():
        return None
    # \text{...} wrappers are the extractor's fallback for non-LaTeX-looking
    # content; they convert fine but add no math semantics. Still emit — a
    # MathML <mtext> is a valid, readable representation.
    try:
        from latex2mathml.converter import convert
    except Exception as e:  # pragma: no cover - latex2mathml is a hard dep
        logger.warning("latex2mathml unavailable, no MathML emitted: %s", e)
        return None
    try:
        mathml = convert(latex.strip())
    except Exception as e:
        logger.debug("LaTeX→MathML failed for %r: %s", latex[:60], e)
        return None
    if not mathml or "<math" not in mathml:
        return None
    if not is_inline:
        # latex2mathml defaults to display="inline"; promote display formulas.
        mathml = mathml.replace('display="inline"', 'display="block"', 1)
    return mathml
