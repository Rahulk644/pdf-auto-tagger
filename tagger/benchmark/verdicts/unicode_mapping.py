"""Adjacent axis — unicode_mapping (PDF/UA-1 7.21.x / Unicode recoverability).

NOT a benchmark-labeled criterion. Like font_embedding, a deterministic FILE-
CONFORMANCE axis: can the rendered text be recovered as Unicode (for screen
readers, search, copy/paste)?

Measured by EXTRACTION, not font-dict inspection. An earlier font-dict heuristic
(require /ToUnicode or a standard named encoding) badly OVER-FLAGGED: LaTeX/Computer-
Modern subset fonts carry custom /Differences encodings with no /ToUnicode yet their
glyph names resolve to Unicode (AGL), so the text IS recoverable — veraPDF agrees
(it reports only font-embedding, not a Unicode clause, on those docs). So instead we
ask the real question: extract the text and count glyphs that are NOT real Unicode.
Two failure modes (the ones Adobe / PAC flag):
  1. MISSING MAPPING — the text layer can't map the glyph at all; pdfminer emits
     "(cid:N)" and U+FFFD is the replacement char.
  2. PRIVATE USE AREA — the glyph maps to a PUA codepoint (U+E000-F8FF and the two
     supplementary PUA planes): looks like valid text to the eye, but is a private,
     non-standard assignment that assistive tech reads as nothing / garbage.
Because pdfminer applies /ActualText, a PUA glyph that carries an ActualText
replacement extracts as its real Unicode — so the PDF/UA ActualText exemption is
handled for free (only un-replaced PUA is counted).
"""
from __future__ import annotations

import re

from tagger.benchmark.verdicts.base import CannotDeriveReason, Verdict

_CID = re.compile(r"\(cid:\d+\)")
_WS = re.compile(r"\s")
# Below this fraction of unmappable glyphs, treat the doc as recoverable (a stray
# unmapped glyph — a ligature, an odd symbol — shouldn't fail the whole document).
_FAIL_RATIO = 0.02


def _is_pua(ch: str) -> bool:
    o = ord(ch)
    return (0xE000 <= o <= 0xF8FF        # BMP Private Use Area
            or 0xF0000 <= o <= 0xFFFFD   # Supplementary PUA-A
            or 0x100000 <= o <= 0x10FFFD)  # Supplementary PUA-B


def _unmapped_ratio(text: str) -> tuple[float, int, int]:
    """(ratio, unmapped, total) over non-whitespace glyphs. Unmapped = a '(cid:N)'
    run, a U+FFFD, or a Private-Use-Area codepoint; everything else is mapped."""
    cids = len(_CID.findall(text))
    rest = _WS.sub("", _CID.sub("", text))
    fffd = rest.count("�")
    pua = sum(1 for ch in rest if _is_pua(ch))
    unmapped = cids + fffd + pua
    mapped = len(rest) - fffd - pua
    total = mapped + unmapped
    return (unmapped / total if total else 0.0), unmapped, total


def verdict(pdf, pdf_path) -> Verdict:
    import pdfplumber

    parts = []
    with pdfplumber.open(pdf_path) as doc:
        for page in doc.pages:
            parts.append(page.extract_text() or "")
    text = "\n".join(parts)

    ratio, unmapped, total = _unmapped_ratio(text)
    if total == 0:
        return Verdict.cannot(CannotDeriveReason.NoElementsOfType, chars=0)
    detail = dict(unmapped_ratio=round(ratio, 4), unmapped_glyphs=unmapped, total_glyphs=total)
    return Verdict.failed(**detail) if ratio > _FAIL_RATIO else Verdict.passed(**detail)
