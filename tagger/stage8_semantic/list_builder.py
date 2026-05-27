"""
List structure builder.

PDF/UA requires lists to be structured as:
  L > LI > Lbl (bullet/number) + LBody (content)

This module takes flat LI-tagged elements and groups them into
proper list structures, splitting each LI into its label and body.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict

from tagger.models.data_types import PDFTag, TaggedElement

# Left-edge alignment tolerance as a fraction of marker line height
# (bbox-relative, so it is unit/DPI-invariant; equivalent to ODL's fontSize*0.3).
_ALIGN_TOL_FRAC = 0.3

logger = logging.getLogger(__name__)

# Bullet label characters — ported from OpenDataLoader's POSSIBLE_LABELS set
# (geometric shapes, dingbats, arrows, circled/parenthesized numbers used as
# single-character list markers). Private-use-area font glyphs are excluded.
_BULLET_CHARS = set(
    "∘*+-.=‐‑‒–—―•‣․‧※⁃⁎→↳⇒⇨⇾∙"
    "■□▢▣▤▥▦▧▨▩▪▬▭▮▯▰▱▲△▴▵▶▷▸▹►▻▼▽▾▿◀◁◂◃◄◅"
    "◆◇◈◉◊○◌◍◎●◐◑◒◓◔◕◖◗◘◙◢◣◤◥◦◧◨◩◪◫◬◭◮◯◰◱◲◳◴◵◶◷◸◹◺◻◼◽◾◿"
    "★☆☐☑☒☓☛☞♠♡♢♣♤♥♦♧⚪⚫⚬✓✔✕✖✗✘✙✚✛✜✝✞✟✦✧✨❍❏❐❑❒❖"
    "➔➙➛➜➝➞➟➠➡➢➣➤➥➦➧➨➩➪➭➮➯➱⬛⬜⬝⬞⬟⬠⬡⬢⬣⬤⬥⬦⬧⬨⬩⭐⭑⭒⭓⭔⭕"
    "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳⒜⒝⒞⒟⒠ⓐⓑⓒⓓⓔⒶⒷⒸⒹⒺ❶❷❸❹❺➀➁➂➃➄"
)

# A list label, e.g. "1." "12)" "(3)" "[4]" "a." "b)" "(c)" "iv." "IV)"
_NUMBER_PATTERN = re.compile(
    r"^("
    r"\d{1,3}[\.\)]\s?"          # 1.  12)
    r"|\(\d{1,3}\)\s?"           # (3)
    r"|\[\d{1,3}\]\s?"           # [4]
    r"|[a-zA-Z][\.\)]\s?"        # a.  b)
    r"|\([a-zA-Z]\)\s?"          # (c)
    r"|[ivxlcdm]+[\.\)]\s?"      # iv.
    r"|[IVXLCDM]+[\.\)]\s?"      # IV)
    r")"
)

# Decimal section numbers ("1.1", "2.3.4") are NOT list labels — they are
# hierarchical section references. Reject before list-label splitting.
_DECIMAL_SECTION = re.compile(r"^\d+\.\d+")


def build_list_structure(
    elements: list[TaggedElement],
) -> list[TaggedElement]:
    """
    Convert flat LI elements into proper list structure.

    Groups consecutive LI elements on the same page into L (list)
    containers, and splits each LI into Lbl + LBody.

    Modifies the elements list in-place and returns it.
    """
    if not elements:
        return elements

    # Promote separated bare-marker lists (e.g. "1" "2" "3" markers in their own
    # elements, bodies in others) into merged LI elements before run grouping.
    elements[:] = _promote_bare_marker_lists(elements)

    # Find consecutive runs of LI elements on the same page
    runs: list[list[int]] = []
    current_run: list[int] = []

    for i, el in enumerate(elements):
        if el.pdf_tag == PDFTag.LI:
            if current_run and (
                elements[current_run[-1]].page_num != el.page_num
                or i - current_run[-1] > 1
            ):
                # Page break or non-consecutive → end current run
                if len(current_run) >= 1:
                    runs.append(current_run)
                current_run = [i]
            else:
                current_run.append(i)
        else:
            if current_run:
                if len(current_run) >= 1:
                    runs.append(current_run)
                current_run = []

    if current_run and len(current_run) >= 1:
        runs.append(current_run)

    # Process each run: split LI into Lbl + LBody
    for run in runs:
        for idx in run:
            el = elements[idx]
            # Skip items already split by the bare-marker promotion above
            if el.specialist_data and el.specialist_data.get("list_label"):
                continue
            label, body = _split_label_body(el.text)
            if label:
                # Store the split in specialist_data for writeback
                el.specialist_data = {
                    "list_label": label,
                    "list_body": body,
                }

    total_lists = len(runs)
    total_items = sum(len(r) for r in runs)

    if total_lists > 0:
        logger.info(
            "List builder: %d lists with %d total items",
            total_lists, total_items,
        )

    return elements


def _split_label_body(text: str) -> tuple[str | None, str]:
    """
    Split a list item's text into label and body.

    Returns (label, body) — label is None if no pattern matched.
    """
    if not text:
        return None, text

    stripped = text.lstrip()

    # Check for bullet characters
    if stripped and stripped[0] in _BULLET_CHARS:
        label = stripped[0]
        body = stripped[1:].lstrip()
        return label, body

    # Decimal section numbers ("1.1 Methods") are section references, not lists
    if _DECIMAL_SECTION.match(stripped):
        return None, text

    # Check for numbered patterns (1. , a) , iv. , etc.)
    match = _NUMBER_PATTERN.match(stripped)
    if match:
        label = match.group(1).rstrip()
        body = stripped[match.end():].lstrip()
        return label, body

    return None, text


def _bare_ordinal_value(text: str) -> tuple[str, int] | None:
    """
    Parse a bare list marker into (kind, value).

    Accepts "1", "1.", "1)", "a", "a.", "a)" (and uppercase). Returns
    ("int", n) or ("alpha", n) where n is the 1-based ordinal, else None.
    """
    t = (text or "").strip()
    if not t:
        return None
    core = t[:-1] if t[-1] in ".)" else t
    if core.isdigit():
        return ("int", int(core))
    if len(core) == 1 and core.isalpha():
        return ("alpha", ord(core.lower()) - ord("a") + 1)
    return None


def _promote_bare_marker_lists(
    elements: list[TaggedElement],
) -> list[TaggedElement]:
    """
    Detect lists whose markers ("1" "2" "3"…) are standalone elements separate
    from their body text, and merge each marker+body into one LI element.

    Three signals must hold together (any one alone produces false positives):
      1. marker is a short standalone element (<= 3 chars, bare ordinal)
      2. markers form a consecutive sequence starting at 1 (or 'a')
      3. markers share a left edge (within marker_height * _ALIGN_TOL_FRAC) and
         each binds to a body to its right at a consistent indent
    """
    by_page: dict[int, list[int]] = defaultdict(list)
    for i, el in enumerate(elements):
        by_page[el.page_num].append(i)

    consumed: set[int] = set()       # body indices merged away
    li_at: dict[int, TaggedElement] = {}  # marker index -> merged LI element

    for idxs in by_page.values():
        cands = [
            i for i in idxs
            if elements[i].pdf_tag == PDFTag.P and elements[i].bbox
            and len((elements[i].text or "").strip()) <= 3
            and _bare_ordinal_value(elements[i].text) is not None
        ]
        if len(cands) < 2:
            continue
        cands.sort(key=lambda i: elements[i].bbox[1])  # by vertical position

        used: set[int] = set()
        for s in range(len(cands)):
            si = cands[s]
            if si in used:
                continue
            kind, num = _bare_ordinal_value(elements[si].text)
            if num != 1:  # sequences must start at 1 / 'a'
                continue
            run = [si]
            prev = num
            for k in range(s + 1, len(cands)):
                ck = cands[k]
                if ck in used:
                    continue
                vk = _bare_ordinal_value(elements[ck].text)
                if vk and vk[0] == kind and vk[1] == prev + 1:
                    run.append(ck)
                    prev = vk[1]
            if len(run) < 2:
                continue

            heights = [elements[i].bbox[3] - elements[i].bbox[1] for i in run]
            tol = (sum(heights) / len(heights)) * _ALIGN_TOL_FRAC
            mxs = [elements[i].bbox[0] for i in run]
            if max(mxs) - min(mxs) > tol:  # markers must share a left edge
                continue

            pairs = []
            for i in run:
                b = _find_body_for_marker(elements[i], idxs, elements, consumed, used)
                if b is None:
                    break
                pairs.append((i, b))
            if len(pairs) != len(run):
                continue

            bxs = [elements[b].bbox[0] for _, b in pairs]
            if max(bxs) - min(bxs) > tol * 2:  # bodies must share a left edge
                continue

            for m, b in pairs:
                li_at[m] = _merge_marker_body(elements[m], elements[b])
                consumed.add(b)
                used.add(m)

    if not li_at:
        return elements

    out: list[TaggedElement] = []
    for i, el in enumerate(elements):
        if i in li_at:
            out.append(li_at[i])
        elif i not in consumed:
            out.append(el)
    logger.info("List builder: promoted %d bare-marker list items", len(li_at))
    return out


def _find_body_for_marker(
    marker: TaggedElement,
    page_idxs: list[int],
    elements: list[TaggedElement],
    consumed: set[int],
    used: set[int],
) -> int | None:
    """Find the body element a bare marker binds to: a longer P element to the
    marker's right, vertically nearest the marker's top."""
    mh = marker.bbox[3] - marker.bbox[1]
    best, best_d = None, None
    for j in page_idxs:
        if j in consumed or j in used:
            continue
        el = elements[j]
        if el.pdf_tag != PDFTag.P or not el.bbox:
            continue
        if len((el.text or "").strip()) <= 3:
            continue
        if el.bbox[0] <= marker.bbox[0] + mh:  # body must be clearly to the right
            continue
        d_top = abs(el.bbox[1] - marker.bbox[1])
        if d_top > 2 * mh:  # within ~2 lines vertically
            continue
        if best is None or d_top < best_d:
            best, best_d = j, d_top
    return best


def _merge_marker_body(marker: TaggedElement, body: TaggedElement) -> TaggedElement:
    """Merge a separated marker + body into one LI element with Lbl/LBody data."""
    label = (marker.text or "").strip()
    body_text = (body.text or "").strip()
    bbox = (
        min(marker.bbox[0], body.bbox[0]),
        min(marker.bbox[1], body.bbox[1]),
        max(marker.bbox[2], body.bbox[2]),
        max(marker.bbox[3], body.bbox[3]),
    )
    return TaggedElement(
        element_id=body.element_id,
        page_num=body.page_num,
        pdf_tag=PDFTag.LI,
        text=f"{label} {body_text}",
        bbox=bbox,
        confidence=min(marker.confidence, body.confidence),
        original_mcid=body.original_mcid,
        font_name=body.font_name,
        font_size=body.font_size,
        font_weight=body.font_weight,
        merged_from=list(marker.merged_from) + list(body.merged_from),
        layout_category=body.layout_category,
        specialist_data={"list_label": label, "list_body": body_text},
    )
