"""
Stage 2 — Text merger.

Converts individual character-level PageElements into merged text blocks:
    chars → words → lines → paragraphs

This directly addresses the #1 bug found in PREP comparison:
text fragmentation where single words or even characters were tagged
as separate elements.

Three-pass merge:
  Pass 1 (chars → words): Horizontal adjacency within font-size tolerance
  Pass 2 (words → lines): Same Y-band, sorted by X
  Pass 3 (lines → paragraphs): Vertical gap analysis

All merges preserve the dominant font metadata and accumulate
`merged_from` provenance lists.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from tagger.config import TEXT_MERGER
from tagger.models.data_types import PageElement

logger = logging.getLogger(__name__)


def merge_chars_to_words(
    chars: list[PageElement],
    page_num: int,
) -> list[PageElement]:
    """
    Pass 1: Merge horizontally adjacent characters into words.

    Characters on the same line (overlapping Y) with horizontal gaps
    smaller than `word_gap_multiplier * avg_char_width` are merged.

    Args:
        chars: Character-level PageElements from Stage 1, sorted by appearance.
        page_num: Page number for element ID generation.

    Returns:
        List of word-level PageElements.
    """
    if not chars:
        return []

    # Sort by vertical position first, then horizontal
    sorted_chars = sorted(chars, key=lambda c: (c.bbox[1], c.bbox[0]))

    # Group into horizontal runs on the same line
    lines: list[list[PageElement]] = []
    current_line: list[PageElement] = [sorted_chars[0]]

    for ch in sorted_chars[1:]:
        prev = current_line[-1]
        # Check vertical overlap
        overlap = _vertical_overlap_ratio(prev.bbox, ch.bbox)
        if overlap >= TEXT_MERGER.line_overlap_threshold:
            current_line.append(ch)
        else:
            lines.append(current_line)
            current_line = [ch]
    lines.append(current_line)

    # Within each line, merge adjacent chars into words
    # We detect word boundaries by looking for gaps larger than
    # the typical intra-character gap (space-width heuristic)
    words: list[PageElement] = []
    word_counter = 0

    for line_chars in lines:
        # Sort by X position within the line
        line_chars.sort(key=lambda c: c.bbox[0])

        if not line_chars:
            continue

        # Compute average char width for this line
        avg_char_width = _avg_width(line_chars)
        max_gap = avg_char_width * TEXT_MERGER.word_gap_multiplier

        # Threshold for inserting a space between chars:
        # Gaps larger than ~30% of avg char width are treated as
        # word boundaries (a space in most fonts is ~30-50% of char width)
        space_threshold = avg_char_width * 0.3

        current_word_chars: list[PageElement] = [line_chars[0]]
        word_boundaries: list[bool] = []  # True = space before this char

        for ch in line_chars[1:]:
            prev = current_word_chars[-1] if current_word_chars else None
            gap = ch.bbox[0] - prev.bbox[2] if prev else 0  # horizontal gap

            if gap > max_gap:
                # Large gap — this is a new word entirely
                word = _merge_elements_with_spaces(
                    current_word_chars,
                    word_boundaries,
                    element_id=f"p{page_num}_w{word_counter}",
                    page_num=page_num,
                )
                words.append(word)
                word_counter += 1
                current_word_chars = [ch]
                word_boundaries = []
            else:
                # Same word group — but check if we need a space
                word_boundaries.append(gap > space_threshold)
                current_word_chars.append(ch)

        # Emit final word in line
        if current_word_chars:
            word = _merge_elements_with_spaces(
                current_word_chars,
                word_boundaries,
                element_id=f"p{page_num}_w{word_counter}",
                page_num=page_num,
            )
            words.append(word)
            word_counter += 1

    logger.debug(
        "Page %d: %d chars → %d words",
        page_num, len(chars), len(words),
    )
    return words


def merge_words_to_lines(
    words: list[PageElement],
    page_num: int,
) -> list[PageElement]:
    """
    Pass 2: Merge words on the same visual line.

    Words with overlapping Y-bands are grouped into lines.
    Within each line, words are ordered by X position and joined
    with spaces.

    Args:
        words: Word-level PageElements from Pass 1.
        page_num: Page number for element ID generation.

    Returns:
        List of line-level PageElements.
    """
    if not words:
        return []

    # Sort by vertical center, then horizontal
    sorted_words = sorted(words, key=lambda w: (w.center_y, w.bbox[0]))

    # Group into lines by vertical overlap
    lines: list[list[PageElement]] = []
    current_line: list[PageElement] = [sorted_words[0]]

    for word in sorted_words[1:]:
        # Check if this word overlaps vertically with the current line
        line_bbox = _union_bbox([w.bbox for w in current_line])
        overlap = _vertical_overlap_ratio(line_bbox, word.bbox)

        if overlap >= TEXT_MERGER.line_overlap_threshold:
            current_line.append(word)
        else:
            lines.append(current_line)
            current_line = [word]
    lines.append(current_line)

    # Merge each line's words into a single element, splitting at large horizontal gaps
    merged_lines: list[PageElement] = []
    line_counter = 0

    for line_words in lines:
        line_words.sort(key=lambda w: w.bbox[0])
        if not line_words:
            continue

        total_chars = sum(len(w.text) for w in line_words)
        total_width = sum(w.width for w in line_words)
        avg_char_width = (total_width / total_chars) if total_chars > 0 else 1.0
        
        max_gap = avg_char_width * TEXT_MERGER.line_gap_multiplier

        current_chunk: list[PageElement] = [line_words[0]]

        for word in line_words[1:]:
            prev = current_chunk[-1]
            gap = word.bbox[0] - prev.bbox[2]

            if gap > max_gap:
                line_elem = _merge_elements(
                    current_chunk,
                    element_id=f"p{page_num}_l{line_counter}",
                    page_num=page_num,
                    join_with=" ",
                )
                merged_lines.append(line_elem)
                line_counter += 1
                current_chunk = [word]
            else:
                current_chunk.append(word)

        if current_chunk:
            line_elem = _merge_elements(
                current_chunk,
                element_id=f"p{page_num}_l{line_counter}",
                page_num=page_num,
                join_with=" ",
            )
            merged_lines.append(line_elem)
            line_counter += 1

    logger.debug(
        "Page %d: %d words → %d line fragments",
        page_num, len(words), len(merged_lines),
    )
    return merged_lines


def merge_page_elements(
    chars: list[PageElement],
    page_num: int,
) -> list[PageElement]:
    """
    Two-pass merge: chars → words → lines.

    This is the main entry point for Stage 2.

    Args:
        chars: Character-level PageElements from Stage 1.
        page_num: Page number.

    Returns:
        List of line-level PageElements.
    """
    words = merge_chars_to_words(chars, page_num)
    lines = merge_words_to_lines(words, page_num)

    logger.info(
        "Page %d: merged %d chars → %d words → %d line fragments",
        page_num, len(chars), len(words), len(lines),
    )
    return lines


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _merge_elements_with_spaces(
    elements: list[PageElement],
    word_boundaries: list[bool],
    element_id: str,
    page_num: int,
) -> PageElement:
    """
    Merge elements with intelligent space insertion.

    word_boundaries[i] indicates whether a space should be inserted
    before elements[i+1] (i.e., between elements[i] and elements[i+1]).
    """
    if len(elements) == 1:
        el = elements[0]
        return PageElement(
            element_id=element_id,
            page_num=page_num,
            text=el.text,
            bbox=el.bbox,
            font_name=el.font_name,
            font_size=el.font_size,
            font_weight=el.font_weight,
            font_color=el.font_color,
            is_italic=el.is_italic,
            source=el.source,
            confidence=el.confidence,
            mcid=el.mcid,
            merged_from=list(el.merged_from) if el.merged_from else [el.element_id],
        )

    # Build text with spaces at word boundaries
    parts = [elements[0].text]
    for i, el in enumerate(elements[1:]):
        if i < len(word_boundaries) and word_boundaries[i]:
            parts.append(" ")
        parts.append(el.text)
    text = "".join(parts)

    # Rest is same as _merge_elements
    bbox = _union_bbox([el.bbox for el in elements])
    font_name = _most_common([el.font_name for el in elements if el.font_name])
    font_size = _dominant_font_size(elements)
    font_weight = "bold" if any(el.font_weight == "bold" for el in elements) else "normal"
    font_color = _most_common([el.font_color for el in elements if el.font_color])
    is_italic = any(el.is_italic for el in elements)
    source = "mineru_ocr" if any(el.source == "mineru_ocr" for el in elements) else "pdfplumber"
    confidence = min(el.confidence for el in elements)
    mcids = {el.mcid for el in elements if el.mcid is not None}
    mcid = mcids.pop() if len(mcids) == 1 else None

    merged_from: list[str] = []
    for el in elements:
        if el.merged_from:
            merged_from.extend(el.merged_from)
        else:
            merged_from.append(el.element_id)

    return PageElement(
        element_id=element_id,
        page_num=page_num,
        text=text,
        bbox=bbox,
        font_name=font_name,
        font_size=font_size,
        font_weight=font_weight,
        font_color=font_color,
        is_italic=is_italic,
        source=source,
        confidence=confidence,
        mcid=mcid,
        merged_from=merged_from,
    )


def _merge_elements(
    elements: list[PageElement],
    element_id: str,
    page_num: int,
    join_with: str = "",
) -> PageElement:
    """
    Merge multiple PageElements into a single one.

    Text is concatenated (with optional separator).
    Bbox is the union of all bboxes.
    Font metadata is taken from the dominant (most common) element.
    """
    if len(elements) == 1:
        el = elements[0]
        return PageElement(
            element_id=element_id,
            page_num=page_num,
            text=el.text,
            bbox=el.bbox,
            font_name=el.font_name,
            font_size=el.font_size,
            font_weight=el.font_weight,
            font_color=el.font_color,
            is_italic=el.is_italic,
            source=el.source,
            confidence=el.confidence,
            mcid=el.mcid,
            merged_from=list(el.merged_from) if el.merged_from else [el.element_id],
        )

    # Concatenate text
    text = join_with.join(el.text for el in elements)

    # Union bbox
    bbox = _union_bbox([el.bbox for el in elements])

    # Dominant font metadata (most common font_name)
    font_name = _most_common([el.font_name for el in elements if el.font_name])
    font_size = _dominant_font_size(elements)
    font_weight = "bold" if any(el.font_weight == "bold" for el in elements) else "normal"
    font_color = _most_common([el.font_color for el in elements if el.font_color])
    is_italic = any(el.is_italic for el in elements)

    # Source: if any element came from OCR, mark as OCR
    source = "mineru_ocr" if any(el.source == "mineru_ocr" for el in elements) else "pdfplumber"

    # Confidence: minimum across merged elements
    confidence = min(el.confidence for el in elements)

    # MCID: only if all elements have the same MCID
    mcids = {el.mcid for el in elements if el.mcid is not None}
    mcid = mcids.pop() if len(mcids) == 1 else None

    # Provenance
    merged_from: list[str] = []
    for el in elements:
        if el.merged_from:
            merged_from.extend(el.merged_from)
        else:
            merged_from.append(el.element_id)

    return PageElement(
        element_id=element_id,
        page_num=page_num,
        text=text,
        bbox=bbox,
        font_name=font_name,
        font_size=font_size,
        font_weight=font_weight,
        font_color=font_color,
        is_italic=is_italic,
        source=source,
        confidence=confidence,
        mcid=mcid,
        merged_from=merged_from,
    )


def _vertical_overlap_ratio(
    bbox_a: tuple[float, float, float, float],
    bbox_b: tuple[float, float, float, float],
) -> float:
    """
    Compute the fraction of vertical overlap between two bboxes.

    Returns 0.0 if no overlap, 1.0 if one fully contains the other vertically.
    """
    top = max(bbox_a[1], bbox_b[1])
    bottom = min(bbox_a[3], bbox_b[3])

    if bottom <= top:
        return 0.0

    overlap_height = bottom - top
    min_height = min(bbox_a[3] - bbox_a[1], bbox_b[3] - bbox_b[1])

    if min_height <= 0:
        return 0.0

    return overlap_height / min_height


def _union_bbox(
    bboxes: list[tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    """Compute the bounding box that contains all given bboxes."""
    x0 = min(b[0] for b in bboxes)
    y0 = min(b[1] for b in bboxes)
    x1 = max(b[2] for b in bboxes)
    y1 = max(b[3] for b in bboxes)
    return (x0, y0, x1, y1)


def _avg_width(elements: list[PageElement]) -> float:
    """Average width of elements."""
    widths = [el.width for el in elements if el.width > 0]
    return sum(widths) / len(widths) if widths else 1.0


def _median(values: list[float]) -> float:
    """Simple median calculation."""
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n % 2 == 0:
        return (s[n // 2 - 1] + s[n // 2]) / 2.0
    return s[n // 2]


def _most_common(values: list) -> any:
    """Return the most common non-None value from a list."""
    if not values:
        return None
    counts: dict = {}
    for v in values:
        if v is not None:
            counts[v] = counts.get(v, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def _dominant_font_size(elements: list[PageElement]) -> float | None:
    """
    Return the most common font size, weighted by text length.

    Longer text content has more influence on the "dominant" size.
    """
    size_weights: dict[float, int] = defaultdict(int)
    for el in elements:
        if el.font_size is not None:
            size_weights[el.font_size] += len(el.text)

    if not size_weights:
        return None

    return max(size_weights, key=size_weights.get)
