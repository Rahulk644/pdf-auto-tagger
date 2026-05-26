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

from tagger.models.data_types import PDFTag, TaggedElement

logger = logging.getLogger(__name__)

# Patterns to split label from body
_BULLET_CHARS = set("•‣◦⁃∙◆■□▪▸►▻–—-")
_NUMBER_PATTERN = re.compile(
    r"^(\d{1,3}[\.\)]\s?|[a-zA-Z][\.\)]\s?|[ivxlcdm]+[\.\)]\s?|[IVXLCDM]+[\.\)]\s?)"
)


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

    # Check for numbered patterns (1. , a) , iv. , etc.)
    match = _NUMBER_PATTERN.match(stripped)
    if match:
        label = match.group(1).rstrip()
        body = stripped[match.end():].lstrip()
        return label, body

    return None, text
