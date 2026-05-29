"""NID reading-order similarity (replicated from opendataloader-bench, MIT).

NID = rapidfuzz indel-based ratio over whitespace-normalized markdown (tables
converted to HTML). NID-S strips HTML tables first, scoring narrative order only.
"""
from __future__ import annotations

import re
from typing import Optional, Tuple

from rapidfuzz import fuzz

from tagger.benchmark.dpbench.converter import convert_to_markdown_with_html_tables

_HTML_TABLE_PATTERN = re.compile(r"<table[^>]*?>.*?</table>", re.IGNORECASE | re.DOTALL)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _strip_tables(text: str) -> str:
    return _HTML_TABLE_PATTERN.sub(" ", text)


def evaluate_reading_order(
    gt: str, pred: str
) -> Tuple[Optional[float], Optional[float]]:
    gt_with_html = convert_to_markdown_with_html_tables(gt)
    gt_normalized = _normalize(gt_with_html or "")
    gt_stripped_normalized = _normalize(_strip_tables(gt_with_html or ""))
    if not gt_normalized:
        return None, None

    pred_with_html = convert_to_markdown_with_html_tables(pred)
    pred_normalized = _normalize(pred_with_html or "")
    pred_stripped_normalized = _normalize(_strip_tables(pred_with_html or ""))

    nid_score = fuzz.ratio(gt_normalized, pred_normalized) / 100.0
    nid_s_score = fuzz.ratio(gt_stripped_normalized, pred_stripped_normalized) / 100.0
    return nid_score, nid_s_score


__all__ = ["evaluate_reading_order"]
