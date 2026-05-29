"""MHS heading-structure similarity (replicated from opendataloader-bench, MIT).

Parses markdown into a flat tree (headings under root, content under nearest
heading) and compares via APTED tree-edit distance. Heading LEVELS are treated as
equivalent (tag "heading" regardless of #/##) — only heading-vs-paragraph
segmentation + text matter. MHS includes cell-stripped text; MHS-S is structure
only. Tables are excluded from text comparison.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from rapidfuzz.distance import Levenshtein
from apted import APTED, Config
from apted.helpers import Tree

from tagger.benchmark.dpbench.converter import convert_to_markdown_with_html_tables

_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.*)$")


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


class HeadingTree(Tree):
    def __init__(
        self, tag: str, text: Optional[str] = None, *children: "HeadingTree"
    ) -> None:
        self.tag = tag
        self.text = text
        self.children = list(children)


class HeadingConfig(Config):
    def __init__(self, include_text: bool) -> None:
        self.include_text = include_text

    @staticmethod
    def _normalized_distance(text_a: str, text_b: str) -> float:
        if not text_a and not text_b:
            return 0.0
        length = max(len(text_a), len(text_b), 1)
        return Levenshtein.distance(text_a, text_b) / float(length)

    def rename(self, node1: HeadingTree, node2: HeadingTree) -> float:
        if node1.tag != node2.tag:
            return 1.0
        if not self.include_text:
            return 0.0
        return self._normalized_distance(node1.text or "", node2.text or "")


def _flush_content(content_lines: List[str], parent: HeadingTree) -> None:
    if not content_lines:
        return
    content_text = _normalize_text(" ".join(content_lines))
    if not content_text:
        content_lines.clear()
        return
    parent.children.append(HeadingTree("content", content_text))
    content_lines.clear()


def _parse_markdown_structure(markdown: Optional[str]) -> HeadingTree:
    root = HeadingTree("document")
    if not markdown:
        return root

    current_container = root
    pending_lines: List[str] = []
    for raw_line in markdown.splitlines():
        match = _HEADING_PATTERN.match(raw_line)
        if match:
            _flush_content(pending_lines, current_container)
            heading_node = HeadingTree("heading", _normalize_text(match.group(2)))
            root.children.append(heading_node)
            current_container = heading_node
            continue
        normalized = _normalize_text(raw_line)
        if normalized:
            pending_lines.append(normalized)

    _flush_content(pending_lines, current_container)
    return root


def _count_nodes(node: HeadingTree) -> int:
    return 1 + sum(_count_nodes(child) for child in node.children)


def _compute_edit_distance(
    tree_a: HeadingTree, tree_b: HeadingTree, include_text: bool
) -> float:
    config = HeadingConfig(include_text=include_text)
    return float(APTED(tree_a, tree_b, config).compute_edit_distance())


def evaluate_heading_level(
    gt: Optional[str], pred: Optional[str]
) -> Tuple[Optional[float], Optional[float]]:
    gt_with_html = convert_to_markdown_with_html_tables(gt)
    pred_with_html = convert_to_markdown_with_html_tables(pred)

    gt_tree = _parse_markdown_structure(gt_with_html)
    if not any(child.tag == "heading" for child in gt_tree.children):
        return None, None

    pred_tree = _parse_markdown_structure(pred_with_html)
    if not any(child.tag == "heading" for child in pred_tree.children):
        return 0.0, 0.0

    max_nodes = max(_count_nodes(gt_tree), _count_nodes(pred_tree), 1)
    edit_with_text = _compute_edit_distance(gt_tree, pred_tree, include_text=True)
    edit_structure_only = _compute_edit_distance(gt_tree, pred_tree, include_text=False)

    mhs = max(0.0, min(1.0, 1.0 - (edit_with_text / max_nodes)))
    mhs_s = max(0.0, min(1.0, 1.0 - (edit_structure_only / max_nodes)))
    return mhs, mhs_s


__all__ = ["evaluate_heading_level"]
