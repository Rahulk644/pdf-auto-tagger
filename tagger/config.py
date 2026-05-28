"""
Centralized configuration for the PDF auto-tagger pipeline.

All thresholds, model paths, DPI settings, and confidence cutoffs live here.
Import from this module — never hardcode magic numbers in stage code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# DPI & coordinate system
# ---------------------------------------------------------------------------

STANDARD_DPI: int = 150
"""Every bbox in the pipeline is normalized to this DPI, origin top-left."""

PDF_NATIVE_DPI: int = 72
"""PDF user-space units are 1/72 inch."""


# ---------------------------------------------------------------------------
# Stage 0 — Page classifier thresholds
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PageClassifierConfig:
    """Heuristic thresholds for native / scanned / mixed / corrupt detection."""

    # Minimum extractable characters for "native" classification
    min_native_chars: int = 50

    # If char count is between 1 and this, page may be "mixed"
    mixed_char_upper: int = 50

    # Fraction of page area covered by image XObjects
    # Above this → likely scanned
    scanned_image_coverage: float = 0.70

    # Below this → likely native (text-dominant)
    native_image_coverage: float = 0.30

    # Minimum fraction of characters that must be valid Unicode
    # Below this for non-zero char pages → corrupt or garbled OCR
    min_unicode_validity: float = 0.95

    # Character density (chars / page_area_sq_inches)
    # Pages with density below this AND some chars → mixed
    min_char_density: float = 0.001


PAGE_CLASSIFIER = PageClassifierConfig()


# ---------------------------------------------------------------------------
# Stage 2 — Text merger thresholds
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TextMergerConfig:
    """Controls how characters are grouped into words, lines, paragraphs."""

    # Horizontal gap (in standard-DPI points) below which adjacent chars
    # are merged into the same word.  Expressed as a multiplier of the
    # average character width on the line.
    word_gap_multiplier: float = 1.0

    # Vertical overlap fraction required for two words to be on the
    # "same line".  1.0 = perfect overlap, 0.5 = half overlap.
    line_overlap_threshold: float = 0.5

    # Horizontal gap multiplier for words on the same line. If the gap between
    # two words exceeds this multiple of the average character width, they are
    # split into separate line elements (e.g. separate table cells).
    line_gap_multiplier: float = 3.0

    # Tags that should NEVER be merged across MCIDs
    # (table cells, list labels, etc.)
    no_merge_tags: frozenset[str] = field(default_factory=lambda: frozenset({
        "TD", "TH", "TR", "Lbl", "THead", "TBody", "TFoot",
    }))

    # pdfplumber extract_words settings
    use_text_flow: bool = True
    keep_blank_chars: bool = False
    x_tolerance: int = 3
    y_tolerance: int = 3


TEXT_MERGER = TextMergerConfig()


# ---------------------------------------------------------------------------
# Stage 3 — Layout detection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LayoutConfig:
    """MinerU2.5 layout detection settings."""

    model_name: str = "opendatalab/MinerU2.5-2509-1.2B"

    # Categories MinerU outputs (canonical names)
    categories: tuple[str, ...] = (
        "Title",
        "Section-header",
        "Text",
        "List-item",
        "Table",
        "Formula",
        "Picture",
        "Caption",
        "Footnote",
        "Page-header",
        "Page-footer",
    )

    # Minimum confidence to accept a detected region
    min_region_confidence: float = 0.5


LAYOUT = LayoutConfig()


# ---------------------------------------------------------------------------
# Stage 5 — Specialist models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TableConfig:
    """Table extraction settings."""

    # Use pdfplumber for native PDFs
    use_pdfplumber_for_native: bool = True

    # Fall back to MinerU's built-in table extraction for scanned
    use_mineru_for_scanned: bool = True

    # Minimum cells for a valid table (below → likely misclassified paragraph)
    min_cells: int = 2

    # Minimum dimensions (standard DPI) to consider a table region
    min_width_px: int = 40
    min_height_px: int = 20


TABLE = TableConfig()


@dataclass(frozen=True)
class FormulaConfig:
    """Formula extraction via UniMERNet."""

    model_name: str = "wanderkid/unimernet_base"

    # Empty or invalid LaTeX fallback tag
    fallback_tag: str = "P"


FORMULA = FormulaConfig()


# ---------------------------------------------------------------------------
# Stage 6 — Consistency validator
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ValidatorConfig:
    """Thresholds for the consistency validator rule engine."""

    # Elements failing validation get this confidence cap
    failed_confidence_cap: float = 0.55

    # Minimum figure dimensions (standard DPI) — below → Artifact
    min_figure_width_px: int = 20
    min_figure_height_px: int = 20

    # IoU threshold for detecting overlapping regions
    overlap_iou_threshold: float = 0.80


VALIDATOR = ValidatorConfig()


# ---------------------------------------------------------------------------
# Stage 7 — Cross-page merge
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CrossPageConfig:
    """Heuristics for detecting elements spanning page boundaries."""

    # Column width tolerance for table continuation (fraction)
    table_column_width_tolerance: float = 0.05

    # Confidence assigned to cross-page merged elements
    cross_page_confidence: float = 0.70


CROSS_PAGE = CrossPageConfig()


# ---------------------------------------------------------------------------
# Stage 8 — Semantic refinement
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SemanticConfig:
    """Controls for heading ranking, TOC, artifact, caption detection."""

    # Heading font-size tolerance: sizes within this many points are same level
    heading_size_tolerance_pt: float = 1.0

    # Maximum heading levels (H1–H6)
    max_heading_levels: int = 6

    # Heading rarity: a font size occurring in more than this fraction of all
    # document elements is treated as body text, not a distinct heading level.
    heading_body_frequency_fraction: float = 0.10

    # TOC: pages in the first N% of the document are candidates
    toc_page_fraction: float = 0.10

    # Artifact: text must appear at same Y-position on at least N pages
    artifact_min_page_occurrences: int = 3

    # Artifact: Y-position tolerance (standard DPI points)
    artifact_y_tolerance_px: float = 5.0

    # Caption regex patterns
    caption_patterns: tuple[str, ...] = (
        r"^(Figure|Fig\.?|Table|Tbl\.?)\s*\d+",
        r"^(Exhibit|Chart|Diagram|Graph|Plate)\s*\d+",
    )


SEMANTIC = SemanticConfig()


# ---------------------------------------------------------------------------
# Stage 9 — Alt text generation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AltTextConfig:
    """Qwen2.5-VL-7B settings for figure alt text."""

    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"

    # Maximum tokens for alt text output
    max_output_tokens: int = 150

    # Temperature for alt text generation
    temperature: float = 0.3

    # System prompt
    system_prompt: str = (
        "You are an accessibility expert. Describe this figure for a "
        "screen reader user. Be concise (1-2 sentences), factual, and "
        "focus on what the figure communicates. Do not describe decorative "
        "elements. Do not start with 'This figure shows'."
    )


ALT_TEXT = AltTextConfig()


# ---------------------------------------------------------------------------
# Stage 10 — Struct tree writeback
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WritebackConfig:
    """pikepdf struct tree writeback settings."""

    # PDF tag role map — maps our canonical tags to PDF struct types
    tag_role_map: dict[str, str] = field(default_factory=lambda: {
        "H1": "H1", "H2": "H2", "H3": "H3", "H4": "H4", "H5": "H5", "H6": "H6",
        "P": "P", "Span": "Span",
        "L": "L", "LI": "LI", "Lbl": "Lbl", "LBody": "LBody",
        "Table": "Table", "TR": "TR", "TH": "TH", "TD": "TD",
        "THead": "THead", "TBody": "TBody", "TFoot": "TFoot",
        "Figure": "Figure", "Formula": "Formula",
        "Caption": "Caption",
        "TOC": "TOC", "TOCI": "TOCI",
        "BlockQuote": "BlockQuote", "Quote": "Quote",
        "Note": "Note", "Reference": "Reference",
        "Code": "Code",
        "Artifact": "Artifact",
    })


WRITEBACK = WritebackConfig()


# ---------------------------------------------------------------------------
# Pipeline-wide
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PipelineConfig:
    """Top-level pipeline settings."""

    # Confidence threshold below which elements are flagged for human review
    review_threshold: float = 0.60

    # Default output directory
    default_output_dir: Path = Path("output")

    # Flask API settings
    flask_host: str = "0.0.0.0"
    flask_port: int = 5002  # 5001 is PREP-QA-Tool
    flask_debug: bool = False

    # Structural-repair gating (see stage10_writeback/repair_gate.py). Additive
    # tagging always runs; these control only the source-modifying font repairs.
    #   "auto"      — apply all modifying repairs (default; differentiator vs PREP)
    #   "confirm"   — apply only repairs whose finding_id is in repair_approval_file
    #   "flag-only" — never apply; only report them
    repair_mode: str = "auto"
    repair_approval_file: Path | None = None


PIPELINE = PipelineConfig()
